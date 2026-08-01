"""Uniform per-episode (success, final distance, distance trajectory) loader.

Why this exists. The trajectory-quality results in the paper's "The price of a step
size" section are rescored from evaluation output that was already on disk -- every
replan logs each episode's state distance to the goal -- so they cost no new planning.
The obstacle is that eval_outputs/ accumulated three different directory conventions
over the project, and a per-episode comparison has to read all three and align them.

Alignment note. Batched runs and episode-isolated trials are paired by index: batched
column i and trial<i> use the same entry of `eval_seed` (see plan.py), so they are the
same episode. scripts/paired_indist_test.py already relies on this, and the paired
p-values in the paper reproduce under it.

Termination note. Success is absorbing (planning/mpc.py masks succeeded episodes'
actions to zero), so a successful episode's distance is held at its value on the
replan it succeeded. Trajectories from isolated trials are therefore forward-filled
to a common length, matching what the batched runs record.

Handles the three directory conventions in eval_outputs/:
  A. shape-sharded isolated trials   <run>/<shape>_trials/trial<NNN>/logs.json
  B. flat isolated trials            <run>/trial<NNN>/logs.json
  C. batched columns                 <run>/logs/<shape>.log   (or <run>/<shape>/plan.log)
"""
import re, json, os, glob, numpy as np

NUM = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
FIELD = re.compile(r"'(success|state_dist)':\s*array\(\[(.*?)\](?:,\s*dtype=\w+)?\)", re.S)

def _floats(s): return np.array([float(x) for x in NUM.findall(s)])
def _bools(s):  return np.array([1.0 if t == 'True' else 0.0 for t in re.findall(r'True|False', s)])

def _batched_log(path):
    txt = open(path, errors='ignore').read()
    S, D, pend = [], [], None
    for m in FIELD.finditer(txt):
        if m.group(1) == 'success':
            pend = _bools(m.group(2))
        elif pend is not None:
            d = _floats(m.group(2))
            if len(d) == len(pend): S.append(pend); D.append(d)
            pend = None
    if not S: return None
    L = min(len(x) for x in S)
    return np.array([x[:L] for x in S]), np.array([x[:L] for x in D])   # (T,n)

def _trial_dir(d):
    """one trial dir -> (success, final_dist, traj list) or None"""
    lp = os.path.join(d, 'logs.json')
    if not os.path.exists(lp): return None
    succ = fd = None; per = []
    for line in open(lp):
        try: dd = json.loads(line)
        except Exception: continue
        if 'final_eval/success_rate' in dd:
            succ = float(dd['final_eval/success_rate']) > 0.5
            fd = float(dd.get('final_eval/mean_state_dist', np.nan))
        elif 'mpc/mean_state_dist' in dd:
            per.append(float(dd['mpc/mean_state_dist']))
    if succ is None: return None
    return succ, fd, per

def _collect_trials(root):
    """root containing trial* dirs -> (succ[n], fd[n], traj[n,T]) padded by holding."""
    tr = {}
    for d in glob.glob(os.path.join(root, 'trial*')):
        r = _trial_dir(d)
        if r is None: continue
        tr[int(os.path.basename(d)[5:])] = r
    if not tr: return None
    n = max(tr) + 1
    L = max((len(v[2]) for v in tr.values() if v[2]), default=1)
    succ = np.zeros(n, bool); fd = np.full(n, np.nan); traj = np.full((n, L), np.nan)
    for i, (s, f, p) in tr.items():
        succ[i] = s; fd[i] = f
        if p: traj[i] = list(p) + [p[-1]] * (L - len(p))
        else: traj[i] = f
    return succ, fd, traj

def load_run(run_dir, shapes=None):
    """-> (succ[n], final_dist[n], traj[n,T]) concatenated over shapes in order."""
    # B: flat trials
    if glob.glob(os.path.join(run_dir, 'trial*')):
        return _collect_trials(run_dir)
    parts = []
    for sh in (shapes or []):
        # A: shape-sharded trials
        r = _collect_trials(os.path.join(run_dir, f'{sh}_trials'))
        if r is not None: parts.append(r); continue
        # C: batched
        for cand in (os.path.join(run_dir, 'logs', f'{sh}.log'),
                     os.path.join(run_dir, sh, 'plan.log')):
            if os.path.exists(cand):
                b = _batched_log(cand)
                if b is not None:
                    S, D = b
                    parts.append((S[-1].astype(bool), D[-1], D.T))
                break
        else:
            return None
    if not parts or len(parts) != len(shapes or []): return None
    L = min(p[2].shape[1] for p in parts)
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2][:, :L] for p in parts]))
