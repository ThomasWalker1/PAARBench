#!/usr/bin/env python3
"""Rescore every online-adaptation grid cell on trajectory quality, all settings.

Reads only what is already in eval_outputs/ -- no planning, no GPU. Writes
paper/floats/{safety_span,grid_distance,wrongguess}.tex and a stats.json cache.

    .venv/bin/python scripts/rescore_distance.py

Statistics. The metric is heavy-tailed (distances of 1e4 occur under the frozen model
itself), so every reported statistic is a median or a quantile and every test is a
paired Wilcoxon signed-rank over identical episodes. Do not report means or standard
deviations off this metric: they are determined by two or three episodes.

Determinism. conf/planner/mpc_gd.yaml sets sample_type=zero and action_noise=0, so the
MPC inner loop is deterministic and cross-cell differences on a fixed episode are
effects of the hyperparameter rather than planner noise.

HyperJEPA epochs are the ones the paper's protocol selects -- argmax success on each
setting's own selection cohort (PushObj ref_ep2, PushT hyper_ep3, PointMaze hyper_ep6).
"""
import json, sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from episode_outcomes import load_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATS = ROOT / 'paper/floats/distance_stats.json'
OUT = str(ROOT / 'paper/floats')



LRS = ['5e-4', '2e-3', '1e-2', '5e-2']; STEPS = ['1', '3', '5', '10']
PUSH4 = ['L', 'T', 'Z', 'plus']; OOD3 = ['I', 'small_tee', 'square']; PT3 = ['L', 'T', 'Z']

SETTINGS = {
 'pushobj': dict(shapes=PUSH4, seed='seed300',
    cell=lambda fam, s, lr: f'eval_outputs/e1_grid/e1_{fam}_s{s}_lr{lr}/seed300',
    frozen='eval_outputs/indist_matrix/frozen/seed300',
    hyper='eval_outputs/ablation_grid/ref_ep2/seed300', fams=['dense', 'lora']),
 'pushobj_ood': dict(shapes=OOD3, seed='seed100',
    cell=lambda fam, s, lr: f'eval_outputs/e1_grid_heldout/e1_{fam}_s{s}_lr{lr}/seed100',
    frozen='eval_outputs/heldout_matrix/frozen/seed100',
    hyper='eval_outputs/heldout_matrix/hyper_d0_zeroshot/seed100', fams=['dense']),
 'pusht': dict(shapes=PT3, seed='seed100',
    cell=lambda fam, s, lr: f'eval_outputs/e1_grid_pvs/e1_{fam}_s{s}_lr{lr}/seed100',
    frozen='eval_outputs/e1_grid_pvs/frozen_pvs/seed100',
    hyper='eval_outputs/pvs_matrix/hyper_ep3/seed100', fams=['dense', 'lora']),
 'pointmaze': dict(shapes=None, seed='seed300',
    cell=lambda fam, s, lr: f'eval_outputs/d2_pointmaze/ada_{fam}_s{s}_lr{lr}/seed300',
    frozen='eval_outputs/d2_pointmaze/frozen/seed300',
    hyper='eval_outputs/d2_pointmaze/hyper_ep6/seed300', fams=['dense', 'lora']),
}

def wilcoxon(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x[x != 0]
    n = len(x)
    if n < 6: return float('nan')
    r = np.argsort(np.argsort(np.abs(x))) + 1.0
    W = min(r[x > 0].sum(), r[x < 0].sum())
    mu = n * (n + 1) / 4; sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return 2 * 0.5 * (1 + erf(((W - mu) / sd) / sqrt(2)))

def summarize(s, d, fd=None, traj=None):
    o = dict(n=int(len(s)), success=float(s.mean()),
             med=float(np.nanmedian(d)), mean=float(np.nanmean(d)),
             p90=float(np.nanpercentile(d, 90)), mx=float(np.nanmax(d)),
             iqr=float(np.subtract(*np.nanpercentile(d, [75, 25]))))
    if fd is not None:
        D = d - fd; r = d / np.maximum(fd, 1e-9)
        o.update(medD=float(np.nanmedian(D)), p_wil=float(wilcoxon(D)),
                 n_2x=int(np.nansum(r > 2)), frac_2x=float(np.nanmean(r > 2)),
                 med_logratio=float(np.nanmedian(np.log(np.maximum(r, 1e-9)))))
    if traj is not None and traj.shape[1] > 1:
        prog = traj[:, -1] / np.maximum(traj[:, 0], 1e-9)
        o.update(prog_med=float(np.nanmedian(prog)), prog_frac_lt1=float(np.nanmean(prog < 1)))
    return o

out = {}
for name, cfg in SETTINGS.items():
    sh = cfg['shapes']
    fr = load_run(cfg['frozen'], sh)
    if fr is None: print(f'[skip] {name}: frozen missing'); continue
    fs, fd, ftr = fr
    ent = {'frozen': summarize(fs, fd, traj=ftr), 'cells': {}}
    if cfg['hyper']:
        hr = load_run(cfg['hyper'], sh)
        if hr is not None:
            ent['hyper'] = summarize(hr[0], hr[1], fd, hr[2])
    for fam in cfg['fams']:
        for s in STEPS:
            for lr in LRS:
                r = load_run(cfg['cell'](fam, s, lr), sh)
                if r is None: continue
                ent['cells'][f'{fam}|{s}|{lr}'] = summarize(r[0], r[1], fd, r[2])
    out[name] = ent
    nf = len(ent['cells'])
    print(f"{name:14s} frozen succ={ent['frozen']['success']:.3f} med={ent['frozen']['med']:.1f}"
          f" | {nf} cells | hyper={'yes' if 'hyper' in ent else 'n/a'}")
json.dump(out, open(str(STATS), 'w'), indent=1)
print('\nwrote stats.json')


# ============ LaTeX floats ============
S = out


NAMES={'pushobj':r'\pushobj','pushobj_ood':r'\pushobj{} held-out shapes','pusht':r'\pusht','pointmaze':r'\maze'}
ORDER=['pushobj','pusht','pointmaze','pushobj_ood']
def lab(c):
    fam,s,lr=c.split('|')
    ex={'5e-4':r'5{\times}10^{-4}','2e-3':r'2{\times}10^{-3}','1e-2':r'10^{-2}','5e-2':r'5{\times}10^{-2}'}[lr]
    return rf'$K{{=}}{s}$, $\eta{{=}}{ex}$'
def pfmt(p):
    if not np.isfinite(p): return '---'
    if p<1e-4: return r'$<10^{-4}$'
    return f'${p:.2f}$' if p>=0.01 else f'${p:.4f}$'
def dfmt(v,dec):  return f'${v:.{dec}f}$'

# ---------- Table 1: the range within each sweep, in trajectory terms ----------
L=[r'\begin{tabular}{llcccc}', r'\hline',
   r'Setting & Arm & success & median $d$ & median $d-d_{\rm frozen}$ ($p$) & \#\,$>2\times$ frozen \\', r'\hline']
for k in ORDER:
    e=S[k]; f=e['frozen']; n=f['n']; dec=1 if f['med']>20 else 2
    L.append(rf'\multicolumn{{6}}{{l}}{{\emph{{{NAMES[k]}}} --- $n{{=}}{n}$ per cell}}\\')
    L.append(rf'\quad & frozen & ${f["success"]:.3f}$ & {dfmt(f["med"],dec)} & --- & --- \\')
    if 'hyper' in e:
        h=e['hyper']
        L.append(rf'\quad & \textbf{{\method}} & $\mathbf{{{h["success"]:.3f}}}$ & $\mathbf{{{h["med"]:.{dec}f}}}$ & '
                 rf'${h["medD"]:+.{dec}f}$ ({pfmt(h["p_wil"])}) & $\mathbf{{{h["n_2x"]}}}$ \\')
    for fam,flab in [('dense','full predictor-last'),('lora','rank-2')]:
        cs={c:v for c,v in e['cells'].items() if c.startswith(fam+'|')}
        if not cs: continue
        bs=max(cs,key=lambda c:cs[c]['success']); ws=max(cs,key=lambda c:cs[c]['med'])
        for tag,c in [('best',bs),('worst',ws)]:
            v=cs[c]
            L.append(rf'\quad & online {flab}, {tag} & ${v["success"]:.3f}$ & {dfmt(v["med"],dec)} & '
                     rf'${v["medD"]:+.{dec}f}$ ({pfmt(v["p_wil"])}) & ${v["n_2x"]}$ \\')
        med=[x['med'] for x in cs.values()]; n2=[x['n_2x'] for x in cs.values()]
        L.append(rf'\quad & \emph{{span over 16 cells}} & '
                 rf'\emph{{{min(x["success"] for x in cs.values()):.3f}--{max(x["success"] for x in cs.values()):.3f}}} & '
                 rf'\emph{{{min(med):.{dec}f}--{max(med):.{dec}f}}} & '
                 rf'\emph{{a factor of {max(med)/max(min(med),1e-9):.1f}}} & \emph{{{min(n2)}--{max(n2)}}} \\')
    L.append(r'\hline')
L.append(r'\end{tabular}')
open(f'{OUT}/grid_distance.tex','w').write('\n'.join(L)+'\n')
print('wrote grid_distance.tex')

# ---------- Table 2: compact span summary (the headline) ----------
L=[r'\begin{tabular}{lcccccc}', r'\hline',
   r' & \multicolumn{3}{c}{median distance to goal} & \multicolumn{3}{c}{episodes ending $>2\times$ frozen} \\',
   r'\cline{2-4}\cline{5-7}',
   r'Setting & frozen & \method & span over the sweep & \method & \multicolumn{2}{c}{span over the sweep} \\',
   r' & & & full \;/\; rank 2 & & full & rank 2 \\', r'\hline']
for k in ORDER:
    e=S[k]; f=e['frozen']; dec=1 if f['med']>20 else 2
    hm=f'${e["hyper"]["med"]:.{dec}f}$' if 'hyper' in e else '---'
    hx=f'${e["hyper"]["n_2x"]}$' if 'hyper' in e else '---'
    sp={}; n2={}
    for fam in ('dense','lora'):
        cs={c:v for c,v in e['cells'].items() if c.startswith(fam+'|')}
        if not cs: sp[fam]='---'; n2[fam]='---'; continue
        med=[x['med'] for x in cs.values()]; nn=[x['n_2x'] for x in cs.values()]
        sp[fam]=rf'${max(med)/max(min(med),1e-9):.1f}\times$'
        n2[fam]=rf'${min(nn)}$--${max(nn)}$'
    L.append(rf'{NAMES[k]} & ${f["med"]:.{dec}f}$ & {hm} & {sp["dense"]} \;/\; {sp["lora"]} & '
             rf'{hx} & {n2["dense"]} & {n2["lora"]} \\')
L+=[r'\hline', rf'\multicolumn{{7}}{{l}}{{\emph{{Counts are out of $n$ per cell: '
   + ', '.join(f'{NAMES[k]} ${S[k]["frozen"]["n"]}$' for k in ORDER) + r'.}}\\', r'\end{tabular}']
open(f'{OUT}/safety_span.tex','w').write('\n'.join(L)+'\n')
print('wrote safety_span.tex')


# ============ test-seed failure/progress profiles + wrong-guess panel ============
PUSH4=['L','T','Z','plus']; SEEDS=(100,200,400)
def cat(pattern, shapes=PUSH4):
    P=[]
    for sd in SEEDS:
        r=load_run(pattern.format(seed=sd), shapes)
        if r is None: return None
        P.append(r)
    L=min(p[2].shape[1] for p in P)
    return (np.concatenate([p[0] for p in P]), np.concatenate([p[1] for p in P]),
            np.concatenate([p[2][:,:L] for p in P]))
A={'frozen':cat('eval_outputs/indist_matrix/frozen/seed{seed}'),
   'hyper':cat('eval_outputs/indist_matrix/hyper_d0/seed{seed}'),
   'online_lora':cat('eval_outputs/e1_grid/e1_lora_s10_lr5e-4/seed{seed}'),
   'online_dense':cat('eval_outputs/indist_matrix/ada_online/seed{seed}')}
for k,v in A.items(): print(k, v[0].shape, round(v[0].mean(),3))
fs,fd,ftr=A['frozen']
def _wil(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; x=x[x!=0]; n=len(x)
    r=np.argsort(np.argsort(np.abs(x)))+1.0
    W=min(r[x>0].sum(),r[x<0].sum()); mu=n*(n+1)/4; sd=np.sqrt(n*(n+1)*(2*n+1)/24)
    return 2*0.5*(1+erf(((W-mu)/sd)/sqrt(2)))
def pf(p): return r'$<10^{-4}$' if p<1e-4 else (f'${p:.3f}$' if p<0.01 else f'${p:.2f}$')
ROWS=[(r'\method',      'hyper'),
      (r'online rank-2, tuned','online_lora'),
      (r'online full-weight, optimum','online_dense')]
# ---- Table: partition on frozen's own outcome ----
L=[r'\begin{tabular}{lccccc}',r'\hline',
   r'Method & success & median $d$ & median $d-d_{\rm frozen}$ ($p$) & \#\,$>2\times$ frozen \\',r'\hline']
for cname,cond,ctext in [('succ',fs,r'\emph{Episodes the frozen model solves} ($n=%d$) --- can adaptation break them?'),
                         ('fail',~fs,r'\emph{Episodes the frozen model fails} ($n=%d$) --- can adaptation fix them?')]:
    n=int(cond.sum())
    L.append(rf'\multicolumn{{5}}{{l}}{{{ctext % n}}}\\')
    L.append(rf'frozen & {"$1.000$" if cname=="succ" else "$0.000$"} & ${np.nanmedian(fd[cond]):.0f}$ & --- & --- \\')
    for lbl,k in ROWS:
        s,d,_=A[k]; D=(d-fd)[cond]; r=(d/np.maximum(fd,1e-9))[cond]
        L.append(rf'{lbl} & ${s[cond].mean():.3f}$ & ${np.nanmedian(d[cond]):.0f}$ & '
                 rf'${np.nanmedian(D):+.1f}$ ({pf(_wil(D))}) & ${int(np.nansum(r>2))}$ \\')
    L.append(r'\hline')
L.append(r'\end{tabular}')
open(f'{OUT}/failure_profile.tex','w').write('\n'.join(L)+'\n'); print('wrote failure_profile.tex')

# ---- Table: progress on the set where every arm fails ----
triple=(~fs)&(~A['hyper'][0])&(~A['online_lora'][0])&(~A['online_dense'][0])
n=int(triple.sum()); T=min(v[2].shape[1] for v in A.values())
L=[r'\begin{tabular}{lccccc}',r'\hline',
   r'Method & \multicolumn{3}{c}{effect on final distance vs frozen} & \multicolumn{2}{c}{within-episode progress} \\',
   r' & closer $>$10\% & flat & further $>$10\% & median $d_{\rm end}/d_{\rm start}$ & made progress \\',r'\hline']
t=np.log(1.1)
pr=ftr[triple][:,T-1]/np.maximum(ftr[triple][:,0],1e-9)
L.append(rf'frozen & --- & --- & --- & ${np.nanmedian(pr):.2f}$ & ${np.nanmean(pr<1)*100:.1f}\%$ \\')
for lbl,k in ROWS:
    s,d,tr=A[k]; lr=np.log(np.maximum(d[triple]/np.maximum(fd[triple],1e-9),1e-9))
    pr=tr[triple][:,T-1]/np.maximum(tr[triple][:,0],1e-9)
    L.append(rf'{lbl} & ${(lr<-t).mean()*100:.1f}\%$ & ${(np.abs(lr)<=t).mean()*100:.1f}\%$ & '
             rf'${(lr>t).mean()*100:.1f}\%$ & ${np.nanmedian(pr):.2f}$ & ${np.nanmean(pr<1)*100:.1f}\%$ \\')
L+=[r'\hline',r'\end{tabular}']
open(f'{OUT}/progress_profile.tex','w').write('\n'.join(L)+'\n')
print(f'wrote progress_profile.tex (n={n})')

# ---- Table: the wrong-guess panel (cross-parameterization inheritance) ----

NAMES={'pushobj':r'\pushobj','pusht':r'\pusht','pointmaze':r'\maze'}
L=[r'\begin{tabular}{llccccc}',r'\hline',
   r' & & \multicolumn{2}{c}{success} & \multicolumn{2}{c}{median $d$} & \#\,$>2\times$ frozen \\',
   r'\cline{3-4}\cline{5-6}',
   r'Setting & class deployed in & own opt. & inherited & own opt. & inherited & own / inherited \\',r'\hline']
for k in ['pushobj','pusht','pointmaze']:
    e=S[k]; f=e['frozen']; dec=1 if f['med']>20 else 2
    D={fam:{c.split('|',1)[1]:v for c,v in e['cells'].items() if c.startswith(fam+'|')} for fam in ('dense','lora')}
    if not D['dense'] or not D['lora']: continue
    bd=max(D['dense'],key=lambda c:D['dense'][c]['success'])
    bl=max(D['lora'],key=lambda c:D['lora'][c]['success'])
    for src,tgt,slab,tlab in [(bd,'lora',bd,'rank 2'),(bl,'dense',bl,'full weights')]:
        own=max(D[tgt],key=lambda c:D[tgt][c]['success'])
        inh=D[tgt].get(src)
        if inh is None: continue
        o=D[tgt][own]
        L.append(rf'{NAMES[k]} & {tlab} & ${o["success"]:.3f}$ & ${inh["success"]:.3f}$ '
                 rf'($\mathit{{{inh["success"]-o["success"]:+.3f}}}$) & ${o["med"]:.{dec}f}$ & '
                 rf'${inh["med"]:.{dec}f}$ & ${o["n_2x"]}$ / ${inh["n_2x"]}$ \\')
    L.append(r'\hline')
L+=[r'\end{tabular}']
open(f'{OUT}/wrongguess.tex','w').write('\n'.join(L)+'\n'); print('wrote wrongguess.tex')
