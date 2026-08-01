class ToyAdapter:
    def __init__(self, wm, preprocessor, scale=1.0):
        self.wm, self.preprocessor, self.scale = wm, preprocessor, float(scale)
        self.calls = []

    def on_episode_start(self, obs_0, goal):
        self.calls.append("start"); return {}

    def on_transition(self, obs_0, actions, rollout_obs, frameskip):
        self.calls.append("transition"); return {}

    def before_plan(self, obs):
        self.calls.append("plan"); return {}

    def metrics(self):
        return {"toy/scale": self.scale}
