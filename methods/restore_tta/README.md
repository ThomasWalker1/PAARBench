# Restore TTA

`restore_tta` is online gradient TTA over the world model's own latent prediction
loss, with stochastic restoration after each optimizer update. For every scalar an
optimizer can change, it independently copies the pretrained snapshot back with
probability `p`. The correction can still accumulate across replans, but restoration
creates a persistent leak against drift.

It is deliberately **not** named `cotta`. Full CoTTA (Wang et al., CVPR 2022)
combines a teacher-student EMA, augmentation-averaged pseudo-labels, and stochastic
restoration. This world-model setting has no prediction target for the first two
mechanisms, so this directory implements and claims only the restoration mechanism.

The question is narrowly useful: AdaJEPA's compounding slope is positive on PushObj
(+0.91 per replan, 95% CI [+0.40, +1.68]), whereas recomputed corrections are flat.
Does restoring random parameter elements flatten that slope, and what does it cost in
success rate? A null result is reported as such; the rule is a four-value selection
grid for `p` (`0.001, 0.01, 0.05, 0.1`), not repeated tuning until a preferred
result appears.

Like AdaJEPA, this method owns shared world-model weights and AdamW state, so it
declares episode isolation. Its base gradient configuration is AdaJEPA's already
declared selected cell; the only additional search cost is the four restoration
probabilities. Each reset also restores the full pretrained snapshot and rebuilds the
optimizer, preventing cross-episode leakage.

The implementation shares AdaJEPA's model-specific loss and parameterization closely
by design. That makes it a focused mechanism comparison, but it is a weaker test of
the adapter interface's generality than a method with a different update structure.

