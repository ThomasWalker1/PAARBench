# PAD

PAD (Hansen et al.) adapts the world-model encoder during deployment using an
inverse-dynamics objective: from the latent before and after an executed action chunk,
an auxiliary head predicts that chunk. This differs materially from the existing arms:
the head is a module owned by the adapter, pretrained offline and then optimized with
the encoder online.

The head is pretrained on 2,048 transition pairs from the existing PushObj training
set, using frozen base latents. `train_inverse_head.py` records the checkpoint and is
the source for `checkpoints/pad/pushobj_inverse_dynamics.pth`. At deployment PAD
updates both the encoder and head on the most recently executed model transition; it
declares episode isolation because the optimizer owns shared weights.

The original PAD head is jointly pretrained with the policy. This benchmark has a
world model and MPC planner rather than that policy, so the shipped head is the closest
honest substitute: it is pretrained on the available offline transitions but never on
selection or test episodes. Results should be read as a test of whether this auxiliary
signal transfers to latent world-model control, not a bit-identical reproduction of
the original policy architecture.
