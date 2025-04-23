# Next:
- Rewrite forward method
- Rewrite model notebook/script to run network


# Generally:
## Implementation
- Finish model script, create model notebook
- Implement training and inference over many t_n
- Loss:
    - Add loss for kappa >= 0
- Figure out sqrt(...) and pow(..., 2) bug in GaussianDistanceWeight
- Fix device handling in Network
    - In some places, tensors are created on the GPU
    - In some places, tensors are created on the CPU and the user is expected to move them to the GPU

## Writing
- Check sign (+/-) fix on diffusivity, velocity in README and implementation
- Remake figures in README.md
- Rewrite in README how module works
    - "PINN for inverse vector-field parameterized PDE conditioned on solution"
