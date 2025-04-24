# Next:
- Create model notebook
- Implement training and inference over many t_n


# Generally:
## Implementation
- Create model notebook
- Implement training and inference over many t_n
- Fix device handling in Network
    - In some places, tensors are created on the GPU
    - In some places, tensors are created on the CPU and the user is expected to move them to the GPU
- Figure out sqrt(...) and pow(..., 2) bug in GaussianDistanceWeight


## Writing
- Check sign (+/-) fix on diffusivity, velocity in README and implementation
- Remake figures in README.md
- Rewrite in README how module works
    - "PINN for inverse vector-field parameterized PDE conditioned on solution"
