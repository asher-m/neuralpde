# Next:
- Rewrite forward method
- Rewrite model notebook/script to run network


# Generally:
## Implementation
- Reimplement PINN as function over scalar (x, y)
- Loss:
    - Add bc loss for kappa = 0
    - Add loss for kappa >= 0
- Initialization:
    - Implement Xavier initialization (i.e. for layers)
- Profile network and try to speed up
- Figure out sqrt(.) and (.)**2 bug in GaussianDistanceWeight

## Writing
- Finish fixing sign errors on diffusivity, velocity in README and implementation
    - Check
- Remake figures in README.md and for Delaney's paper
- Rewrite in README how module works
    - "PINN for inverse vector-field parameterized PDE conditioned on solution"
