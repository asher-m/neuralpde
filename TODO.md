# For Delaney:
1. For Monday: 
    - Basic diffusivity, velocity, and forcing maps
2. For Wednesday:
    - Expand paper:
        - Inverse PINN conditioned on input maps

# Generally:
## Implementation
- Implement additional loss terms to handle regularity of kappa, v
- Refigure how to handle land boundary
    - Thinking: mask out land with zeros (i.e., multiply loss terms by 0) so it does not contribute to the loss
    - Can/should do the same for regularity conditions on kappa, v
- Refigure how to handle ocean boundary
    - Does anything need to be done?
    - How to address edge-of-domain artifacting?
- Implement Xavier initialization
- Profile network and try to speed up
- Batch processing

## Writing
- Remake figures in README.md and for Delaney's paper
- Rewrite in README how module works
    - "PINN for inverse vector-field parameterized PDE conditioned on solution"
