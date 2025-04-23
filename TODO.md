# Next:
- Finish model script, create model notebook
- Implement training and inference over many t_n


# Generally:
## Implementation
- Finish model script, create model notebook
- Implement training and inference over many t_n
- Loss:
    - Add loss for kappa >= 0
- Fix device handling in Network
    - In some places, tensors are created on the GPU
    - In some places, tensors are created on the CPU and the user is expected to move them to the GPU
- Figure out sqrt(...) and pow(..., 2) bug in GaussianDistanceWeight
- Investigate OMP error when attempting to save a model from xpu device:
    ```
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
    ```
    - Workaround: move model back to CPU before saving.

## Writing
- Check sign (+/-) fix on diffusivity, velocity in README and implementation
- Remake figures in README.md
- Rewrite in README how module works
    - "PINN for inverse vector-field parameterized PDE conditioned on solution"
