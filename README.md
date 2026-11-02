### Setup & Running Experiments

The `flow_matching` directory is now configured as a local Python module. After pulling the repository, follow these steps to set up the environment and run the experiments:

1. **Install dependencies and the local module:** Run this command from the root of the repository:
   ```bash
   uv sync

2. **Run an experiment:** Navigate to your experiment folder and execute the scripts using uv run. This automatically manages the environment and paths:
    ```bash
    cd results/experiment_3
    uv run python generating.py


Experiments: 

1. First approach with a 2 hidden-fully-connected layers. 
2. Adding pooling layers to the previous architecture. 
3. First implementation of a 1D convolutional net. 
4. Songs U-Net with 1D convolutions.
5. Songs U-Net with kernel size as a parameter. 
6. Convolutional net with a Fourier Projection time embedding added on each step. 
7. Adding skip connections to the previous Net. 
8. Adding group norm to the previous Net. 
9. Showing how granularity can make models with same kernel sizes stop learning. 
10. Experiment 10 with `flow_matching_fda`, the `flow_matching` module rewritten on scikit-fda: data as `FDataGrid`, ODE integrated with `skfda.datasets.make_sde_trajectories` (zero diffusion) and metrics computed with scikit-fda integrals and L2 norms. Also searching of the optimal k with granularity 256. 
11. Upgrading the kernel size for higher granularities, in order to stablish the optimal relation between them.