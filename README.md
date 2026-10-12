### Setup & Running Experiments

The `flow_matching` directory is now configured as a local Python module. After pulling the repository, follow these steps to set up the environment and run the experiments:

1. **Install dependencies and the local module:** Run this command from the root of the repository:
   ```bash
   uv sync

2. **Run an experiment:** Navigate to your experiment folder and execute the scripts using uv run. This automatically manages the environment and paths:
    ```bash
    cd results/experiment_3
    uv run python generating.py

