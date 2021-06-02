# Installation

This doc provides instructions to get started.

## Install CARLA
* Download this repo `git clone --recurse-submodules -j8 git@github.com:dotchen/WorldOnRails.git`
* Download and unzip [CARLA 0.9.10.1](https://github.com/carla-simulator/carla/releases/tag/0.9.10.1)

## Install dependencies
* First, inside the repo, create a dedicated conda environment. Refer [here](https://www.anaconda.com/products/individual#Downloads) if you do not have conda.
```
conda env create -f environment.yaml
```
* Inside the conda environment, install the CARLA PythonAPI `easy_install [PATH TO CARLA EGG]`. Refer to [this link](https://leaderboard.carla.org/get_started/) if you are confused at this step.
* Setup [wandb](https://docs.wandb.ai/quickstart)

## Configure environment variables

**Note**: the following instructions mostly only apply to Linux.

Set the following environmental variables to your conda environment. 
Refer [here](https://docs.conda.io/projects/conda/en/4.6.0/user-guide/tasks/manage-environments.html#saving-environment-variables) for instructions to do so.

```bash
export CARLA_ROOT=[LINK TO YOUR CARLA FOLDER]
export LEADERBOARD_ROOT=[LINK TO WORLD ON RAILS REPO]/leaderboard
export SCENARIO_RUNNER_ROOT=[LINK TO WORLD ON RAILS REPO]/scenario_runner
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}"
```

Now, you can treat this conda environment the dedicated one (by defauld named `world_on_rails`).
