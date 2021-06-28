# FAQ

**Please read this doc before creating any issues.**

**Please don't spam the GitHub issue section.**

## CARLA
Please **DO NOT** create any issues if you encounter any problems or bugs running CARLA, with **one exception** listed below. 
For everything else, please consider posting to the great [CARLA community](https://github.com/carla-simulator/carla/)!

**Q: I cannot run CARLA 0.9.10.1 with vulkan headlessly**

**A**: This is a known issue of Unreal 4.24. The solution is to build CARKA 
from source with Unreal 4.25, with this [patch](https://github.com/carla-simulator/carla/issues/3214#issuecomment-769140155).

## Install

**Q: I cannot install package XX in the dependencies**

**A**: We do not have time to individually answer these questions, due to different setups and systems.

The best chance of quicky fixing these problems is usually through [Google](https://www.google.com/) or [Stack Overflow](https://stackoverflow.com/).

## Training

**Q: Why is phase2 loss different from what is described in the paper?**

**A**: They are mathematically equivalent. See [here](https://github.com/dotchen/WorldOnRails/issues/17#issuecomment-868044575).

## Dataset

**Q: I see error: `TypeError: must be real number, not NoneType` while collecting dataset.**

**A**: reinstall `moviepy`, `imageio`, `wandb`

**Q: Why the data loader cannot read your public dataset?**

**A**: We reformatted our dataset for release. If you would like to train a leaderboard model using our dataste, you need to write a custom data loader.
Or you can choose to collect your own data, starting from phase0/1. Refer to [RAILS.md](RAILS.md) for more details. 

## ProcGen

**Q: When will the ProcGen code be released?**

**A**: We are working on the code clean-up at this point, please stay tuned for the release!

## Evaluation

**Q: Why the released model fails/has collision on Route XX?**

**A**: CARLA leaderboard is hard task! As of now (Apr 2021), the top solution at this point only gets ~30 driving score. 
Depending on the route, the driving score of our model could vary from single digits to 100.

**Q: Why does the model always run through stop signs?**

**A**: Currently, our reward function does not consider stopping at stop/yield signs. 



