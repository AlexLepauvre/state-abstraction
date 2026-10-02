#  Task-structured preferences guide forward planning in sequential decision-making paper repository

[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-blue)](https://doi.org/10.64898/2026.09.25.754065)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit)


**Authors:** Alex Lepauvre¹

¹ Department of Psychology, Technische Universität Dresden, Dresden, Germany

---

## About

This repository contains a toolbox with various function to solve Markov Decsion Problems. It implements
some basic RL functions, such as backward induction, value iteration, policy evaluation... In addition,
the toolbox contains a few options to perform state abstraction: bisimulation distance, Q distance... and generates
reduced MDP based on these techniques. 

This repository was created to be used for the paper:

> **Task-structured preferences guide forward planning in sequential decision-making paper repository**
> Alex Lepauvre, Florian Ott, Stefan Kiebel
> BioRxiv, 2026. DOI: [10.64898/2026.09.25.754065](https://doi.org/10.64898/2026.09.25.754065)


## Links

- **Paper (HTML):** https://alexlepauvre.github.io/state_abstraction_paper/

## Repository Structure

```
.
├── stabst/                               # Package folder
├──     __init__.py                       # 
├──     MarkovDecisionProcess.py          # Main MDP functions
├──     TaskConfig.py                     # Class to create MDP object matching a particular task that fnctions in MarkovDecsionProcess can work with
├──     utils.py                          # Various useful functions
```

### Usage

To install the code: 

```bash
# Set up the environment, e.g.:
pip install virtualenv
virtualenv stabst_env
source stabst_env/bin/activate
pip install stabst @ git+https://github.com/AlexLepauvre/state-abstraction.git@main
```

Once you have installed the package, you should first create the task, and then apply the functions from MDP on it:

```python
# Custom packages:
from stabst.MarkovDecisionProcess import MDP
from stabst.TaskConfig import LimitedEnergyTask

# 1. Create the task:
task = LimitedEnergyTask()
task.build()

# 2. Instantiate MDP
gamma = 1
task_mdp = MDP(task.states, task.tp, task.r, s2i=task.s2i)

# 3. Apply backward induction to solve the MDP
V_full, Q_full = task_mdp.backward_induction()
```

Note that only the `LimitedEnergyTask` is implemented, which corresponds to the task used in the above mentioned [paper](https://doi.org/10.64898/2026.09.25.754065). To use with another task, you can either create another task structure function in `TaskConfig.py`, or directly pass a list of states, transition probability and reward matrices. Note that the scripts have not been tested for other tasks, so make sure to thoroughly test it if you wish to use it for anything else. 
