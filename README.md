<h1>Table of Contents</h1>

- [Environment Description](#environment-description)
  - [What does it look like?](#what-does-it-look-like)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Rewards](#rewards)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Please Cite](#please-cite)

# Environment Description

The multi-agent infrastructure management planning (IMP-ACT) environment simulates a real-world road network consisting of multiple road edges decomposed into distinct road segments which need to be adequately maintained over a planning horizon. The environment involves multiple agents, where each agent is responsible for maintaining a road segment in order to minimize certain shared objectives while meeting specific budget constraints. 

## Action Space
In this environment simulation, agents for the road segments have the following discrete action space:

A={ Do Nothing, Inspect, Minor Repair, Major Repair, Replacement }

The inspect action reveals the underlying condition state of the road segment with a high level of accuracy. Every maintenance action is associated with their respective inspection actions. The replacement action is associated with perfect inspection. The other actions are associated with minor inspections which reveal the underlying state with a relatively lower level of accuracy.

## Observation Space
The conditon of each road segment is characterized by 5 discrete states. The observation of an agent is partially observable and constitutes the underlying condition state of the road segment with a degree of inspection accuracy. The observation of an agent can incorporate other deterioration parameters of the road segment and the overall system:
- The deterioration rate of the road segment.
- The current timestep of the decision horizon.
- The remaining budget at the current timestep.

## Rewards
At each timestep, the agents share the costs of maintaining the road segments of the network. The agents in this cooperative environment share the goal of minimizing the overall maintenance costs of the road network, while satisfying certain budget constraints which is renewed at regular intervals. 

# Environment Parameters
The IMP-ACT environment is characterized by a number of parameters:
- The number of agents which represent the road segments (N)
- The budget constraints to be maintained for the entire road network

# Installation
* Python >=3.7,<3.11 (note that you will need Python < 3.10 to run PyMARL or EPyMARL)
* The imp-act package needs to be installed. This package will be released later. Until then, the requirements can be installed from the GitHub repository [IMP-ACT](https://github.com/AI-for-Infrastructure-Management/imp-act) as:
  ```bash
  git clone https://github.com/AI-for-Infrastructure-Management/imp-act.git
```
Installation via *pip requirements*
```bash
pip install -r requirements/requirements.txt
pip install -e .
```
  

# Getting Started
