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
The IMP-ACT environment can be created using the IMP-ACT module as:
```python
from imp_act import make
# initialize the environment
env = make("ToyExample-v2")
```
## Road network(Graph) parameters
We use the igraph library to represent the road network graph, which is stored in the env object. For more information on igraph, see: https://python.igraph.org/en/stable/tutorial.html
```python
g = env.graph
```
We demonstrate how to access some basic properties of the graph
```python
print(f"Number of nodes: {env.graph.vcount()}")
print(f"Number of edges: {env.graph.ecount()}")
print(f"Adjacency matrix: {g.get_adjacency().data}")
print()
```
Each graph edge represents a unique road edge in the environment, and each road edge has multiple road segments.
```python
for edge in env.graph.es:
    edge_index = edge.index

    # Each graph edge has a unique RoadEdge object associated with it
    road_edge = edge["road_edge"]

    # Each RoadEdge object has a list of RoadSegment objects
    num_segments = len(road_edge.segments)

    print(f"Edge {edge_index} has {num_segments} segments")

print()
```
## Road segment variables
A road segment is the fundamental deteriorating unit of the road network. It has a damage state, observation, belief, capacity, and base travel time. Let us access the first road segment of the second road edge
```python
road_edge = env.graph.es[1]["road_edge"]
road_segment = road_edge.segments[0]

print(f"Road segment 0 of edge 1 has the following properties:")
print(f"Damage state: {road_segment.state}")
print(f"Observation: {road_segment.observation}")
print(f"Belief: {road_segment.belief}")
print(f"Capacity: {road_segment.capacity}")
print(f"Base travel time: {road_segment.base_travel_time}")
```
There are 5 actions for each road segment: 
- 0: do-nothing
- 1: inspect
- 2: minor-repair
- 3: major-repair 
- 4: replace
  
## Traffic summary
The traffic summary for each road edge in the network is shown here.
```python
print(env.get_edge_traffic_summary())
```
## Environment functions
The reset and step functions are identical to Gym. The environment can be reset as:
```python
obs = env.reset()
```
Let us pick the minor-repair (2) action for all road segments
```python
system_actions = []
for edge in env.graph.es:
    road_edge = edge["road_edge"]
    segment_actions = []
    for segment in road_edge.segments:
        segment_actions.append(2)

    system_actions.append(segment_actions)

print(f"System actions: {system_actions}")
```
A step in the environment can be taken as:
```python
observation, reward, done, info = env.step(system_actions)

print("Observation:")
pprint(observation)
pprint(f"Reward: {reward}")
pprint(f"Done: {done}")
print("Info:")
pprint(info)
```
## Environment Rollout
```python
obs = env.reset()
done = False
timestep = 0

while not done:

    # system_actions = policy(observation)

    observation, reward, done, info = env.step(system_actions)

    timestep += 1

print(f"Number of timesteps: {timestep}")
print(f"Done: {done}")
```
