from heuristics.heuristic import Heuristic
import numpy as np


class HumbleHeuristic(Heuristic):
    def __init__(self, env, config, norm_constant):
        super().__init__(env, config, norm_constant)

    def policy(self, obs):

        edge_obs = obs["edge_observations"]
        current_time = obs["time_step"]

        replacement_threshold = self.rules_values["replacement_threshold"]
        major_repair_threshold = self.rules_values["major_repair_threshold"]
        minor_repair_threshold = self.rules_values["minor_repair_threshold"]
        inspection_interval = self.rules_values["inspection_interval"]

        actions = []
        for e in edge_obs:
            edge_actions = []
            for obs in e:
                if obs >= replacement_threshold:
                    edge_actions.append(4)  # Reconstruction
                elif obs >= major_repair_threshold:
                    edge_actions.append(3)  # Major repair
                elif obs >= minor_repair_threshold:
                    edge_actions.append(2)  # Minor repair
                elif current_time % inspection_interval == 0:
                    edge_actions.append(1)  # Inspection
                else:
                    edge_actions.append(0)  # Do nothing
            actions.append(edge_actions)
        return actions


class DoNothing(Heuristic):
    def __init__(self, env, config, norm_constant):
        super().__init__(env, config, norm_constant)

    def optimize_heuristics(self, num_episodes):
        pass

    def policy(self, obs):

        edge_obs = obs["edge_observations"]

        actions = []
        # Always do nothing
        for e in edge_obs:
            actions_edge = [0] * len(e)
            actions.append(actions_edge)

        return actions


class RandomPolicy(Heuristic):
    def __init__(self, env, config, norm_constant):
        super().__init__(env, config, norm_constant)

    def optimize_heuristics(self, num_episodes):
        pass

    def policy(self, obs):

        edge_obs = obs["edge_observations"]

        actions = []
        # Always do nothing
        for e in edge_obs:
            # Generate a random action for each segment in the edge observation (values 0 to 4)
            actions_edge = np.random.randint(0, 5, size=len(e)).tolist()
            actions.append(actions_edge)

        return actions
