from heuristics.heuristic import Heuristic

class HumbleHeuristic(Heuristic):
    def __init__(self, env, norm_constant=1e6, rules_range=None):
        super().__init__(env, norm_constant, rules_range)

    def policy(self, obs):

        edge_obs = obs['edge_observations']
        current_time = obs['time_step']
        
        replacement_threshold = self.rules_values['replacement_threshold'] 
        major_repair_threshold = self.rules_values['major_repair_threshold'] 
        minor_repair_threshold = self.rules_values['minor_repair_threshold'] 
        inspection_interval = self.rules_values['inspection_interval'] 

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
    def __init__(self, env, norm_constant=1e6, rules_range=None):
        super().__init__(env, norm_constant, rules_range)

    def policy(self, obs):

        edge_obs = obs['edge_observations']
        
        # Always do nothing
        actions = [[0] * len(e["road_edge"].segments) for e in edge_obs]

        return actions

