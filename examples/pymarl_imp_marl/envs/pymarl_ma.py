""" Wrapper for road_env respecting the interface of PyMARL. """
import numpy as np

from imp_act import make
from .MultiAgentEnv import MultiAgentEnv


class PymarlMARoadEnv(MultiAgentEnv):
    """
    Wrapper for RoadEnv respecting the interface of PyMARL.
    It manipulates an imp_env to create all inputs for PyMARL agents.
    """

    def __init__(
        self,
        env_type: str = "ToyExample-v1",
        discount_reward: float = 1.0,
        state_obs: bool = True,
        obs_multiple: bool = False,
        reward_normalization: float = 10000.0,
        tuning_factor: float = -500.0,
        seed=None,
    ):
        """
        Initialise based on the full configuration.

        Args:
            env_type: (str) Type of the struct env, either "struct" or "owf".
            discount_reward: (float) Discount factor [0,1[
            state_obs: (bool) State contains the concatenation of obs
            obs_multiple: (bool) Obs contains the concatenation of all obs
            seed: (int) seed for the random number generator
        """

        assert isinstance(state_obs, bool) and isinstance(
            obs_multiple, bool
        ), "Error in env parameters"
        assert 0 <= discount_reward <= 1, "Error in discount_reward"

        self.env_type = env_type
        self.discount_reward = discount_reward
        self.state_obs = state_obs
        self.obs_multiple = obs_multiple
        self.reward_normalization = reward_normalization
        self.tuning_factor = tuning_factor
        self._seed = seed

        self.road_env = make(env_type)
        self.road_env.travel_time_reward_factor = tuning_factor

        self.number_edges = []
        for edge in self.road_env.graph.es:
            self.number_edges.append(edge["road_segments"].number_of_segments)

        self.n_agents = sum(self.number_edges)

        self.episode_limit = self.road_env.max_timesteps
        # self.agent_list = self.struct_env.agent_list
        self.n_actions = 5
        self.road_env.reset()

        # obs_list = self.get_list_obs(self.road_env._get_observation()['edge_beliefs'])
        # observations = []
        # for item in obs_list:
        #     observations.append( np.append(item, self.road_env._get_observation()['time_step'] / self.road_env.max_timesteps ))
        # self.observations = observations

        self.observations = self.get_obs_and_time()

        # self.observations = self.get_list_obs(self.road_env._get_observation()['edge_beliefs'])

        # self.action_histogram = {"action_" + str(k): 0 for k in range(self.n_actions)}

        self.unit_dim = self.get_unit_dim()  # Qplex requirement

    # def update_action_histogram(self, actions):
    #     """
    #     Update the action histogram for logging.

    #     Args:
    #         actions: list of actions
    #     """
    #     for k, action in zip(self.struct_env.agent_list, actions):
    #         if type(action) is torch.Tensor:
    #             action_str = str(action.cpu().numpy())
    #         else:
    #             action_str = str(action)
    #         self.action_histogram["action_" + action_str] += 1

    def step(self, actions):  # Done
        """
        Ask to run a step in the environment.

        Args:
            actions: list of actions

        Returns:
            rewards: list of rewards
            done: True if the episode is finished
            info: dict of info for logging
        """

        # self.update_action_histogram(actions)
        # action_dict = {
        #     k: action for k, action in zip(self.struct_env.agent_list, actions)
        # }
        actions_nested = self.get_nested_list(actions, self.number_edges)
        _, reward, done, _ = self.road_env.step(actions_nested)
        reward /= self.reward_normalization
        self.observations = self.get_obs_and_time()
        # obs_list = self.get_list_obs(self.road_env._get_observation()['edge_beliefs'])
        # observations = []
        # for item in obs_list:
        #     observations.append( np.append(item, self.road_env._get_observation()['time_step'] / self.road_env.max_timesteps ))
        # self.observations = observations
        # 'time_step'
        info = {}
        # if done:
        #     for k in self.action_histogram:
        #         self.action_histogram[k] /= self.episode_limit * self.n_agents
        #     info = self.action_histogram
        return reward, done, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_unit_dim(self):
        """Returns the dimension of the unit observation used by QPLEX."""
        return len(self.all_obs_from_struct_env()) // self.n_agents

    def get_obs_agent(self, agent_id: int):
        """
        Returns observation for agent_id

        Args:
            agent_id: id of the agent (int in range(self.n_agents)).
        """

        if self.obs_multiple:
            obs = self.all_obs_from_struct_env()
        else:
            obs = self.observations[agent_id]

        return obs

    def get_obs_size(self):
        """Returns the size of the observation."""
        return len(self.get_obs_agent(0))

    def all_obs_from_struct_env(self):
        """Returns all observations concatenated in a single vector."""
        # Concatenate all obs with a single time.
        idx = 0
        obs = None
        for k in self.observations:
            if idx == 0:
                obs = k
                idx = 1
            else:
                obs = np.append(obs, k)
        return obs

    def get_nested_list(self, flat_list, pattern):
        """Structures a flat list into a nested list following a specific size pattern."""
        nested_list = []
        start_index = 0
        for size in pattern:
            end_index = start_index + size
            nested_list.append(flat_list[start_index:end_index])
            start_index = end_index
        return nested_list

    def get_list_obs(self, nested_list):
        """Structures a nested list into a flat list."""
        flat_list = []
        for sublist in nested_list:
            for item in sublist:
                flat_list.append(item)
        return flat_list

    def get_obs_and_time(self):
        """Returns observations and normalized time step."""
        obs_list = self.get_list_obs(self.road_env._get_observation()["edge_beliefs"])
        observations = []
        for item in obs_list:
            observations.append(
                np.append(
                    item,
                    self.road_env._get_observation()["time_step"]
                    / self.road_env.max_timesteps,
                )
            )
        return observations

    def get_state(self):
        """Returns the state of the environment."""
        state = []
        if self.state_obs:
            state = np.append(state, self.all_obs_from_struct_env())
        return state

    def get_state_size(self):
        """Returns the shape of the state"""
        return len(self.get_state())

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """
        Returns the available actions for agent_id.

        Args:
            agent_id: id of the agent (int in range(self.n_agents)).
        """
        return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        # self.action_histogram = {"action_" + str(k): 0 for k in range(self.n_actions)}
        self.road_env.reset()

        self.observations = self.get_obs_and_time()
        # obs_list = self.get_list_obs(self.road_env._get_observation()['edge_beliefs'])
        # observations = []
        # for item in obs_list:
        #     observations.append( np.append(item, self.road_env._get_observation()['time_step'] / self.road_env.max_timesteps ))
        # self.observations = observations
        # self.observations = self.get_list_obs(self.road_env._get_observation()['edge_beliefs'])
        return self.get_obs(), self.get_state()

    def render(self):
        """See base class."""
        pass

    def close(self):
        """See base class."""
        pass

    def seed(self):
        """Returns the random seed"""
        return self._seed

    def save_replay(self):
        """See base class."""
        pass

    def get_stats(self):
        """See base class."""
        return {}