from itertools import product

import numpy as np


class Heuristic:
    def __init__(self, env, norm_constant=1e6, rules_range=None):
        self.env = env
        self.norm_constant = norm_constant
        self.rules_range = rules_range

        if rules_range is not None:
            # Initialize the rules values with the first set of rules
            self.rules_values = {
                key: list(rules)[0] for key, rules in rules_range.items()
            }
            self.best_rules = self.rules_values

    def policy(self, obs):
        """Returns actions"""
        raise NotImplementedError

    def get_rollout(self, env, policy, verbose=False):
        obs = env.reset()
        done = False
        total_reward = 0
        store_rewards = {
            "reward": [],
            "travel_time_reward": [],
            "maintenance_reward": [],
            "total_travel_time": [],
        }

        while not done:
            # To be modified according to the policy
            actions = policy(obs)

            next_obs, reward, done, info = env.step(actions)

            obs = next_obs

            if verbose:
                print(f"timestep: {obs['time_step']}")
                print(f"actions: {actions}")
                print(
                    f"reward: {reward/1e6:.3f}, total_travel_time: {info['total_travel_time']:.3f}"
                )
                print(
                    f"travel time reward: {info['reward_elements']['travel_time_reward']/1e6:.3f}, maintenance reward: {info['reward_elements']['maintenance_reward']/1e6:.3f}"
                )
                print(
                    f"Remaining maintenance budget: {obs['budget_remaining']/1e6:.3f}"
                )
                print(f"Budget until renewal: {obs['budget_time_until_renewal']}")
                print("=" * 50)
                print(f"observations: {obs['edge_observations']}")

            total_reward += reward
            store_rewards["reward"].append(reward)
            store_rewards["total_travel_time"].append(info["total_travel_time"])
            store_rewards["travel_time_reward"].append(
                info["reward_elements"]["travel_time_reward"]
            )
            store_rewards["maintenance_reward"].append(
                info["reward_elements"]["maintenance_reward"]
            )

        return total_reward, store_rewards

    def optimize_heuristics(self, num_episodes):
        # Determine the dimensions for each rule range
        rules_range_dimensions = [len(rule) for rule in self.rules_range.values()]
        store_policy_rewards = np.zeros((num_episodes, *rules_range_dimensions))

        # Generate all possible combinations of rules
        combinations = list(product(*self.rules_range.values()))

        for episode in range(num_episodes):
            for idx, rules in enumerate(combinations):
                # Dynamically find indices for each threshold based on `self.rules_range` keys
                indices = [
                    list(self.rules_range[key]).index(threshold)
                    for key, threshold in zip(self.rules_range.keys(), rules)
                ]

                # Store the rules values in a dictionary
                self.rules_values = {
                    key: threshold
                    for key, threshold in zip(self.rules_range.keys(), rules)
                }

                # Use unpacking to store rewards in the correct array dimension
                store_policy_rewards[(episode, *indices)], _ = self.get_rollout(
                    self.env, lambda obs: self.policy(obs)
                )

        # Compute heuristic policies by averaging across episodes and normalizing
        policies_heur = (
            np.sum(store_policy_rewards, axis=0) / num_episodes / self.norm_constant
        )

        # Find the indices of the maximum value in the policies_heur array
        best_policy_idx = np.unravel_index(
            np.argmax(policies_heur), policies_heur.shape
        )

        # Retrieve the best rules corresponding to the best policy
        self.best_rules = {
            key: list(self.rules_range[key])[index]
            for key, index in zip(self.rules_range.keys(), best_policy_idx)
        }

        # Print the best rules and the associated policy value
        for rule, value in self.best_rules.items():
            print(f"Best {rule}: {value}")
        print(f"Best policy value: {policies_heur[best_policy_idx]:.4f}")

        return store_policy_rewards

    def evaluate_heuristics(self, num_episodes):
        if self.rules_range is not None:
            self.rules_values = self.best_rules
        # Re-evaluate the best policy
        best_policy_rewards = np.zeros(num_episodes)
        for episode in range(num_episodes):
            best_policy_rewards[episode], _ = self.get_rollout(
                self.env, lambda obs: self.policy(obs)
            )

        best_policy_mean = np.mean(best_policy_rewards) / self.norm_constant
        best_policy_std = np.std(best_policy_rewards) / self.norm_constant
        confidence_interval = 1.96 * best_policy_std / np.sqrt(num_episodes)
        print(f"Best policy with evaluated reward: {best_policy_mean:.3f}")
        print(f"Standard deviation of the best policy: {best_policy_std:.3f}")
        print(f"95% Confidence interval: {confidence_interval:.3f}")
        reward_stats = [best_policy_mean, best_policy_std, confidence_interval]
        return best_policy_rewards, reward_stats

    def print_policy(self, num_episodes):
        if self.rules_range is not None:
            self.rules_values = self.best_rules
        for _ in range(num_episodes):
            print_policy_reward, _ = self.get_rollout(
                self.env, lambda obs: self.policy(obs), verbose=True
            )

            print(print_policy_reward / self.norm_constant)
