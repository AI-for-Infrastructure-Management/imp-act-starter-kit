import itertools
import multiprocessing as mp

import numpy as np

import tqdm

def parallel_rollout(env, heuristic, rollout_method, num_episodes):

    # create an iterable for the starmap
    iterable = zip(
        itertools.repeat(env, num_episodes), itertools.repeat(heuristic, num_episodes)
    )

    # multiprocessing using all available cores
    with mp.Pool(mp.cpu_count()) as pool:
        list_func_evaluations = pool.starmap(rollout_method, iterable)

    return np.hstack(list_func_evaluations)


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

    def get_rollout(self, env, policy, seed=None, verbose=False):
        env.seed(seed)
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            actions = policy(obs)

            obs, reward, done, info = env.step(actions)

            if verbose:
                print(f"timestep: {obs['time_step']}")
                print(f"actions: {actions}")
                print(
                    f"reward: {reward/self.norm_constant:.3f}, total_travel_time: {info['total_travel_time']:.3f}"
                )
                print(
                    f"travel time reward: {info['reward_elements']['travel_time_reward']/self.norm_constant:.3f}, maintenance reward: {info['reward_elements']['maintenance_reward']/self.norm_constant:.3f}"
                )
                print(
                    f"Remaining maintenance budget: {obs['budget_remaining']/self.norm_constant:.3f}"
                )
                print(f"Time until budget renewal: {obs['budget_time_until_renewal']}")
                print("=" * 50)
                print(f"observations: {obs['edge_observations']}")

            total_reward += reward

        return total_reward

    def optimize_heuristics(self, num_episodes):
        # Determine the dimensions for each rule range
        rules_range_dimensions = [len(rule) for rule in self.rules_range.values()]
        store_policy_rewards = np.zeros((num_episodes, *rules_range_dimensions))

        # Generate all possible combinations of rules
        combinations = itertools.product(*self.rules_range.values())

        for rules in tqdm.tqdm(combinations, total=np.prod(rules_range_dimensions)):
            # Dynamically find indices for each threshold based on `self.rules_range` keys
            indices = [
                list(self.rules_range[key]).index(threshold)
                for key, threshold in zip(self.rules_range.keys(), rules)
            ]

            # Store the rules values in a dictionary
            self.rules_values = {
                key: threshold for key, threshold in zip(self.rules_range.keys(), rules)
            }

            # Evaluate the rule using environment rollouts

            # sequential evaluation
            # for episode in range(num_episodes):

            #     # Use unpacking to store rewards in the correct array dimension
            #     store_policy_rewards[(episode, *indices)], _ = self.get_rollout(
            #         self.env, self.policy
            #     )

            # parallel evaluation
            results = parallel_rollout(
                self.env, self.policy, self.get_rollout, num_episodes
            )

            # Python 3.9 does not support tuple unpacking in indexing, so we use slice(None) to unpack the tuple
            store_policy_rewards[(slice(None), *indices)] = results

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
        best_policy_rewards = parallel_rollout(
            self.env, self.policy, self.get_rollout, num_episodes
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
            total_reward = self.get_rollout(self.env, self.policy, verbose=True)

            print(f"Episode return: {total_reward/self.norm_constant:.3f}")
