import math
import itertools
import multiprocessing as mp
from tqdm import tqdm

from heuristics.heuristic import Heuristic
import numpy as np


def parallel_rollout(env, heuristic, rollout_method, num_episodes, policy_params):

    # create an iterable for the starmap
    iterable = zip(
        itertools.repeat(env, num_episodes),
        itertools.repeat(heuristic, num_episodes),
        itertools.repeat(policy_params, num_episodes),
    )

    # multiprocessing using all available cores
    with mp.Pool(mp.cpu_count()) as pool:
        list_func_evaluations = pool.starmap(rollout_method, iterable)

    return np.hstack(list_func_evaluations)


class ImportanceHeuristic(Heuristic):

    def __init__(self, env, config, norm_constant):
        super().__init__(env, config, norm_constant)

        # expand rules range
        self.prioritized_components = config["prioritized_components"]
        self.num_prioritized = len(self.prioritized_components)
        self.prioritized_rules_range = {
            f"{key}": np.arange(value["min"], value["max"], value["interval"])
            for key, value in config["prioritized_rules_range"].items()
        }
        self.unprioritized_rules_range = {
            key: np.arange(value["min"], value["max"], value["interval"])
            for key, value in config["unprioritized_rules_range"].items()
        }

        print(
            f"Prioritized components ({self.num_prioritized}): {self.prioritized_components}"
        )

    def get_rollout(self, env, policy, policy_params, verbose=False):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            actions = policy(obs, policy_params)

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

    def policy(self, obs, policy_params):

        edge_obs = obs["edge_observations"]
        current_time = obs["time_step"]

        def component_policy(thresholds):

            replacement_threshold = thresholds["replacement_threshold"]
            major_repair_threshold = thresholds["major_repair_threshold"]
            minor_repair_threshold = thresholds["minor_repair_threshold"]
            inspection_interval = thresholds["inspection_interval"]

            if obs >= replacement_threshold:
                return 4  # Reconstruction
            elif obs >= major_repair_threshold:
                return 3  # Major repair
            elif obs >= minor_repair_threshold:
                return 2  # Minor repair
            elif current_time % inspection_interval == 0:
                return 1  # Inspection
            else:
                return 0  # Do nothing

        c = 0
        actions = []
        for e in edge_obs:
            edge_actions = []
            for obs in e:

                # prioritized components
                if c in self.prioritized_components:
                    _dict = policy_params[c]
                    edge_actions.append(component_policy(_dict))

                # unprioritized components
                else:
                    edge_actions.append(component_policy(policy_params))

            c += 1
            actions.append(edge_actions)

        return actions

    def get_policy_params(self, rules):
        policy_params = {
            key: threshold
            for key, threshold in zip(
                self.unprioritized_rules_range.keys(), *rules[-1][1:]
            )
        }
        for i in range(self.num_prioritized):
            c = self.prioritized_components[i]
            policy_params.update(
                {
                    f"{c}": {
                        f"{key}": threshold
                        for key, threshold in zip(
                            self.prioritized_rules_range.keys(), *rules[i][1:]
                        )
                    }
                }
            )
        return policy_params

    def optimize_heuristics(self, num_episodes):
        p_rules_range_dimensions = [
            len(rule) for rule in self.prioritized_rules_range.values()
        ]
        np_rules_range_dimensions = [
            len(rule) for rule in self.unprioritized_rules_range.values()
        ]

        print(f"Prioritized rules range dimensions: {p_rules_range_dimensions}")
        print(f"Unprioritized rules range dimensions: {np_rules_range_dimensions}")

        # compute all possible combinations of rules
        num_rules = (
            self.num_prioritized
            * math.prod(p_rules_range_dimensions)
            * math.prod(np_rules_range_dimensions)
        )

        print(f"Number of rules: {num_rules}")

        # combine prioritized rules and prioritized components ids
        comb_prio = itertools.product(*self.prioritized_rules_range.values())
        comb_prio = itertools.product(self.prioritized_components, comb_prio)

        # combine unprioritized rules with None
        comb_unprio = itertools.product(*self.unprioritized_rules_range.values())
        comb_unprio = itertools.product([None], comb_unprio)

        all_combinations = itertools.product(comb_prio, comb_unprio)

        store_policy_rewards = np.zeros((num_episodes, num_rules))
        best_policy_idx = 0
        best_policy_reward = -np.inf
        best_policy_params = None

        for i, rules in tqdm(enumerate(all_combinations)):

            # set the rules
            policy_params = self.get_policy_params(rules)

            # parallel evaluation
            evals = parallel_rollout(
                self.env, self.policy, self.get_rollout, num_episodes, policy_params
            )

            # check if the policy is the best
            if evals.mean() > best_policy_reward:
                best_policy_reward = evals.mean()
                best_policy_idx = i
                best_policy_params = policy_params

            store_policy_rewards[:, i] = evals

        # Find the best rules corresponding to the best policy
        self.best_rules = best_policy_params

        print(f"Best rules: {self.best_rules}")

        return store_policy_rewards

    def evaluate_heuristics(self, num_episodes):

        best_policy_rewards = parallel_rollout(
            self.env, self.policy, self.get_rollout, num_episodes, self.best_rules
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

        print(f"Best rules: {self.best_rules}")

        for _ in range(num_episodes):
            total_reward = self.get_rollout(
                self.env, self.policy, verbose=True, policy_params=self.best_rules
            )

            print(f"Episode return: {total_reward/self.norm_constant:.3f}")
