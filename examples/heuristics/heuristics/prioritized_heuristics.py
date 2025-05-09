import math
import itertools

import numpy as np
from tqdm import tqdm

from heuristic import Heuristic, parallel_rollout


class PrioritizedPolicy:

    def __init__(self, policy_params, priritized_components, component_id_map):
        self.policy_params = policy_params
        self.prioritized_components = priritized_components
        self.component_id_map = component_id_map

    def __call__(self, obs):

        edge_obs = obs["edge_observations"]
        current_time = obs["time_step"]

        actions = []
        for comp_ids, road_edge in zip(self.component_id_map, edge_obs):
            edge_actions = []
            for c, o in zip(comp_ids, road_edge):

                if c in self.prioritized_components:
                    _params = self.policy_params[c]
                else:
                    _params = self.policy_params

                a = self._policy(_params, o, current_time)
                edge_actions.append(a)
            actions.append(edge_actions)

        return actions

    def _policy(self, thresholds, o, t):

        if o >= thresholds["replacement_threshold"]:
            return 4  # Reconstruction
        elif o >= thresholds["major_repair_threshold"]:
            return 3  # Major repair
        elif o >= thresholds["minor_repair_threshold"]:
            return 2  # Minor repair
        elif t % thresholds["inspection_interval"] == 0:
            return 1  # Inspection
        else:
            return 0  # Do nothing


class PrioritizedHeuristic(Heuristic):

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.norm_constant = config["norm_constant"]
        self.rules_range = config["rules_range"]

        # prioritized components
        self.prioritized_components = self.rules_range["prioritized"]["components"]
        prio_range = {
            k: v
            for k, v in config["rules_range"]["prioritized"].items()
            if k != "components"  # remove components key
        }
        self.num_prioritized = len(self.prioritized_components)
        self.prioritized_rules_range = {
            f"{key}": np.arange(value["min"], value["max"], value["interval"])
            for key, value in prio_range.items()
        }
        # default rules
        self.default_rules_range = {
            key: np.arange(value["min"], value["max"], value["interval"])
            for key, value in self.rules_range["default"].items()
        }

        print(
            f"Prioritized components ({self.num_prioritized}): {self.prioritized_components}"
        )

        self.segment_id_map = env.segment_id_map
        self.best_rules = None

    def get_policy_params(self, rules):

        # default rules
        policy_params = {
            key: threshold
            for key, threshold in zip(self.default_rules_range.keys(), *rules[-1][1:])
        }

        # prioritized rules
        for i, c in enumerate(self.prioritized_components):
            policy_params.update(
                {
                    c: {
                        f"{key}": threshold
                        for key, threshold in zip(
                            self.prioritized_rules_range.keys(), *rules[i][1:]
                        )
                    }
                }
            )
        return policy_params

    def get_policy_space(self):
        p_rules_range_dimensions = [
            len(rule) for rule in self.prioritized_rules_range.values()
        ]
        np_rules_range_dimensions = [
            len(rule) for rule in self.default_rules_range.values()
        ]

        # compute all possible combinations of rules
        num_prioritized_rules = (
            math.prod(p_rules_range_dimensions) ** self.num_prioritized
        )
        num_default_rules = math.prod(np_rules_range_dimensions)
        num_rules = num_prioritized_rules * num_default_rules

        print(
            f"Prioritized rules range dimensions: {p_rules_range_dimensions} ({num_prioritized_rules})"
        )
        print(
            f"Default rules range dimensions: {np_rules_range_dimensions} ({num_default_rules})"
        )

        print(
            f"Number of rules: {num_rules} ({num_prioritized_rules} x {num_default_rules})"
        )

        # combine prioritized rules and prioritized components ids
        comb_prio = itertools.product(*self.prioritized_rules_range.values())
        comb_prio = itertools.product(self.prioritized_components, comb_prio)

        # combine default rules with None
        comb_unprio = itertools.product(*self.default_rules_range.values())
        comb_unprio = itertools.product([None], comb_unprio)

        all_combinations = itertools.product(comb_prio, comb_unprio)

        # Initialize policy space
        policy_space = []
        for rules in all_combinations:
            policy_params = self.get_policy_params(rules)
            policy = PrioritizedPolicy(
                policy_params, self.prioritized_components, self.segment_id_map
            )
            policy_space.append(policy)

        return policy_space

    def optimize_heuristics(self, num_episodes):

        self.policy_space = self.get_policy_space()

        store_policy_rewards = np.zeros((num_episodes, len(self.policy_space)))
        best_policy_reward = -np.inf

        for i, policy in tqdm(enumerate(self.policy_space)):

            # parallel evaluation
            evals, _ = parallel_rollout(
                self.env,
                policy,
                self.get_rollout,
                num_episodes,
            )

            # check if the policy is the best
            if evals.mean() > best_policy_reward:
                best_policy_reward = evals.mean()
                best_policy = policy

            store_policy_rewards[:, i] = evals

        # Find the best rules corresponding to the best policy
        self.best_policy = best_policy
        self.best_rules = self.best_policy.policy_params

        print(f"Best policy reward: {best_policy_reward/self.norm_constant:.3f}")

        return store_policy_rewards

    def evaluate_heuristics(self, num_episodes):

        # Re-evaluate the best policy
        best_policy_rewards, infos = parallel_rollout(
            self.env,
            self.best_policy,
            self.get_rollout,
            num_episodes,
        )

        best_policy_mean = np.mean(best_policy_rewards) / self.norm_constant
        best_policy_std = np.std(best_policy_rewards) / self.norm_constant
        confidence_interval = 1.96 * best_policy_std / np.sqrt(num_episodes)
        print(f"Best policy with evaluated reward: {best_policy_mean:.3f}")
        print(f"Standard deviation of the best policy: {best_policy_std:.3f}")
        print(f"95% Confidence interval: {confidence_interval:.3f}")

        maintenance_rewards_mean = (
            np.mean(infos["total_maintenance_reward"]) / self.norm_constant
        )
        travel_time_rewards_mean = (
            np.mean(infos["total_travel_time_reward"]) / self.norm_constant
        )
        terminal_rewards_mean = (
            np.mean(infos["total_terminal_reward"]) / self.norm_constant
        )

        print(f"Mean maintenance rewards: {maintenance_rewards_mean:.3f}")
        print(f"Mean terminal rewards: {terminal_rewards_mean:.3f}")
        print(f"Mean travel time rewards: {travel_time_rewards_mean:.3f}")

        residual = (
            maintenance_rewards_mean
            + terminal_rewards_mean
            + travel_time_rewards_mean
            - best_policy_mean
        )
        if abs(residual) > 1e-6:
            print(f"Total mean reward: {residual:.3f}")

        reward_stats = [best_policy_mean, best_policy_std, confidence_interval]
        return best_policy_rewards, reward_stats

    def print_policy(self, num_episodes):

        print(f"Best rules: {self.best_rules}")

        for _ in range(num_episodes):
            total_reward, _ = self.get_rollout(
                self.env, self.best_policy, verbose=False
            )

            print(f"Episode return: {total_reward/self.norm_constant:.3f}")
