from itertools import chain

import yaml
import numpy as np
from mpi4py import MPI

from imp_act import make
from prioritized_heuristics import PrioritizedHeuristic
import os

# MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

env = None
ph = None
config = None
best_policy = None
policy_space = None


def mpi_rollout(env, heuristic, rollout_method, num_episodes, rank=0, num_procs=1):

    result = []
    for _ in range(rank, num_episodes, num_procs):
        episode_cost, _ = rollout_method(env, heuristic)
        result.append(episode_cost)

    return result


if rank == 0:

    # Environment
    env = make("Cologne-v1")

    file_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "prioritized_heuristic.yaml"
    )
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    ph = PrioritizedHeuristic(env, config)
    policy_space = ph.get_policy_space()
    episodes_optimize = config["episodes_optimize"]

    store_policy_rewards = np.zeros((episodes_optimize, len(policy_space)))
    best_policy_reward = -np.inf

env = comm.bcast(env, root=0)
ph = comm.bcast(ph, root=0)
policy_space = comm.bcast(policy_space, root=0)
config = comm.bcast(config, root=0)

for i, policy in enumerate(policy_space):

    if rank == 0:
        print(f"Policy: {i+1}/{len(policy_space)}")

    _output = mpi_rollout(
        env, policy, ph.get_rollout, config["episodes_optimize"], rank, num_procs
    )
    _all_data = comm.gather(_output, root=0)

    # Postprocess data in root process
    if rank == 0:
        evals = np.asarray(list(chain(*_all_data)))

        # check if the policy is the best
        if evals.mean() > best_policy_reward:
            best_policy_reward = evals.mean()
            best_policy = policy
            print(f"Best policy reward: {best_policy_reward/ph.norm_constant:.3f}")

        store_policy_rewards[:, i] = evals

if rank == 0:
    print(f"Best policy reward: {best_policy_reward/ph.norm_constant:.3f}")
    print(f"Best policy: {best_policy.policy_params}")
    print("Evaluating best policy...")

# Evaluate best policy
best_policy = comm.bcast(best_policy, root=0)
_output = mpi_rollout(
    env, best_policy, ph.get_rollout, config["episodes_eval"], rank, num_procs
)
_all_data = comm.gather(_output, root=0)

# Postprocess data in root process
if rank == 0:
    evals = np.asarray(list(chain(*_all_data)))

    best_policy_mean = np.mean(evals) / ph.norm_constant
    best_policy_std = np.std(evals) / ph.norm_constant
    confidence_interval = 1.96 * best_policy_std / np.sqrt(len(evals))
    print(f"Best policy with evaluated reward: {best_policy_mean:.3f}")
    print(f"Standard deviation of the best policy: {best_policy_std:.3f}")
    print(f"95% Confidence interval: {confidence_interval:.3f}")
