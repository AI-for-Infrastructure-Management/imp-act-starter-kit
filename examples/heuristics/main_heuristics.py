import sys
import yaml
import importlib
import json
import os
from datetime import datetime
import argparse

import numpy as np

from utils.logger import Tee

from imp_act import make


def load_class_from_config(config):
    # Build the module path dynamically
    module_path = f"heuristics.{config['heuristic_category']}"
    class_name = config["heuristic_class"]

    # Import the module
    module = importlib.import_module(module_path)

    # Get the class from the module
    heuristic_class = getattr(module, class_name)

    return heuristic_class


def create_timestamped_directory(identifier):
    # Set the base path to the 'results' directory in the current working directory
    base_path = os.path.join(os.getcwd(), "results")

    # Get current date and time in the format YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create the directory path with the timestamp and identifier
    dir_name = f"{identifier}_{timestamp}"
    full_path = os.path.join(base_path, dir_name)

    # Create the directory
    os.makedirs(full_path, exist_ok=True)
    return full_path


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run with specified config file")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    # Load the YAML configuration file
    config_file = "config/" + args.config + ".yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    environment_setting = config["environment_setting"]
    exp_name = config["exp_name"]
    norm_constant = config["norm_constant"]
    episodes_optimize = config["episodes_optimize"]
    episodes_eval = config["episodes_eval"]
    episodes_print = config["episodes_print"]

    # Load the heuristic class from the configuration file
    heuristic_class = load_class_from_config(config)

    env = make(environment_setting)
    heuristic_agent = heuristic_class(env, config, norm_constant)
    directory_path = create_timestamped_directory(exp_name)

    # Redirect print statements to both the console and a specified output file
    output_file = os.path.join(directory_path, "output_log.txt")
    sys.stdout = Tee(output_file)  # Redirect print to both file and console
    output_dict = {}

    print(f"Running environment: {environment_setting}")

    # Run all heuristic combinations
    _ = heuristic_agent.optimize_heuristics(episodes_optimize)

    # Re-evaluate the best policy
    _, rew_stats = heuristic_agent.evaluate_heuristics(episodes_eval)

    # Print the policy
    heuristic_agent.print_policy(episodes_print)

    # Save the best rules and policy value to a JSON file
    output_dict["return_stats"] = rew_stats
    output_dict["best_rules"] = heuristic_agent.best_rules

    with open(os.path.join(directory_path, "output.json"), "w") as file:
        json.dump(output_dict, file, indent=4, default=convert_numpy)

    # Restore standard output to the console
    sys.stdout.file.close()  # Close the output file when done
    sys.stdout = sys.stdout.console  # Restore normal printing to the console
