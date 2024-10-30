import sys
import yaml
import importlib

import numpy as np

from utils.logger import Tee

from imp_act import make

def load_class_from_config(heuristic_category, heuristic_class):
    # Build the module path dynamically
    module_path = f"heuristics.{heuristic_category}"
    class_name = heuristic_class

    # Import the module
    module = importlib.import_module(module_path)

    # Get the class from the module
    heuristic_class = getattr(module, class_name)

    return heuristic_class

if __name__ == '__main__':

    # Load the YAML configuration file
    with open("config/humble_heuristic.yaml", "r") as file:
        config = yaml.safe_load(file)

    
    environment_setting = config['environment_setting']
    output_file = config['output_file']
    norm_constant = config['norm_constant']
    episodes_optimize = config['episodes_optimize']
    episodes_eval = config['episodes_eval']
    episodes_print = config['episodes_print']

    # Load the heuristic class from the configuration file
    heuristic_class = load_class_from_config(config['heuristic_category'], 
                                        config['heuristic_class'])

    # Create NumPy arrays for each rule based on min, max, and interval
    if 'rules_range' in config:
        rules_range = {
            key: np.arange(value['min'], value['max'], value['interval'])
            for key, value in config['rules_range'].items()
        }

    # Redirect print statements to both the console and a specified output file
    sys.stdout = Tee(output_file)   # Redirect print to both file and console

    env = make(environment_setting)
    heuristic_agent = heuristic_class(env, norm_constant, rules_range)

    print(f"Running environment: {environment_setting}")

    # Run all heuristic combinations
    heuristic_agent.optimize_heuristics(episodes_optimize)

    # Re-evaluate the best policy
    heuristic_agent.evaluate_heuristics(episodes_eval)

    # Print the policy
    heuristic_agent.print_policy(episodes_print)

    # Restore standard output to the console
    sys.stdout.file.close()  # Close the output file when done
    sys.stdout = sys.stdout.console  # Restore normal printing to the console


