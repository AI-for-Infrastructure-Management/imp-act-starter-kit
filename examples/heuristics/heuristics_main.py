import sys

import numpy as np

from heuristics.standard_heuristics import HumbleHeuristic
from utils.logger import Tee

from imp_act import make

if __name__ == '__main__':
    
    environment_setting = "ToyExample-v2"
    output_file = "option_out.txt"
    norm_constant = 1e6

    episodes_optimize = 100
    episodes_eval = 1000
    episodes_print = 1 

    rules_range = {
        'replacement_threshold': np.arange(5, 6, 1),
        'major_repair_threshold': np.arange(5, 6, 1),
        'minor_repair_threshold': np.arange(2, 4, 1),
        'inspection_interval': np.arange(1, 51, 5)
    }

    # Redirect print statements to both the console and a specified output file
    sys.stdout = Tee(output_file)   # Redirect print to both file and console

    env = make(environment_setting)
    heuristic_agent = HumbleHeuristic(env, norm_constant, rules_range)

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


