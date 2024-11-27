# Heustics

What can you do in this repository? 

You can create, optimize, or evaluate customized heuristic rules.

## Requirements

- Python 3.9 or higher
- Required Python packages:
  - `yaml`
  - `argparse`
  - `numpy`
  - `json`
  - Custom packages (e.g., `utils.logger`, `imp_act`, `heuristics`)

## Usage

Run the script by specifying a config file name (without extension) from the `config/` directory.

```bash
python main_heuristics.py --config <config_file_name>
```

### Example

```bash
python main_heuristics.py --config humble_heuristic
```

### Configuration File

The configuration file should be a YAML file located in `config/` and contain these fields:

### Output

The output is saved to a timestamped directory within a `results/` directory in the current working directory. Files generated include:
- `output_log.txt`: Log of all printed outputs.
- `output.json`: JSON file containing evaluation results and best policy settings.