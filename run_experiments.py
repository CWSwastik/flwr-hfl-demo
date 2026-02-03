import json
import subprocess
import shutil
import os
import re
import time
import sys

CONFIG_FILE = "config.py"
BACKUP_FILE = "config_backup.py"
EXPERIMENTS_FILE = "experiments.json"

# Set how many runs you want per configuration (Only used if DEBUG=False)
NUM_RUNS = 3 

def read_config():
    with open(CONFIG_FILE, "r") as f:
        return f.read()

def write_config(text):
    with open(CONFIG_FILE, "w") as f:
        f.write(text)

def get_debug_mode_status(config_text):
    """
    Parses the config text to find 'DEBUG = True' or 'DEBUG = False'.
    Returns True if DEBUG is enabled, False otherwise.
    """
    match = re.search(r"DEBUG\s*=\s*(True|False)", config_text)
    if match:
        return match.group(1) == "True"
    # Default to True if not found for safety
    return True

def update_config(config_text, updates):
    for key, value in updates.items():
        if isinstance(value, str):
            replacement = f'{key} = "{value}"'
        else:
            # Handle boolean/numbers
            replacement = f"{key} = {value}"

        pattern = rf"{key}\s*=.*"
        print(f"pattern: {pattern}, replacement: {replacement}")
        if re.search(pattern, config_text):
            config_text = re.sub(pattern, replacement, config_text, count=1)
        else:
            # If variable doesn't exist, append it
            config_text += f"\n{replacement}\n"
    return config_text

def run_simulation(run_id=None):
    env = os.environ.copy()
    
    if run_id is not None:
        print(f"   ▶️  Launching Run {run_id}...")
        env["FL_RUN_ID"] = str(run_id)
    else:
        print(f"   ▶️  Launching Single Debug Run...")
    
    try:
        # sys.executable ensures we use the same python interpreter (conda/venv)
        subprocess.run([sys.executable, "simulate.py"], env=env, check=True)
        print(f"   ✅ Finished.")
    except subprocess.CalledProcessError:
        print(f"   ❌ Failed.")

def main():
    if not os.path.exists(BACKUP_FILE):
        shutil.copy(CONFIG_FILE, BACKUP_FILE)

    original_config = read_config()

    try:
        if not os.path.exists(EXPERIMENTS_FILE):
            print(f"Error: {EXPERIMENTS_FILE} not found.")
            return

        with open(EXPERIMENTS_FILE, "r") as f:
            experiments = json.load(f)

        total_exps = len(experiments)
        
        for i, exp in enumerate(experiments):
            print(f"\n{'#'*60}")
            print(f" Running Experiment {i+1}/{total_exps}")
            print(f"{'#'*60}")
            # print(f"Settings: {json.dumps(exp, indent=2)}\n")

            # 1. Update config.py with current experiment settings
            new_config = update_config(original_config, exp)
            write_config(new_config)
            
            # 2. Check if we are in DEBUG mode or BATCH mode based on config.py
            is_debug = get_debug_mode_status(new_config)

            if is_debug:
                print("   [Mode] DEBUG=True (Single Execution)")
                # Run once, no run_id env var needed
                run_simulation(run_id=None)
            else:
                print(f"   [Mode] DEBUG=False (Batch Execution, {NUM_RUNS} Runs)")
                # Run loop
                for r in range(1, NUM_RUNS + 1):
                    print(f"\n--- Cycle {r} of {NUM_RUNS} ---")
                    run_simulation(r)
                    print("   (Cooldown 5s...)")
                    time.sleep(5)

            print(f"   Finished Config {i+1}. Cooling down 5s...")
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user.")
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
    finally:
        print(f"\n{'='*60}")
        print(" Restoring original config.py...")
        write_config(original_config)
        if os.path.exists(BACKUP_FILE):
            os.remove(BACKUP_FILE)
        print(" Done.")

if __name__ == "__main__":
    main()