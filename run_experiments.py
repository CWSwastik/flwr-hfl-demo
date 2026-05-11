import json
import subprocess
import shutil
import os
import re
import time
import sys
import signal
import psutil

CONFIG_FILE = "config.py"
BACKUP_FILE = "config_backup.py"
EXPERIMENTS_FILE = "experiments.json"

# Set how many runs you want per configuration (Only used if DEBUG=False)
NUM_RUNS = 3 

def kill_other_python_processes():
    """Finds and kills ONLY lingering Python processes related to this FL setup."""
    print("\n🧹 Sweeping for orphaned FL processes...")
    current_pid = os.getpid()
    killed_count = 0
    
    # The specific scripts we want to hunt down and kill
    target_scripts = [
        "simulate.py", 
        "client.py", 
        "edge_server.py", 
        "central_server.py", 
        "monitor_process.py"
    ]

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] == current_pid:
                continue
            
            # Grab the exact command used to launch this process
            cmdline = proc.info.get('cmdline', [])
            
            if cmdline:
                # Check if any of our target scripts are in the command line arguments
                is_target = any(script in arg for arg in cmdline for script in target_scripts)
                
                if is_target:
                    os.kill(proc.info['pid'], signal.SIGTERM)
                    # Print a snippet of the command so you know exactly what was killed
                    cmd_snippet = ' '.join(cmdline[:3])
                    print(f"   💀 Killed PID {proc.info['pid']} ({cmd_snippet}...)")
                    killed_count += 1
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError, psutil.ZombieProcess):
            pass
            
    if killed_count == 0:
        print("   ✨ No zombie FL processes found.")
    else:
        print(f"   ✅ Cleaned up {killed_count} FL processes.")
        time.sleep(2) # Give the OS a moment to reclaim ports and VRAM

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
    return True

def update_config(config_text, updates):
    for key, value in updates.items():
        if isinstance(value, str):
            replacement = f'{key} = "{value}"'
        else:
            replacement = f"{key} = {value}"

        pattern = rf"{key}\s*=.*"
        if re.search(pattern, config_text):
            config_text = re.sub(pattern, replacement, config_text, count=1)
        else:
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
        subprocess.run([sys.executable, "simulate.py"], env=env, check=True)
        print(f"   ✅ Finished.")
    except subprocess.CalledProcessError:
        print(f"   ❌ Failed.")

def main():
    # 1. Kill any lingering FL processes from previous crashed runs
    kill_other_python_processes()

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

            new_config = update_config(original_config, exp)
            write_config(new_config)
            
            is_debug = get_debug_mode_status(new_config)

            if is_debug:
                print("   [Mode] DEBUG=True (Single Execution)")
                run_simulation(run_id=None)
            else:
                print(f"   [Mode] DEBUG=False (Batch Execution, {NUM_RUNS} Runs)")
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
            
        # 2. Final cleanup to ensure no background processes outlive the main script
        kill_other_python_processes()
        print(" Done.")

if __name__ == "__main__":
    main()
