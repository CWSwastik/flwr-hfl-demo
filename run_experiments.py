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

def kill_other_python_processes(quiet=False):
    """Finds and kills ONLY lingering Python processes related to this FL setup.

    SIGTERM first, wait, then SIGKILL anything still alive.
    """
    if not quiet:
        print("\n🧹 Sweeping for orphaned FL processes...")
    current_pid = os.getpid()

    target_scripts = [
        "simulate.py",
        "client.py",
        "edge_server.py",
        "central_server.py",
        "monitor_process.py",
    ]

    victims = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] == current_pid:
                continue
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any(script in arg for arg in cmdline for script in target_scripts):
                victims.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not victims:
        if not quiet:
            print("   ✨ No zombie FL processes found.")
        return 0

    # Phase 1: SIGTERM
    for proc in victims:
        try:
            proc.terminate()
            if not quiet:
                cmd_snippet = ' '.join(proc.info.get('cmdline') or [])[:80]
                print(f"   💀 SIGTERM PID {proc.info['pid']} ({cmd_snippet}...)")
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            pass

    # Phase 2: wait, then SIGKILL stragglers
    gone, alive = psutil.wait_procs(victims, timeout=5)
    for proc in alive:
        try:
            proc.kill()
            if not quiet:
                print(f"   ☠️  SIGKILL PID {proc.pid} (ignored SIGTERM)")
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            pass
    if alive:
        psutil.wait_procs(alive, timeout=3)

    if not quiet:
        print(f"   ✅ Cleaned up {len(victims)} FL processes.")
    time.sleep(2)  # let OS reclaim ports / VRAM
    return len(victims)

def read_config():
    with open(CONFIG_FILE, "r") as f:
        return f.read()

def write_config(text):
    # Atomic write: avoid corrupting config.py if interrupted mid-write
    tmp = CONFIG_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CONFIG_FILE)

def get_debug_mode_status(config_text):
    """
    Parses the config text to find 'DEBUG = True' or 'DEBUG = False'.
    Returns True if DEBUG is enabled, False otherwise.
    """
    # Anchor to start-of-line + MULTILINE so we don't match `DEBUG = True`
    # text that appears inside the comment block at the top of config.py.
    match = re.search(r"^DEBUG\s*=\s*(True|False)", config_text, flags=re.MULTILINE)
    if match:
        return match.group(1) == "True"
    return True

def update_config(config_text, updates):
    for key, value in updates.items():
        if isinstance(value, str):
            replacement = f'{key} = "{value}"'
        else:
            replacement = f"{key} = {value}"

        # Word-boundary + MULTILINE so `LR` does not match `BASE_LR`,
        # and `.*` only consumes the rest of the same line.
        pattern = rf"^{re.escape(key)}\s*=[^\n]*"
        if re.search(pattern, config_text, flags=re.MULTILINE):
            config_text = re.sub(pattern, replacement, config_text, count=1, flags=re.MULTILINE)
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
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed (exit {e.returncode}).")
        return False

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
        failures = []  # list of (exp_index, run_id_or_None)

        for i, exp in enumerate(experiments):
            print(f"\n{'#'*60}")
            print(f" Running Experiment {i+1}/{total_exps}")
            print(f"{'#'*60}")

            new_config = update_config(original_config, exp)
            write_config(new_config)

            is_debug = get_debug_mode_status(new_config)

            if is_debug:
                print("   [Mode] DEBUG=True (Single Execution)")
                if not run_simulation(run_id=None):
                    failures.append((i + 1, None))
            else:
                print(f"   [Mode] DEBUG=False (Batch Execution, {NUM_RUNS} Runs)")
                for r in range(1, NUM_RUNS + 1):
                    print(f"\n--- Cycle {r} of {NUM_RUNS} ---")
                    if not run_simulation(r):
                        failures.append((i + 1, r))
                    # Kill any orphans BEFORE next cycle so port/VRAM reclaim
                    # happens during the cooldown, not on top of run startup.
                    kill_other_python_processes(quiet=True)
                    print("   (Cooldown 10s...)")
                    time.sleep(10)

            print(f"   Finished Config {i+1}. Cooling down 10s...")
            kill_other_python_processes(quiet=True)
            time.sleep(10)

        # Surface failures so a "completed" batch doesn't hide silent crashes.
        if failures:
            print(f"\n{'!'*60}")
            print(f" {len(failures)} run(s) FAILED:")
            for exp_idx, run_id in failures:
                tag = f"run {run_id}" if run_id is not None else "single"
                print(f"   - Experiment {exp_idx} / {tag}")
            print(f"{'!'*60}")
        else:
            print("\n✅ All runs completed successfully.")

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
