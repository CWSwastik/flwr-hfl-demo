import json
import os
import re
import time
import sys
import subprocess
import psutil

CONFIG_FILE = "config.py"
# Experiments file: CLI arg > FL_EXPERIMENTS_FILE env > default.
# Lets each SLURM job point at its own json without any `cp`.
EXPERIMENTS_FILE = (
    sys.argv[1] if len(sys.argv) > 1
    else os.environ.get("FL_EXPERIMENTS_FILE", "experiments.json")
)

# Default runs per configuration when DEBUG=False.
# Override per experiment with a "NUM_RUNS" key in the json.
NUM_RUNS = 3

# Per-job identity. Children inherit FL_JOB_TAG via env, so parallel jobs on
# the same node only ever kill their own processes and read their own config
# overrides file.
JOB_TAG = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
OVERRIDES_FILE = os.path.abspath(f".fl_overrides_{JOB_TAG}.json")

def _is_stale_tag(tag):
    """A pid-style tag whose run_experiments.py is dead marks a crashed
    local batch; its orphans are fair game for cleanup."""
    if tag and tag.startswith("pid"):
        try:
            return not psutil.pid_exists(int(tag[3:]))
        except ValueError:
            return False
    return False

def kill_other_python_processes(quiet=False):
    """Finds and kills ONLY lingering FL processes belonging to THIS job
    (matching FL_JOB_TAG), plus orphans of crashed local batches.
    Other parallel jobs and manual (untagged) runs are never touched.

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
            if not (cmdline and any(script in arg for arg in cmdline for script in target_scripts)):
                continue
            tag = proc.environ().get("FL_JOB_TAG")
            if tag == JOB_TAG or _is_stale_tag(tag):
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

def get_default_debug_status():
    """Default DEBUG value from config.py, used when an experiment
    doesn't set its own "DEBUG" key."""
    with open(CONFIG_FILE, "r") as f:
        config_text = f.read()
    # Anchor to start-of-line + MULTILINE so we don't match `DEBUG = True`
    # text that appears inside the comment block at the top of config.py.
    match = re.search(r"^DEBUG\s*=\s*(True|False)", config_text, flags=re.MULTILINE)
    if match:
        return match.group(1) == "True"
    return True

def write_overrides(exp):
    # Atomic write: children read this file on every import of config.py
    tmp = OVERRIDES_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(exp, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OVERRIDES_FILE)

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
    # Children (simulate.py and everything it spawns) inherit these.
    os.environ["FL_JOB_TAG"] = JOB_TAG
    os.environ["FL_CONFIG_OVERRIDES"] = OVERRIDES_FILE
    print(f"[Job] Tag: {JOB_TAG} | Experiments: {EXPERIMENTS_FILE}")

    # 1. Kill any lingering FL processes from previous crashed runs
    kill_other_python_processes()

    try:
        if not os.path.exists(EXPERIMENTS_FILE):
            print(f"Error: {EXPERIMENTS_FILE} not found.")
            return

        with open(EXPERIMENTS_FILE, "r") as f:
            experiments = json.load(f)

        default_debug = get_default_debug_status()
        total_exps = len(experiments)
        failures = []  # list of (exp_index, run_id_or_None)

        for i, exp in enumerate(experiments):
            print(f"\n{'#'*60}")
            print(f" Running Experiment {i+1}/{total_exps}")
            print(f"{'#'*60}")

            write_overrides(exp)

            is_debug = bool(exp.get("DEBUG", default_debug))
            num_runs = int(exp.get("NUM_RUNS", NUM_RUNS))

            if is_debug:
                print("   [Mode] DEBUG=True (Single Execution)")
                if not run_simulation(run_id=None):
                    failures.append((i + 1, None))
            else:
                print(f"   [Mode] DEBUG=False (Batch Execution, {num_runs} Runs)")
                for r in range(1, num_runs + 1):
                    print(f"\n--- Cycle {r} of {num_runs} ---")
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
        if os.path.exists(OVERRIDES_FILE):
            os.remove(OVERRIDES_FILE)

        # 2. Final cleanup to ensure no background processes outlive the main script
        kill_other_python_processes()
        print(" Done.")

if __name__ == "__main__":
    main()
