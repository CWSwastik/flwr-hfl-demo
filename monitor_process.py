import argparse
import subprocess
import time
import psutil
import sys
import threading
from logger import Logger

def monitor(pid, logger, interval=1.0, stop_event=None):
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    while not stop_event.is_set():
        try:
            cpu = proc.cpu_percent(interval=None)
            mem = proc.memory_info().rss / (1024 * 1024) # MB

            logger.log({
                "cpu_percent": cpu,
                "memory_mb": mem
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Monitor process resources.")
    parser.add_argument("--name", required=True, help="Name of the process for logging")
    parser.add_argument("--kind", required=True, choices=["client", "edge", "server"], help="Type of process (client, edge, server)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="The command to execute")

    args = parser.parse_args()

    kind_map = {
        "client": "clients",
        "edge": "edge",
        "server": "central"
    }
    subfolder = kind_map[args.kind]

    monitor_logger = Logger(
        subfolder=subfolder,
        file_path=f"{args.name}_resource.csv",
        headers=["cpu_percent", "memory_mb"],
        init_file=True
    )

    cmd = args.command
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]

    print(f"[Monitor] Starting: {cmd}")
    
    try:
        proc = subprocess.Popen(cmd)
    except Exception as e:
        print(f"[Monitor] Error starting subprocess: {e}")
        proc.terminate()
        return

    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor, args=(proc.pid, monitor_logger, 1.0, stop_event))
    monitor_thread.start()

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
    finally:
        stop_event.set()
        monitor_thread.join()

if __name__ == "__main__":
    main()
