import subprocess
import yaml
import os
import shutil
import platform
import time

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def get_abs_path(filename):
    """Get the absolute path of a file in the same directory."""
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️ Warning: {filename} not found at {path}")
    return path


def spawn_processes():
    topo_file = get_abs_path("topologies/topo.yml")

    if not os.path.exists(topo_file):
        print(f"❌ Error: topo.yml not found at {topo_file}")
        return

    with open(topo_file, "r") as file:
        topology = yaml.safe_load(file)

    current_os = platform.system()

    order = {"server": 0, "edge": 1, "client": 2}
    topology = dict(
        sorted(topology.items(), key=lambda item: order.get(item[1].get("kind"), 99))
    )

    if current_os == "Windows":
        commands = []

        for name, config in topology.items():
            kind = config.get("kind")
            if kind == "server":
                cmd = f'py "{get_abs_path("central_server.py")}" {config["host"]}:{config["port"]}'
            elif kind == "edge":
                cmd = (
                    f'py "{get_abs_path("edge_server.py")}" --server '
                    f'{config["server"]["host"]}:{config["server"]["port"]} --client '
                    f'{config["client"]["host"]}:{config["client"]["port"]}'
                    f" --name {name}"
                )
            elif kind == "client":
                cmd = f'py "{get_abs_path("client.py")}" {config["host"]}:{config["port"]} --partition_id {config["partition_id"]} --model {config["model"]} --name {name}'
            else:
                continue

            commands.append(
                f'new-tab --title "{name}" -p "Command Prompt" cmd /k {cmd}'
            )

        if not shutil.which("wt"):
            print("❌ Error: Windows Terminal (wt) is not installed or not in PATH.")
            return

        full_command = f'wt {" ; ".join(commands)}'
        subprocess.run(full_command, shell=True)

    elif current_os == "Linux":
        procs = []

        for name, config in topology.items():
            kind = config.get("kind")
            if kind == "server":
                cmd = f'python3 "{get_abs_path("central_server.py")}" {config["host"]}:{config["port"]}'
            elif kind == "edge":
                cmd = (
                    f'python3 "{get_abs_path("edge_server.py")}" --server '
                    f'{config["server"]["host"]}:{config["server"]["port"]} --client '
                    f'{config["client"]["host"]}:{config["client"]["port"]}'
                    f" --name {name}"
                )
            elif kind == "client":
                cmd = f'python3 "{get_abs_path("client.py")}" {config["host"]}:{config["port"]} --partition_id {config["partition_id"]} --model {config["model"]} --name {name}'
            else:
                continue

            procs.append((name, subprocess.Popen(cmd, shell=True)))
            print(f"Starting process {name}", f"with command: {cmd}")
            if kind == "server":
                time.sleep(20)

        while procs:
            for p in procs[:]:
                if p[1].poll() is not None:
                    print(f"Process {p[0]} has ended")
                    procs.remove(p)
            time.sleep(5)
    else:
        print(f"❌ Unsupported OS: {current_os}")


if __name__ == "__main__":
    spawn_processes()
