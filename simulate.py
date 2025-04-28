import subprocess
import yaml
import os
import shutil
import platform
import time
import socket

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def get_abs_path(filename):
    """Get the absolute path of a file in the same directory."""
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️ Warning: {filename} not found at {path}")
    return path


def get_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def spawn_processes():
    topo_file = get_abs_path("topologies/topo-new.yml")

    if not os.path.exists(topo_file):
        print(f"❌ Error: topo.yml not found at {topo_file}")
        return

    with open(topo_file, "r") as file:
        topology = yaml.safe_load(file)

    current_os = platform.system()

    # Resolve missing ports and host references
    # 1. Assign default port to coordinator/server if not specified
    for name, cfg in topology.items():
        if cfg.get("kind") == "server":
            if not cfg.get("port"):
                auto_port = get_free_port()
                print(f"Assigning free port {auto_port} to server {name}")
                cfg["port"] = auto_port

    # 2. Resolve edge configurations
    for name, cfg in topology.items():
        if cfg.get("kind") == "edge":
            # Server side
            svr = cfg.get("server", {})
            ref = svr.get("host")
            if ref in topology:
                target = topology[ref]
                svr_host = target.get("host")
                svr_port = target.get("port")
            else:
                svr_host = ref
                svr_port = svr.get("port") or get_free_port()
            cfg["server"]["host"] = svr_host
            cfg["server"]["port"] = svr_port
            print(f"Edge {name} server -> {svr_host}:{svr_port}")

            # Client side
            cli = cfg.get("client", {})
            cli_host = cli.get("host")
            cli_port = cli.get("port") or get_free_port()
            cfg["client"]["host"] = cli_host
            cfg["client"]["port"] = cli_port
            print(f"Edge {name} client -> {cli_host}:{cli_port}")

    # 3. Resolve client configurations
    for name, cfg in topology.items():
        if cfg.get("kind") == "client":
            ref = cfg.get("host")
            if ref in topology and topology[ref].get("kind") == "edge":
                edge_cli = topology[ref]["client"]
                cfg["host"] = edge_cli.get("host")
                cfg["port"] = edge_cli.get("port")
            elif ref in topology and topology[ref].get("kind") == "server":
                server = topology[ref]
                cfg["host"] = server.get("host")
                cfg["port"] = server.get("port")
            else:
                # direct host, ensure port exists
                if not cfg.get("port"):
                    raise ValueError(f"Port not specified for client {name}")
            print(f"Client {name} -> {cfg['host']}:{cfg['port']}")

    # Sort by kind order
    order = {"server": 0, "edge": 1, "client": 2}
    sorted_topo = dict(
        sorted(topology.items(), key=lambda item: order.get(item[1].get("kind"), 99))
    )

    # Spawn processes per OS
    if current_os == "Windows":
        commands = []
        for name, cfg in sorted_topo.items():
            kind = cfg.get("kind")
            if kind == "server":
                cmd = f'py "{get_abs_path("central_server.py")}" {cfg["host"]}:{cfg["port"]}'
            elif kind == "edge":
                cmd = (
                    f'py "{get_abs_path("edge_server.py")}" --server '
                    f'{cfg["server"]["host"]}:{cfg["server"]["port"]} --client '
                    f'{cfg["client"]["host"]}:{cfg["client"]["port"]} --name {name}'
                )
            elif kind == "client":
                cmd = (
                    f'py "{get_abs_path("client.py")}" '
                    f'{cfg["host"]}:{cfg["port"]} --partition_id {cfg["partition_id"]} '
                    f'--model {cfg["model"]} --name {name}'
                )
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
        for name, cfg in sorted_topo.items():
            kind = cfg.get("kind")
            if kind == "server":
                cmd = f'python3 "{get_abs_path("central_server.py")}" {cfg["host"]}:{cfg["port"]}'
            elif kind == "edge":
                cmd = (
                    f'python3 "{get_abs_path("edge_server.py")}" --server '
                    f'{cfg["server"]["host"]}:{cfg["server"]["port"]} '
                    f'--client {cfg["client"]["host"]}:{cfg["client"]["port"]} --name {name}'
                )
            elif kind == "client":
                cmd = (
                    f'python3 "{get_abs_path("client.py")}" '
                    f'{cfg["host"]}:{cfg["port"]} --partition_id {cfg["partition_id"]} '
                    f'--model {cfg["model"]} --name {name}'
                )
            else:
                continue

            proc = subprocess.Popen(cmd, shell=True)
            procs.append((name, proc))
            print(f"Starting process {name} with command: {cmd}")
            if kind == "server":
                # give server time to initialize
                time.sleep(30)

        while procs:
            for name, p in procs[:]:
                if p.poll() is not None:
                    print(f"❌ Process {name} has ended")
                    procs.remove((name, p))
            time.sleep(5)

    else:
        print(f"❌ Unsupported OS: {current_os}")


if __name__ == "__main__":
    spawn_processes()
