import subprocess
import yaml
import os
import shutil
import platform
import time
import socket
import requests
from config import (
    TOPOLOGY_FILE,
    NUM_CLIENTS, 
    CLUSTER_STRATEGY, 
    NUM_CLASSES
)
import config
import random
from utils import post_to_dashboard
from utils import load_datasets, get_dataloader_summary
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import linkage, leaves_list
import json


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EXP_ID = f"experiment_{random.randint(1000, 9999)}"


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


def create_experiment_on_dashboard(topology):
    url = f"{config.DASHBOARD_SERVER_URL}/experiment/{EXP_ID}/create"
    metadata = {
        "num_clients": config.NUM_CLIENTS,
        "rounds": config.NUM_ROUNDS,
        "averaging algorithm": "FedAvg",
        "model": config.MODEL,
        "dataset": config.DATASET,
        "batch_size": config.BATCH_SIZE,
        "topology_file": TOPOLOGY_FILE,
        "partitioner": config.PARTITIONER,
    }
    post_to_dashboard(url, metadata)

    url = f"{config.DASHBOARD_SERVER_URL}/experiment/{EXP_ID}/topology"
    post_to_dashboard(url, topology)

def get_all_distributions():
    """
    Loads all NUM_CLIENTS partitions and returns their
    normalized label distribution vectors.
    """
    print(f"Pre-loading all {NUM_CLIENTS} partitions to calculate distributions...")
    dist_vectors = []
    for pid in range(NUM_CLIENTS):
        # Load this partition's data
        # Note: This loads the *entire* dataset logic, which is fine
        # as we only iterate over the trainloader once.
        trainloader, _, _ = load_datasets(partition_id=pid)
        summary = get_dataloader_summary(trainloader)
        
        dist_counts = summary["label_distribution"]
        num_items = summary["num_items"]
        
        # Create a fixed-length probability vector
        vector = np.zeros(NUM_CLASSES)
        if num_items > 0:
            for label_str, count in dist_counts.items():
                label_int = int(label_str)
                if 0 <= label_int < NUM_CLASSES:
                    # Use normalized probabilities
                    vector[label_int] = count / num_items 
        
        dist_vectors.append(vector)
        print(f"  Loaded distribution for logical partition {pid}")
    
    return np.array(dist_vectors)

def precompute_partition_mapping():
    """
    Generates all partitions, clusters them, and returns
    a mapping from logical client ID (0..N-1) to the
    physical partition ID they should use.
    """
    if CLUSTER_STRATEGY == "none":
        print("No clustering strategy selected. Using default 1-to-1 partition mapping.")
        # Default map: logical client 0 -> partition 0, etc.
        return {i: i for i in range(NUM_CLIENTS)}

    # 1. Get all distribution vectors
    dist_vectors = get_all_distributions()
    
    # 2. Calculate pairwise distance matrix
    n = NUM_CLIENTS
    dist_matrix = np.zeros((n, n))
    
    print(f"Calculating {n*n} pairwise distances using '{CLUSTER_STRATEGY}'...")
    
    # Define class indices (0, 1, ..., 9) for EMD
    class_indices = np.arange(NUM_CLASSES) 

    for i in range(n):
        for j in range(i + 1, n):
            dist = 0.0
            if CLUSTER_STRATEGY == "emd":
                # 1D Wasserstein distance (Earth Mover's Distance)
                # We use the probability vectors as weights for the class indices
                dist = wasserstein_distance(class_indices, class_indices, 
                                            dist_vectors[i], dist_vectors[j])
            elif CLUSTER_STRATEGY == "jsd":
                # Jensen-Shannon Divergence
                dist = jensenshannon(dist_vectors[i], dist_vectors[j])
            else:
                raise ValueError(f"Unknown CLUSTER_STRATEGY: {CLUSTER_STRATEGY}")
            
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # 3. Perform hierarchical clustering
    # We need a condensed distance matrix (upper triangle) for linkage
    condensed_dist_matrix = dist_matrix[np.triu_indices(n, k=1)]
    
    print("Performing hierarchical clustering...")
    linked = linkage(condensed_dist_matrix, 'average')

    # 4. Get optimal leaf ordering
    # This re-orders the *original indices* (0..N-1) so
    # that similar partitions are adjacent in the list.
    optimal_order = leaves_list(linked)
    
    print(f"Optimal partition order (physical partition IDs): {optimal_order}")

    # 5. Create the final map
    # logical_client_id 0 -> physical_partition_id optimal_order[0]
    # logical_client_id 1 -> physical_partition_id optimal_order[1]
    # ...
    client_to_partition_map = {
        logical_id: physical_partition_id 
        for logical_id, physical_partition_id in enumerate(optimal_order)
    }
    
    print("Partition mapping created:")
    print(json.dumps(client_to_partition_map, indent=2))
    return client_to_partition_map

def spawn_processes():
    topo_file = get_abs_path(f"topologies/{TOPOLOGY_FILE}")

    if not os.path.exists(topo_file):
        print(f"❌ Error: topo.yml not found at {topo_file}")
        return

    with open(topo_file, "r") as file:
        topology = yaml.safe_load(file)

    try:
        # This map dictates which physical partition each logical client gets
        partition_map = precompute_partition_mapping()
    except Exception as e:
        print(f"❌ Error during partition pre-clustering: {e}")
        print("Falling back to default 1-to-1 mapping.")
        partition_map = {i: i for i in range(NUM_CLIENTS)}
    
    current_os = platform.system()

    create_experiment_on_dashboard(topology)

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
                cmd = f'py "{get_abs_path("central_server.py")}" {cfg["host"]}:{cfg["port"]} --exp_id {EXP_ID}'
            elif kind == "edge":
                cmd = (
                    f'py "{get_abs_path("edge_server.py")}" --server '
                    f'{cfg["server"]["host"]}:{cfg["server"]["port"]} --client '
                    f'{cfg["client"]["host"]}:{cfg["client"]["port"]} --name {name} --exp_id {EXP_ID}'
                )
            elif kind == "client":
                # Map datasets to clients based on clustering
                # 'cfg["partition_id"]' is the logical ID (0-15) from the YAML
                logical_id = cfg["partition_id"]
                # Look up the *actual* partition ID from our precomputed map
                physical_partition_id = partition_map.get(logical_id, logical_id) # Fallback
                
                print(f"Mapping client {name} (Logical ID {logical_id}) -> Physical Partition ID {physical_partition_id}")
                
                # old one reads partition based on ID in topology yaml file
                # cmd = (
                #     f'py "{get_abs_path("client.py")}" '
                #     f'{cfg["host"]}:{cfg["port"]} --partition_id {cfg["partition_id"]} '
                #     f"--name {name} --exp_id {EXP_ID}"
                # )
                
                cmd = (
                    f'py "{get_abs_path("client.py")}" '
                    f'{cfg["host"]}:{cfg["port"]} --partition_id {physical_partition_id} ' # Use new ID
                    f"--name {name} --exp_id {EXP_ID}"
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
                cmd = f'python "{get_abs_path("central_server.py")}" {cfg["host"]}:{cfg["port"]} --exp_id {EXP_ID}'
            elif kind == "edge":
                cmd = (
                    f'python "{get_abs_path("edge_server.py")}" --server '
                    f'{cfg["server"]["host"]}:{cfg["server"]["port"]} '
                    f'--client {cfg["client"]["host"]}:{cfg["client"]["port"]} --name {name} --exp_id {EXP_ID}'
                )
            elif kind == "client":
                # old one reads partition based on ID in topology yaml file
                # cmd = (
                #     f'python3 "{get_abs_path("client.py")}" '
                #     f'{cfg["host"]}:{cfg["port"]} --partition_id {cfg["partition_id"]}'
                #     f" --name {name} --exp_id {EXP_ID}"
                # )
                # Map datasets to clients based on clustering
                logical_id = cfg["partition_id"]
                physical_partition_id = partition_map.get(logical_id, logical_id) # Fallback
                
                print(f"Mapping client {name} (Logical ID {logical_id}) -> Physical Partition ID {physical_partition_id}")

                cmd = (
                    f'python "{get_abs_path("client.py")}" '
                    f'{cfg["host"]}:{cfg["port"]} --partition_id {physical_partition_id} ' # Use new ID
                    f" --name {name} --exp_id {EXP_ID}"
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

            if len(procs) == 0:
                break
            time.sleep(5)

    else:
        print(f"❌ Unsupported OS: {current_os}")


if __name__ == "__main__":
    spawn_processes()
