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
import json

# Check if we need clustering libraries (only if strategy is set)
if CLUSTER_STRATEGY in ["emd", "balanced_emd"]:
    from scipy.stats import wasserstein_distance
if CLUSTER_STRATEGY in ["jsd"]:
    from scipy.spatial.distance import jensenshannon
if CLUSTER_STRATEGY in ["emd", "jsd"]:
    from scipy.cluster.hierarchy import linkage, leaves_list


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
    try:
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
    except Exception as e:
        print(f"Dashboard error: {e}")

def get_all_partition_counts():
    """
    Loads all NUM_CLIENTS partitions and returns their
    raw label counts and totals.
    """
    print(f"Pre-loading all {NUM_CLIENTS} partitions to calculate distributions...")
    partition_data = []
    for pid in range(NUM_CLIENTS):
        trainloader, _, _ = load_datasets(partition_id=pid)
        summary = get_dataloader_summary(trainloader)
        
        dist_counts_map = summary["label_distribution"]
        num_items = summary["num_items"]
        
        # Create a fixed-length counts vector
        counts_vector = np.zeros(NUM_CLASSES)
        if num_items > 0:
            for label_str, count in dist_counts_map.items():
                label_int = int(label_str)
                if 0 <= label_int < NUM_CLASSES:
                    counts_vector[label_int] = count
        
        partition_data.append({
            "pid": pid,
            "counts": counts_vector,
            "total": num_items
        })
        # print(f"  Loaded counts for logical partition {pid}")
    
    return partition_data

def get_all_distributions():
    """
    Wrapper for get_all_partition_counts that returns
    normalized probability vectors.
    """
    partition_data = get_all_partition_counts()
    dist_vectors = []
    
    for data in partition_data:
        if data["total"] == 0:
            dist_vectors.append(np.zeros(NUM_CLASSES))
        else:
            dist_vectors.append(data["counts"] / data["total"])
            
    return np.array(dist_vectors)


def precompute_partition_mapping(topology):
    """
    Generates all partitions, clusters them based on CLUSTER_STRATEGY,
    and returns a mapping from logical client ID (0..N-1) to the
    physical partition ID they should use.
    """
    if CLUSTER_STRATEGY is None or CLUSTER_STRATEGY == "none":
        print("No clustering strategy selected. Using default 1-to-1 partition mapping.")
        return {i: i for i in range(NUM_CLIENTS)}

    # --- 'balanced_emd' STRATEGY ---
    if CLUSTER_STRATEGY == "balanced_emd":
        print("Using 'balanced_emd' strategy: creating uniform aggregate distributions per cluster.")
        
        partition_data = get_all_partition_counts()
        target_dist = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        class_indices = np.arange(NUM_CLASSES)

        edge_names = [name for name, cfg in topology.items() if cfg.get("kind") == "edge"]
        num_bins = len(edge_names)
        
        if num_bins == 0: # Fallback for flat topology
             return {i: i for i in range(NUM_CLIENTS)}

        bin_capacity = int(np.ceil(NUM_CLIENTS / num_bins))
        
        # Initialize Bins
        bins = [{
            "counts": np.zeros(NUM_CLASSES), 
            "total": 0, 
            "partitions": [], 
            "size": 0
        } for _ in range(num_bins)]
        
        # Greedy Bin Packing
        for partition in partition_data:
            pid = partition["pid"]
            counts_vector = partition["counts"]
            total_items = partition["total"]
            
            emd_scores = []
            
            for i in range(num_bins):
                if bins[i]["size"] >= bin_capacity:
                    emd_scores.append(np.inf)
                else:
                    new_counts = bins[i]["counts"] + counts_vector
                    new_total = bins[i]["total"] + total_items
                    if new_total == 0:
                        emd_scores.append(0)
                        continue
                    new_dist = new_counts / new_total
                    emd = wasserstein_distance(class_indices, class_indices, target_dist, new_dist)
                    emd_scores.append(emd)
            
            best_bin_index = np.argmin(emd_scores)
            bins[best_bin_index]["partitions"].append(pid)
            bins[best_bin_index]["counts"] += counts_vector
            bins[best_bin_index]["total"] += total_items
            bins[best_bin_index]["size"] += 1

        final_map = {}
        logical_client_id = 0
        for bin_ in bins:
            for physical_partition_id in bin_["partitions"]:
                final_map[logical_client_id] = physical_partition_id
                logical_client_id += 1
        
        return final_map

    # --- 'emd' and 'jsd' STRATEGY ---
    elif CLUSTER_STRATEGY in ["emd", "jsd"]:
        print(f"Using '{CLUSTER_STRATEGY}' strategy: grouping similar partitions.")
        dist_vectors = get_all_distributions()
        n = NUM_CLIENTS
        dist_matrix = np.zeros((n, n))
        class_indices = np.arange(NUM_CLASSES) 

        for i in range(n):
            for j in range(i + 1, n):
                dist = 0.0
                if CLUSTER_STRATEGY == "emd":
                    dist = wasserstein_distance(class_indices, class_indices, 
                                                dist_vectors[i], dist_vectors[j])
                elif CLUSTER_STRATEGY == "jsd":
                    dist = jensenshannon(dist_vectors[i], dist_vectors[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        condensed_dist_matrix = dist_matrix[np.triu_indices(n, k=1)]
        linked = linkage(condensed_dist_matrix, 'average')
        optimal_order = leaves_list(linked)

        client_to_partition_map = {
            logical_id: physical_partition_id 
            for logical_id, physical_partition_id in enumerate(optimal_order)
        }
        return client_to_partition_map

    else:
        print(f"Error: Unknown CLUSTER_STRATEGY '{CLUSTER_STRATEGY}'.")
        return {i: i for i in range(NUM_CLIENTS)}


def spawn_processes():
    topo_file = get_abs_path(f"topologies/{TOPOLOGY_FILE}")

    if not os.path.exists(topo_file):
        print(f"❌ Error: topo.yml not found at {topo_file}")
        return

    with open(topo_file, "r") as file:
        topology = yaml.safe_load(file)

    current_os = platform.system()

    create_experiment_on_dashboard(topology)

    try:
        partition_map = precompute_partition_mapping(topology)
    except Exception as e:
        print(f"❌ Error during partition pre-clustering: {e}")
        import traceback
        traceback.print_exc()
        partition_map = {i: i for i in range(NUM_CLIENTS)}

    # Resolve missing ports
    for name, cfg in topology.items():
        if cfg.get("kind") == "server":
            if not cfg.get("port"):
                cfg["port"] = get_free_port()

    for name, cfg in topology.items():
        if cfg.get("kind") == "edge":
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

            cli = cfg.get("client", {})
            cli_host = cli.get("host")
            cli_port = cli.get("port") or get_free_port()
            cfg["client"]["host"] = cli_host
            cfg["client"]["port"] = cli_port

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
                if not cfg.get("port"):
                    raise ValueError(f"Port not specified for client {name}")

    order = {"server": 0, "edge": 1, "client": 2}
    sorted_topo = dict(
        sorted(topology.items(), key=lambda item: order.get(item[1].get("kind"), 99))
    )

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
                logical_id = cfg["partition_id"]
                physical_partition_id = partition_map.get(logical_id, logical_id)
                cmd = (
                    f'py "{get_abs_path("client.py")}" '
                    f'{cfg["host"]}:{cfg["port"]} --partition_id {physical_partition_id} '
                    f"--name {name} --exp_id {EXP_ID}"
                )
            else:
                continue
            commands.append(f'new-tab --title "{name}" -p "Command Prompt" cmd /k {cmd}')

        if not shutil.which("wt"):
            print("❌ Error: Windows Terminal (wt) is not installed.")
            return

        full_command = f'wt {" ; ".join(commands)}'
        subprocess.run(full_command, shell=True)

    elif current_os == "Linux":
        procs = []
        server_process = None # --- 1. Track the Central Server ---

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
                logical_id = cfg["partition_id"]
                physical_partition_id = partition_map.get(logical_id, logical_id)
                
                print(f"Mapping client {name} -> Partition {physical_partition_id}")

                cmd = (
                    f'python "{get_abs_path("client.py")}" '
                    f'{cfg["host"]}:{cfg["port"]} --partition_id {physical_partition_id} '
                    f" --name {name} --exp_id {EXP_ID}"
                )
            else:
                continue

            proc = subprocess.Popen(cmd, shell=True)
            procs.append((name, proc))
            print(f"Starting {name}...")
            
            # --- 2. Identify the server process ---
            if kind == "server":
                server_process = proc
                time.sleep(30) # Give server head start

        # --- 3. UPDATED WAIT LOGIC ---
        if server_process:
            print(f"\nExample {EXP_ID} running. Waiting for Central Server to complete...")
            
            # Wait ONLY for the central server to finish
            server_process.wait()
            
            print("\n✅ Central Server finished. Terminating all other processes...")
            
            # Kill everyone else
            for name, p in procs:
                if p.poll() is None: # If still running
                    print(f"Stopping {name}...")
                    p.terminate()
                    p.wait() # Ensure it dies
        else:
            # Fallback if no server found (unexpected)
            print("Warning: No central server process found. Waiting for all processes...")
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