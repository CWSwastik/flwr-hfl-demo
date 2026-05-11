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
from utils import load_datasets, get_dataloader_summary, post_to_dashboard
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
import json
import pandas as pd
from clustering_utils import (
    parse_topology_for_clustering, 
    cluster_clients_by_distribution
)
from collections import defaultdict


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

def get_all_partition_counts():
    """
    Loads all NUM_CLIENTS partitions and returns their
    raw label counts and totals as a list of dictionaries.
    This is required for the saving function.
    """
    print(f"Pre-loading all {NUM_CLIENTS} partitions to calculate distributions...")
    partition_data = []
    for pid in range(NUM_CLIENTS):
        # Load this partition's data
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
    normalized probability vectors (NumPy array).
    """
    partition_data = get_all_partition_counts()
    dist_vectors = []
    
    for data in partition_data:
        if data["total"] == 0:
            dist_vectors.append(np.zeros(NUM_CLASSES))
        else:
            dist_vectors.append(data["counts"] / data["total"])
            
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
                # Jensen-Shannon Divergence, adding 1e-10 to avoid zero probabilities
                dist = jensenshannon(dist_vectors[i]+ 1e-10, dist_vectors[j]+ 1e-10)
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
    
    print(f"Optimal partition order (original partition IDs): {optimal_order}")

    # 5. Create the final map
    # logical_client_id 0 -> physical_partition_id optimal_order[0]
    # logical_client_id 1 -> physical_partition_id optimal_order[1]
    # ...
    client_to_partition_map = {
        int(logical_id): int(physical_partition_id) 
        for logical_id, physical_partition_id in enumerate(optimal_order)
    }
    
    print("Partition mapping created:")
    print(json.dumps(client_to_partition_map, indent=2))
    return client_to_partition_map

def save_distribution_mapping(topology, partition_map):
    """
    Saves three CSV files using Pandas:
    1. client_distribution_pre_clustering.csv (Logical ID -> Original Data)
    2. client_distribution_post_clustering.csv (Logical ID -> Re-mapped Data)
    3. edge_cluster_distribution.csv (Edge Server -> Aggregated Data)
    """
    print("Fetching partition counts for distribution reports...")
    
    # 1. Fetch Raw Data
    partition_data = get_all_partition_counts()
    # Lookup: physical_pid -> counts_vector
    partition_lookup = {p['pid']: p['counts'] for p in partition_data}
    
    # Prepare Data Structures
    pre_cluster_data = []
    post_cluster_data = []
    edge_aggregation = {}  # edge_name -> numpy array of counts

    # 2. Iterate Topology
    sorted_items = sorted(topology.items(), key=lambda x: x[0])
    
    for name, cfg in sorted_items:
        if cfg.get("kind") == "client":
            logical_id = cfg["partition_id"]
            
            # --- A. Pre-Clustering (Logical ID == Physical ID) ---
            pre_counts = partition_lookup.get(logical_id, np.zeros(NUM_CLASSES))
            pre_row = {
                "ClientName": name,
                "LogicalID": logical_id,
                "TotalSamples": int(np.sum(pre_counts))
            }
            for i, count in enumerate(pre_counts):
                pre_row[f"Class_{i}"] = int(count)
            pre_cluster_data.append(pre_row)

            # --- B. Post-Clustering (Mapped ID) ---
            physical_id = partition_map.get(logical_id, logical_id)
            post_counts = partition_lookup.get(physical_id, np.zeros(NUM_CLASSES))
            
            # Identify Edge Server
            edge_server = "Unknown"
            host_ref = cfg.get("host") 
            if host_ref in topology and topology[host_ref].get("kind") == "edge":
                edge_server = host_ref
            
            post_row = {
                "ClientName": name,
                "EdgeServer": edge_server,
                "LogicalID": logical_id,
                "PhysicalPartitionID": physical_id,
                "TotalSamples": int(np.sum(post_counts))
            }
            for i, count in enumerate(post_counts):
                post_row[f"Class_{i}"] = int(count)
            post_cluster_data.append(post_row)

            # --- C. Edge Aggregation ---
            if edge_server not in edge_aggregation:
                edge_aggregation[edge_server] = np.zeros(NUM_CLASSES)
            edge_aggregation[edge_server] += post_counts

    # 3. Save Files
    log_dir = os.path.join(BASE_DIR, "logs", config.EXPERIMENT_NAME)
    os.makedirs(log_dir, exist_ok=True)

    # Save Pre-Clustering
    df_pre = pd.DataFrame(pre_cluster_data)
    df_pre.to_csv(os.path.join(log_dir, "distribution_pre_clustering.csv"), index=False)
    
    # Save Post-Clustering
    df_post = pd.DataFrame(post_cluster_data)
    df_post.to_csv(os.path.join(log_dir, "distribution_post_clustering.csv"), index=False)

    # Save Edge Aggregation
    edge_rows = []
    for edge, counts in edge_aggregation.items():
        row = {"EdgeServer": edge, "TotalSamples": int(np.sum(counts))}
        for i, count in enumerate(counts):
            row[f"Class_{i}"] = int(count)
        edge_rows.append(row)
    
    df_edge = pd.DataFrame(edge_rows)
    df_edge.to_csv(os.path.join(log_dir, "distribution_edge_clusters.csv"), index=False)

    print(f"✅ Saved 3 distribution CSVs to {log_dir}")

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
        print("✅ Partition pre-clustering completed.\nSaving distributions") 
        save_distribution_mapping(topology, partition_map)
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
