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
    NUM_CLASSES,
    MIN_CLIENTS_PER_EDGE
)
import config
import random
from utils import load_datasets, get_dataloader_summary, post_to_dashboard
import json
from clustering_utils import (
    parse_topology_for_clustering, 
    cluster_clients_by_distribution,
    assign_clusters_to_edge_servers
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
    if not config.ENABLE_DASHBOARD: return
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

'''
def compute_cluster_aware_partition_map(topo_file):
    """
    1. Parses topology to find Edges and Clients.
    2. Clusters data partitions into K clusters (K = num_edges).
    3. Maps Cluster_i partitions -> Clients of Edge_i.
    """
    print(f"--- Starting Cluster-Aware Partition Mapping ({CLUSTER_STRATEGY}) ---")
    
    # 1. Parse Topology
    topo_info = parse_topology_for_clustering(topo_file)
    num_edges = topo_info['num_edge_servers']
    edge_server_names = sorted(topo_info['edge_servers']) # Sort for deterministic assignment
    edge_to_clients = topo_info['edge_to_clients']

    if num_edges == 0:
        print("No edge servers found. Returning 1-to-1 mapping.")
        return {i: i for i in range(NUM_CLIENTS)}

    # 2. Run Clustering (uses clustering_utils)
    # This saves the CSVs and gives us the cluster assignments for partitions
    save_dir = os.path.join(BASE_DIR, "logs", config.EXPERIMENT_NAME)
    
    cluster_results = cluster_clients_by_distribution(
        num_clusters=num_edges,
        distance_metric=CLUSTER_STRATEGY, # 'emd', 'jsd', or 'none'
        save_dir=save_dir,
        topology_info=topo_info
    )
    
    # partition_cluster_map: {PartitionID: ClusterID}
    partition_cluster_map = cluster_results['cluster_assignments'] 
    
    # 3. Group Partitions by ClusterID
    # cluster_partitions[1] = [pid1, pid5, ...]
    cluster_partitions = defaultdict(list)
    for pid, cid in partition_cluster_map.items():
        cluster_partitions[cid].append(pid)

    # 4. Map Clusters to Edges
    # We have clusters 1..K and Edges E1..EK
    # We map Cluster 1 -> Edge 1, Cluster 2 -> Edge 2, etc.
    cluster_ids = sorted(cluster_partitions.keys()) 
    
    final_client_partition_map = {}

    print("\n🔗 Mapping Data Clusters to Edge Servers:")
    
    for i, edge_name in enumerate(edge_server_names):
        if i >= len(cluster_ids): 
            print(f"⚠️ Warning: More edges than clusters. Edge {edge_name} gets no special mapping.")
            break
        
        cluster_id = cluster_ids[i]
        partitions_in_cluster = cluster_partitions[cluster_id]
        clients_on_edge = edge_to_clients[edge_name] # List of client names
        
        print(f"  Edge '{edge_name}' <==> Cluster {cluster_id} (Partitions: {partitions_in_cluster})")
        
        # Assign partitions to clients
        # We iterate through the clients attached to this edge and give them 
        # the partitions that belong to this cluster.
        for j, client_name in enumerate(clients_on_edge):
            if j < len(partitions_in_cluster):
                # Get the logical ID defined in topology for this client name
                client_info = topo_info['client_info'].get(client_name)
                if client_info:
                    logical_pid = client_info['partition_id']
                    if logical_pid is not None:
                        physical_pid = partitions_in_cluster[j]
                        final_client_partition_map[int(logical_pid)] = int(physical_pid)
            else:
                 print(f"    ⚠️ Not enough partitions in Cluster {cluster_id} for all clients on {edge_name}")

    # Fill in any missing clients with 1-to-1 to avoid crash
    for i in range(NUM_CLIENTS):
        if i not in final_client_partition_map:
            final_client_partition_map[i] = i

    print(f"Final Mapping: {json.dumps(final_client_partition_map, indent=2)}")
    return final_client_partition_map
'''
'''
def spawn_processes():
    topo_file = get_abs_path(f"topologies/{TOPOLOGY_FILE}")

    if not os.path.exists(topo_file):
        print(f"❌ Error: topo.yml not found at {topo_file}")
        return

    with open(topo_file, "r") as file:
        topology = yaml.safe_load(file)

    try:
        if CLUSTER_STRATEGY != "none":
            partition_map = compute_cluster_aware_partition_map(topo_file)
        else:
            print("Cluster strategy is 'none'. Using 1-to-1 mapping.")
            partition_map = {i: i for i in range(NUM_CLIENTS)}
            # Still save distributions for reference
            save_dir = os.path.join(BASE_DIR, "logs", config.EXPERIMENT_NAME)
            topo_info = parse_topology_for_clustering(topo_file)
            # Dummy clustering call just to generate CSVs with k=1
            cluster_clients_by_distribution(NUM_CLIENTS, 'none', save_dir, topo_info)

    except Exception as e:
        print(f"❌ Error during clustering: {e}")
        import traceback
        traceback.print_exc()
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
'''

def compute_client_to_edge_mapping(topo_file):
    print(f"--- Starting Topology Analysis ({CLUSTER_STRATEGY}) ---")
    topo_info = parse_topology_for_clustering(topo_file)
    num_edges = topo_info['num_edge_servers']
    if num_edges == 0: return {}

    save_dir = os.path.join(BASE_DIR, "logs", config.EXPERIMENT_NAME)
    
    cluster_results = cluster_clients_by_distribution(
        num_clusters=num_edges,
        distance_metric=CLUSTER_STRATEGY,
        save_dir=save_dir,
        topology_info=topo_info
    )
    
    cluster_assignments = cluster_results['cluster_assignments'] 
    cluster_to_edge = assign_clusters_to_edge_servers(topo_info, cluster_results)
    partition_to_edge_target = {}
    
    print("\n🔗 Re-routing Clients based on Data:")
    for pid, cid in cluster_assignments.items():
        target_edge = cluster_to_edge.get(cid)
        if target_edge:
            partition_to_edge_target[pid] = target_edge

    print(f"partition_to_edge_target: {json.dumps(partition_to_edge_target, indent=2)}")
    return partition_to_edge_target

def print_topology_summary(edge_client_counts, client_assignments, edge_configs, msg="After Clustering"):
    """
    Displays a formatted summary of the new topology in the terminal.
    """
    print("\n" + "="*80)
    print(f"🚀 TOPOLOGY SUMMARY {msg} | Strategy: {CLUSTER_STRATEGY}")
    print("="*80)
    
    # 1. Edge Server Summary
    print(f"\n{ 'Edge Server Name':<20} | { 'Host:Port':<20} | { 'Min Clients':<12} | { 'Status':<10}")
    print("-" * 75)
    
    for edge_name, count in edge_client_counts.items():
        config = edge_configs.get(edge_name, {"host": "???", "port": "???"})
        address = f"{config['host']}:{config['port']}"
        status = "ACTIVE" if count > 0 else "IDLE"
        print(f"{edge_name:<20} | {address:<20} | {count:<12} | {status:<10}")

    # 2. Client Distribution Summary (Brief)
    print("\n📊 Client Assignments:")
    # Group clients by edge for display
    edge_groups = defaultdict(list)
    for client, details in client_assignments.items():
        edge_groups[details['target_edge']].append(f"{client}(P{details['pid']})")
    
    for edge, clients in edge_groups.items():
        # chunk clients for display
        client_str = ", ".join(clients)
        if len(client_str) > 80: client_str = client_str[:77] + "..."
        print(f"  📌 {edge}: {client_str}")
        
    print("="*80 + "\n")

def spawn_processes():
    topo_file = get_abs_path(f"topologies/{TOPOLOGY_FILE}")
    if not os.path.exists(topo_file):
        print(f"❌ Error: topo.yml not found at {topo_file}")
        return

    with open(topo_file, "r") as file:
        topology = yaml.safe_load(file)

    # --- STEP 1: RESOLVE PORTS ---
    edge_configs = {} 
    
    # Resolve Central Server
    for name, cfg in topology.items():
        if cfg.get("kind") == "server":
            if not cfg.get("port"):
                auto_port = get_free_port()
                print(f"Assigning free port {auto_port} to server {name}")
                cfg["port"] = auto_port

    # Resolve Edges
    for name, cfg in topology.items():
        if cfg.get("kind") == "edge":
            # Upstream
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
            
            edge_configs[name] = {"host": cfg["client"]["host"], "port": cfg["client"]["port"]}
            print(f"Recorded Edge Config: {name} -> {edge_configs[name]}")

    # --- STEP 1.5: PRINT INITIAL TOPOLOGY ---
    initial_edge_counts = {edge: 0 for edge in edge_configs}
    initial_assignments = {}

    for name, cfg in topology.items():
        if cfg.get("kind") == "client":
            logical_pid = cfg.get("partition_id")
            final_edge = "Unknown"
            
            # Static Lookup by Host:Port or Name Ref
            ref = cfg.get("host")
            
            # Direct Name Ref
            if ref in topology and topology[ref].get("kind") == "edge":
                final_edge = ref
                # Sync port if it was a reference
                edge_cli = topology[ref]["client"]
                cfg["host"] = edge_cli.get("host")
                cfg["port"] = edge_cli.get("port")
            
            # Host:Port match
            if final_edge == "Unknown":
                c_host = cfg.get("host")
                c_port = cfg.get("port")
                for edge_name, edge_cfg in edge_configs.items():
                     if str(edge_cfg["host"]) == str(c_host) and int(edge_cfg["port"]) == int(c_port):
                        final_edge = edge_name
                        break
            
            if final_edge in initial_edge_counts:
                initial_edge_counts[final_edge] += 1
            
            initial_assignments[name] = {"pid": logical_pid, "target_edge": final_edge}

    print_topology_summary(initial_edge_counts, initial_assignments, edge_configs, msg="Before Clustering")
    
    # --- STEP 2: CLUSTERING & MAPPING ---
    try:
        partition_to_edge_target = compute_client_to_edge_mapping(topo_file)
    except Exception as e:
        print(f"❌ Error during clustering: {e}")
        import traceback
        traceback.print_exc()
        partition_to_edge_target = {}

    # --- STEP 3: UPDATE TOPOLOGY & COUNT ---
    edge_client_counts = {edge: 0 for edge in edge_configs} # Initialize all edges to 0
    client_assignments = {} # For display purposes

    for name, cfg in topology.items():
        if cfg.get("kind") == "client":
            logical_pid = cfg.get("partition_id")
            final_edge = "Unknown"
            
            # A. Dynamic Re-routing
            if logical_pid in partition_to_edge_target:
                target_edge_name = partition_to_edge_target[logical_pid]
                if target_edge_name in edge_configs:
                    cfg["host"] = edge_configs[target_edge_name]["host"]
                    cfg["port"] = edge_configs[target_edge_name]["port"]
                    final_edge = target_edge_name
            
            # B. Static Fallback
            if final_edge == "Unknown":
                ref = cfg.get("host")
                if ref in topology and topology[ref].get("kind") == "edge":
                    final_edge = ref
                    # Sync port if it was a reference
                    edge_cli = topology[ref]["client"]
                    cfg["host"] = edge_cli.get("host")
                    cfg["port"] = edge_cli.get("port")

            # Update Counts
            if final_edge in edge_client_counts:
                edge_client_counts[final_edge] += 1
            
            client_assignments[name] = {"pid": logical_pid, "target_edge": final_edge}
            print(f"Client {name} (Logical ID {logical_pid}) assigned to Edge {final_edge}")

    # --- STEP 4: PRINT SUMMARY ---
    print_topology_summary(edge_client_counts, client_assignments, edge_configs)

    min_edges = sum(1 for count in edge_client_counts.values() if count > 0)
    
    # Wait for user validation (optional)
    print("🚀 Spawning processes in 3 seconds...")
    time.sleep(3)

    # --- STEP 5: SPAWN ---
    current_os = platform.system()
    if config.ENABLE_DASHBOARD:
        create_experiment_on_dashboard(topology)
    
    order = {"server": 0, "edge": 1, "client": 2}
    sorted_topo = dict(sorted(topology.items(), key=lambda item: order.get(item[1].get("kind"), 99)))

    if current_os == "Windows":
        commands = []
        for name, cfg in sorted_topo.items():
            kind = cfg.get("kind")
            if kind == "server":
                cmd = f'py "{get_abs_path("central_server.py")}" {cfg["host"]}:{cfg["port"]} --exp_id {EXP_ID} --min_edges {min_edges} --enable_dashboard {config.ENABLE_DASHBOARD}'
            elif kind == "edge":
                # STRICT COUNT: defaults to 0 if not in list, preventing hangs on unused edges
                required_clients = edge_client_counts.get(name, 0)
                if required_clients == 0:
                    print(f"⚠️ Edge server {name} has 0 assigned clients. It is not invoked to prevent hangs.")
                cmd = (
                    f'py "{get_abs_path("edge_server.py")}" --server '
                    f'{cfg["server"]["host"]}:{cfg["server"]["port"]} --client '
                    f'{cfg["client"]["host"]}:{cfg["client"]["port"]} --name {name} --exp_id {EXP_ID} '
                    f'--min_clients {required_clients}' # Pass EXACT calculated count
                )
            elif kind == "client":
                cmd = (
                    f'py "{get_abs_path("client.py")}" '
                    f'{cfg["host"]}:{cfg["port"]} --partition_id {cfg["partition_id"]} '
                    f"--name {name} --exp_id {EXP_ID}"
                )
            else:
                continue
            commands.append(f'new-tab --title "{name}" -p "Command Prompt" cmd /k {cmd}')
        
        if shutil.which("wt"):
            subprocess.run(f'wt {" ; ".join(commands)}', shell=True)
        else:
            print("Windows Terminal (wt) not found. Processes not started.")

    elif current_os == "Linux":
        procs = []
        for name, cfg in sorted_topo.items():
            kind = cfg.get("kind")
            if kind == "server":
                cmd = f'python "{get_abs_path("central_server.py")}" {cfg["host"]}:{cfg["port"]} --exp_id {EXP_ID} --min_edges {min_edges} --enable_dashboard {config.ENABLE_DASHBOARD}'
            elif kind == "edge":
                required_clients = edge_client_counts.get(name, 0)
                if required_clients == 0:
                    print(f"⚠️ Edge server {name} has 0 assigned clients. It is not invoked to prevent hangs.")
                else:
                    cmd = (
                        f'python "{get_abs_path("edge_server.py")}" --server '
                        f'{cfg["server"]["host"]}:{cfg["server"]["port"]} '
                        f'--client {cfg["client"]["host"]}:{cfg["client"]["port"]} --name {name} --exp_id {EXP_ID} '
                        f'--min_clients {required_clients}'
                    )
            elif kind == "client":
                cmd = (
                    f'python "{get_abs_path("client.py")}" '
                    f'{cfg["host"]}:{cfg["port"]} --partition_id {cfg["partition_id"]} '
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
