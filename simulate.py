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
    MIN_CLIENTS_PER_EDGE,
    DISSIMILAR_CLUSTERING
)
import config
import random
from utils import load_datasets, get_dataloader_summary, post_to_dashboard
import json
from clustering_utils import (
    parse_topology_for_clustering, 
    cluster_clients_by_distribution,
    assign_clusters_to_edge_servers,
    assign_dissimilar_clusters_to_edges,
    calculate_and_save_final_metrics
)
from collections import defaultdict
import torch
import psutil  # <--- Added for RAM monitoring

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


def print_system_resources():
    """Prints available CPU, GPU, and RAM resources."""
    print("\n" + "="*60)
    print("🖥️  SYSTEM RESOURCES CHECK")
    print("="*60)

    # 1. GPU Info
    try:
        gpu_count = torch.cuda.device_count()
        print(f"GPUs available: {gpu_count}")
        if gpu_count > 0:
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                # Check memory of each GPU
                try:
                    mem_info = torch.cuda.mem_get_info(i)
                    free_mem = mem_info[0] / (1024**3)
                    total_mem = mem_info[1] / (1024**3)
                    print(f"  [{i}] {gpu_name} | Mem: {free_mem:.2f}/{total_mem:.2f} GB Free")
                except:
                    print(f"  [{i}] {gpu_name}")
        else:
            print("  No GPUs detected. Training will run on CPU.")
    except Exception as e:
        print(f"  ⚠️ GPU Check Error: {e}")

    # 2. CPU Info
    try:
        total_cpus = os.cpu_count()
        pytorch_threads = torch.get_num_threads()
        print(f"\nCPUs:")
        print(f"  Total Logical Cores: {total_cpus}")
        print(f"  PyTorch Threads:     {pytorch_threads}")
    except Exception as e:
        print(f"  ⚠️ CPU Check Error: {e}")

    # 3. RAM Info
    try:
        mem = psutil.virtual_memory()
        total_ram = mem.total / (1024 ** 3)
        available_ram = mem.available / (1024 ** 3)
        percent_used = mem.percent
        
        print(f"\nSystem RAM:")
        print(f"  Total:     {total_ram:.2f} GB")
        print(f"  Available: {available_ram:.2f} GB")
        print(f"  Used:      {percent_used}%")
        
        if available_ram < 2.0:
            print("\n  ⚠️ WARNING: Low System RAM available (<2GB). Experiments might crash.")
            
    except Exception as e:
        print(f"  ⚠️ RAM Check Error (install psutil?): {e}")

    print("="*60 + "\n")


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

    partition_to_edge_target = {}

    cluster_assignments = cluster_results['cluster_assignments'] 
    cluster_to_edge = assign_clusters_to_edge_servers(topo_info, cluster_results)
    
    if DISSIMILAR_CLUSTERING:
        print("\n🔀 Mode: DISSIMILAR (Creating Diverse Edge Groups)")
        # Direct mapping: {pid -> edge_name}
        partition_to_edge_target = assign_dissimilar_clusters_to_edges(topo_info, cluster_results)
    else:
        print("\nBlob Mode: SIMILAR (Keeping Clusters Together)")
        # Old logic: {cid -> edge_name}
        cluster_assignments = cluster_results['cluster_assignments'] 
        cluster_to_edge = assign_clusters_to_edge_servers(topo_info, cluster_results)
        
        print("\n🔗 Re-routing Clients based on Data:")
        for pid, cid in cluster_assignments.items():
            target_edge = cluster_to_edge.get(cid)
            if target_edge:
                partition_to_edge_target[pid] = target_edge
    
    calculate_and_save_final_metrics(
        save_dir, 
        partition_to_edge_target, 
        cluster_results, 
        NUM_CLASSES
    )

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
    edge_groups = defaultdict(list)
    for client, details in client_assignments.items():
        edge_groups[details['target_edge']].append(f"{client}(P{details['pid']})")
    
    for edge, clients in edge_groups.items():
        client_str = ", ".join(clients)
        if len(client_str) > 80: client_str = client_str[:77] + "..."
        print(f"  📌 {edge}: {client_str}")
        
    print("="*80 + "\n")

def spawn_processes():
    # --- STEP 0: CHECK RESOURCES ---
    print_system_resources()

    topo_file = get_abs_path(f"topologies/{TOPOLOGY_FILE}")
    if not os.path.exists(topo_file):
        print(f"❌ Error: topo.yml not found at {topo_file}")
        return

    with open(topo_file, "r") as file:
        topology = yaml.safe_load(file)

    # --- STEP 1: RESOLVE PORTS ---
    edge_configs = {} 
    
    for name, cfg in topology.items():
        if cfg.get("kind") == "server":
            if not cfg.get("port"):
                auto_port = get_free_port()
                print(f"Assigning free port {auto_port} to server {name}")
                cfg["port"] = auto_port

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
            print(f"Edge {name} server -> {svr_host}:{svr_port}")

            cli = cfg.get("client", {})
            cli_host = cli.get("host")
            cli_port = cli.get("port") or get_free_port()
            cfg["client"]["host"] = cli_host
            cfg["client"]["port"] = cli_port
            print(f"Edge {name} client -> {cli_host}:{cli_port}")
            
            edge_configs[name] = {"host": cfg["client"]["host"], "port": cfg["client"]["port"]}

    # --- STEP 1.5: PRINT INITIAL TOPOLOGY ---
    initial_edge_counts = {edge: 0 for edge in edge_configs}
    initial_assignments = {}

    for name, cfg in topology.items():
        if cfg.get("kind") == "client":
            logical_pid = cfg.get("partition_id")
            final_edge = "Unknown"
            ref = cfg.get("host")
            
            if ref in topology and topology[ref].get("kind") == "edge":
                final_edge = ref
                edge_cli = topology[ref]["client"]
                cfg["host"] = edge_cli.get("host")
                cfg["port"] = edge_cli.get("port")
            
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
    edge_client_counts = {edge: 0 for edge in edge_configs}
    client_assignments = {}

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
    
    print("🚀 Spawning processes in 3 seconds...")
    time.sleep(3)

    # --- STEP 5: GPU BALANCING SETUP ---
    try:
        num_gpus = torch.cuda.device_count()
        print(f"🎮 Detected {num_gpus} GPUs available for load balancing.")
    except:
        num_gpus = 0
        print("⚠️  No GPUs detected. Running on CPU.")

    gpu_iterator = 0  # Round-robin counter

    # --- STEP 6: SPAWN ---
    current_os = platform.system()
    if config.ENABLE_DASHBOARD:
        create_experiment_on_dashboard(topology)
    
    order = {"server": 0, "edge": 1, "client": 2}
    sorted_topo = dict(sorted(topology.items(), key=lambda item: order.get(item[1].get("kind"), 99)))

    if current_os == "Windows":
        commands = []
        for name, cfg in sorted_topo.items():
            kind = cfg.get("kind")
            
            # --- GPU ASSIGNMENT (Windows) ---
            gpu_cmd_prefix = ""
            if num_gpus > 0:
                gpu_id = gpu_iterator % num_gpus
                gpu_iterator += 1
                gpu_cmd_prefix = f"set CUDA_VISIBLE_DEVICES={gpu_id} && "

            cmd = ""
            if kind == "server":
                cmd = f'py "{get_abs_path("monitor_process.py")}" --name {name} --kind server -- py "{get_abs_path("central_server.py")}" {cfg["host"]}:{cfg["port"]} --exp_id {EXP_ID} --min_edges {min_edges}'
            elif kind == "edge":
                # STRICT COUNT: defaults to 0 if not in list, preventing hangs on unused edges
                required_clients = edge_client_counts.get(name, 0)
                if required_clients == 0:
                    print(f"⚠️ Edge server {name} skipped (0 clients).")
                    continue
                cmd = (
                    f'py "{get_abs_path("monitor_process.py")}" --name {name} --kind edge -- '
                    f'py "{get_abs_path("edge_server.py")}" --server '
                    f'{cfg["server"]["host"]}:{cfg["server"]["port"]} --client '
                    f'{cfg["client"]["host"]}:{cfg["client"]["port"]} --name {name} --exp_id {EXP_ID} '
                    f'--min_clients {required_clients}'
                )
            elif kind == "client":
                cmd = (
                    f'py "{get_abs_path("monitor_process.py")}" --name {name} --kind client -- '
                    f'py "{get_abs_path("client.py")}" '
                    f'{cfg["host"]}:{cfg["port"]} --partition_id {cfg["partition_id"]} '
                    f"--name {name} --exp_id {EXP_ID}"
                )
            else:
                continue
            
            full_cmd = f"{gpu_cmd_prefix}{cmd}"
            commands.append(f'new-tab --title "{name}" -p "Command Prompt" cmd /k {full_cmd}')
        
        if shutil.which("wt"):
            subprocess.run(f'wt {" ; ".join(commands)}', shell=True)
        else:
            print("Windows Terminal (wt) not found.")

    elif current_os == "Linux":
        procs = []
        for name, cfg in sorted_topo.items():
            kind = cfg.get("kind")
            
            # --- GPU ASSIGNMENT (Linux) ---
            process_env = os.environ.copy()
            if num_gpus > 0:
                gpu_id = gpu_iterator % num_gpus
                gpu_iterator += 1
                process_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            cmd = ""
            if kind == "server":
                cmd = f'python "{get_abs_path("monitor_process.py")}" --name {name} --kind server -- python "{get_abs_path("central_server.py")}" {cfg["host"]}:{cfg["port"]} --exp_id {EXP_ID} --min_edges {min_edges}'
            elif kind == "edge":
                required_clients = edge_client_counts.get(name, 0)
                if required_clients == 0:
                    print(f"⚠️ Edge server {name} skipped (0 clients).")
                    continue
                cmd = (
                    f'python "{get_abs_path("monitor_process.py")}" --name {name} --kind edge -- '
                    f'python "{get_abs_path("edge_server.py")}" --server '
                    f'{cfg["server"]["host"]}:{cfg["server"]["port"]} '
                    f'--client {cfg["client"]["host"]}:{cfg["client"]["port"]} --name {name} --exp_id {EXP_ID} '
                    f'--min_clients {required_clients}'
                )
            elif kind == "client":
                cmd = (
                    f'python "{get_abs_path("monitor_process.py")}" --name {name} --kind client -- '
                    f'python "{get_abs_path("client.py")}" '
                    f'{cfg["host"]}:{cfg["port"]} --partition_id {cfg["partition_id"]} '
                    f" --name {name} --exp_id {EXP_ID}"
                )
            else:
                continue

            proc = subprocess.Popen(cmd, shell=True, env=process_env)
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
