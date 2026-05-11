# Modified sections for simulate.py
# Add these imports at the top of simulate.py

"""
Add to imports section:
from clustering_utils import (
    parse_topology_for_clustering,
    cluster_clients_by_distribution,
    assign_clusters_to_edge_servers
)
"""

# Replace the precompute_partition_mapping() function with this improved version:

def precompute_partition_mapping_v2(topology_file):
    """
    Enhanced version that uses the new clustering utilities.
    Clusters clients based on label distribution and assigns them to edge servers.
    
    Args:
        topology_file: Path to topology YAML file
    
    Returns:
        dict: Mapping from logical client ID to physical partition ID
    """
    if CLUSTER_STRATEGY == "none":
        print("✅ No clustering strategy selected. Using default 1-to-1 partition mapping.")
        return {i: i for i in range(NUM_CLIENTS)}
    
    print(f"\n{'='*70}")
    print(f"🔬 Starting Partition Clustering with {CLUSTER_STRATEGY.upper()} Distance")
    print(f"{'='*70}")
    
    # Step 1: Parse topology to understand the hierarchical structure
    topo_file_path = get_abs_path(f"topologies/{topology_file}")
    topology_info = parse_topology_for_clustering(topo_file_path)
    
    num_edge_servers = topology_info['num_edge_servers']
    
    if num_edge_servers == 0:
        print("⚠️  No edge servers found in topology. Using default mapping.")
        return {i: i for i in range(NUM_CLIENTS)}
    
    # Step 2: Cluster clients based on label distribution
    cluster_result = cluster_clients_by_distribution(
        num_clusters=num_edge_servers,
        distance_metric=CLUSTER_STRATEGY,
        save_dir=os.path.join(BASE_DIR, "logs", config.EXPERIMENT_NAME)
    )
    
    partition_map = cluster_result['partition_mapping']
    
    # Step 3: Assign clusters to edge servers
    cluster_to_edge = assign_clusters_to_edge_servers(topology_info, cluster_result)
    
    # Step 4: Display summary statistics
    print(f"\n{'='*70}")
    print("📊 Clustering Summary")
    print(f"{'='*70}")
    print(f"Strategy: {CLUSTER_STRATEGY.upper()}")
    print(f"Total Clients: {NUM_CLIENTS}")
    print(f"Edge Servers: {num_edge_servers}")
    print(f"Clusters Created: {len(set(cluster_result['cluster_assignments'].values()))}")
    
    # Show distribution similarity within clusters
    print(f"\n📈 Cluster Statistics:")
    for cluster_id in sorted(set(cluster_result['cluster_assignments'].values())):
        clients_in_cluster = [k for k, v in cluster_result['cluster_assignments'].items() if v == cluster_id]
        print(f"  Cluster {cluster_id}: {len(clients_in_cluster)} clients")
    
    print(f"\n{'='*70}")
    print("✅ Partition Clustering Complete")
    print(f"{'='*70}\n")
    
    return partition_map


# Alternative: If you want to keep your existing function but enhance it,
# replace the save_distribution_mapping() function with this version:

def save_distribution_mapping_v2(topology, partition_map):
    """
    Enhanced version that saves more detailed clustering information.
    Saves three CSV files using the clustering utilities.
    
    Args:
        topology: Topology dictionary from YAML
        partition_map: Mapping from logical to physical partition IDs
    """
    print("📊 Saving distribution mappings and cluster information...")
    
    # Parse topology
    topo_file_path = get_abs_path(f"topologies/{TOPOLOGY_FILE}")
    topology_info = parse_topology_for_clustering(topo_file_path)
    
    # Load partition data
    print(f"Loading partition data for {NUM_CLIENTS} clients...")
    partition_data = []
    for pid in range(NUM_CLIENTS):
        trainloader, _, _ = load_datasets(partition_id=pid)
        summary = get_dataloader_summary(trainloader)
        dist_counts_map = summary["label_distribution"]
        num_items = summary["num_items"]
        
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
    
    partition_lookup = {p['pid']: p['counts'] for p in partition_data}
    
    # Prepare data structures
    pre_cluster_data = []
    post_cluster_data = []
    edge_aggregation = {}
    
    # Iterate through topology
    sorted_items = sorted(topology.items(), key=lambda x: x[0])
    for name, cfg in sorted_items:
        if cfg.get("kind") == "client":
            logical_id = cfg["partition_id"]
            
            # Pre-clustering (1-to-1)
            pre_counts = partition_lookup.get(logical_id, np.zeros(NUM_CLASSES))
            pre_row = {
                "ClientName": name,
                "LogicalID": logical_id,
                "TotalSamples": int(np.sum(pre_counts))
            }
            for i, count in enumerate(pre_counts):
                pre_row[f"Class_{i}"] = int(count)
            pre_cluster_data.append(pre_row)
            
            # Post-clustering (mapped)
            physical_id = partition_map.get(logical_id, logical_id)
            post_counts = partition_lookup.get(physical_id, np.zeros(NUM_CLASSES))
            
            # Identify edge server
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
            
            # Edge aggregation
            if edge_server not in edge_aggregation:
                edge_aggregation[edge_server] = np.zeros(NUM_CLASSES)
            edge_aggregation[edge_server] += post_counts
    
    # Save files
    log_dir = os.path.join(BASE_DIR, "logs", config.EXPERIMENT_NAME)
    os.makedirs(log_dir, exist_ok=True)
    
    df_pre = pd.DataFrame(pre_cluster_data)
    df_pre.to_csv(os.path.join(log_dir, "distribution_pre_clustering.csv"), index=False)
    
    df_post = pd.DataFrame(post_cluster_data)
    df_post.to_csv(os.path.join(log_dir, "distribution_post_clustering.csv"), index=False)
    
    edge_rows = []
    for edge, counts in edge_aggregation.items():
        row = {"EdgeServer": edge, "TotalSamples": int(np.sum(counts))}
        for i, count in enumerate(counts):
            row[f"Class_{i}"] = int(count)
        edge_rows.append(row)
    
    df_edge = pd.DataFrame(edge_rows)
    df_edge.to_csv(os.path.join(log_dir, "distribution_edge_clusters.csv"), index=False)
    
    print(f"✅ Saved 3 distribution CSVs to {log_dir}")


# Integration guide for spawn_processes() function:
"""
In your spawn_processes() function, replace the partition clustering section with:

    # This map dictates which physical partition each logical client gets
    try:
        # Use the new version
        partition_map = precompute_partition_mapping_v2(TOPOLOGY_FILE)
        print("✅ Partition pre-clustering completed.\\nSaving distributions")
        save_distribution_mapping_v2(topology, partition_map)
    except Exception as e:
        print(f"❌ Error during partition pre-clustering: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to default 1-to-1 mapping.")
        partition_map = {i: i for i in range(NUM_CLIENTS)}
"""
