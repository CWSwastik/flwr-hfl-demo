# clustering_utils.py
# Utility functions for clustering clients in Hierarchical FL based on label distribution

import yaml
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from collections import defaultdict
import pandas as pd
import os
from config import NUM_CLIENTS, NUM_CLASSES, EXPERIMENT_NAME, TOPOLOGY_FILE
from utils import load_datasets, get_dataloader_summary
import pprint
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

rand_seed = 42
np.random.seed(seed=rand_seed)

def parse_topology_for_clustering(topology_file_path):
    """
    Reads the topology YAML file and extracts:
    1. Edge server to clients mapping
    2. Number of edge servers
    3. Client information with their partition IDs
    
    Args:
        topology_file_path: Path to the topology YAML file
    
    Returns:
        dict: {
            'num_edge_servers': int,
            'edge_servers': [edge_server_names],
            'edge_servers_configs': {edge_server: config},
            'edge_to_clients': {edge_name: [client_names]},
            'client_info': {client_name: {'partition_id': int, 'edge_server': str}},
            'central_server': str,
            'pid_to_clientname': {partition_id: client_name}
        }
    """
    with open(topology_file_path, 'r') as f:
        topology = yaml.safe_load(f)
    
    edge_to_clients = defaultdict(list)
    client_info = {}
    central_server = None
    edge_servers_names = []
    edge_servers_configs = {}
    pid_to_clientname = {}
    
    # First pass: identify central server and edge servers
    for name, cfg in topology.items():
        if cfg.get('kind') == 'server':
            central_server = name
        elif cfg.get('kind') == 'edge':
            edge_servers_names.append(name)
            edge_servers_configs[name] = cfg

    
    # Second pass: map clients to edge servers
    for name, cfg in topology.items():
        if cfg.get('kind') == 'client':
            partition_id = cfg.get('partition_id')
            host_ref = cfg.get('host')
            host_port = cfg.get('port')
            
            # find the edge server this client connects to
            for edge_server in edge_servers_names:
                if edge_servers_configs[edge_server].get('client')['host'] == host_ref and \
                   edge_servers_configs[edge_server].get('client')['port'] == host_port:
                    edge_to_clients[edge_server].append(name)
                    break

            client_info[name] = {
                'partition_id': partition_id,
                'edge_server': edge_server,
                'host_ref': host_ref,
                'host_port': host_port
            }
            if partition_id is not None:
                pid_to_clientname[int(partition_id)] = name
    
    num_edge_servers = len(edge_servers_names)
    
    result = {
        'num_edge_servers': num_edge_servers,
        'edge_servers': edge_servers_names,
        'edge_servers_configs': edge_servers_configs,
        'edge_to_clients': dict(edge_to_clients),
        'client_info': client_info,
        'central_server': central_server,
        'pid_to_clientname': pid_to_clientname
    }
    
    print(f"\n📊 Topology Analysis:")
    print(f"  Central Server: {central_server}")
    print(f"  Number of Edge Servers: {num_edge_servers}")
    print(f"  Edge Servers: {edge_servers_names}")
    for edge, clients in edge_to_clients.items():
        print(f"    {edge}: {len(clients)} clients")
    
    return result


def cluster_clients_by_distribution(num_clusters, distance_metric='emd', save_dir=None, topology_info=None):
    """
    Clusters clients based on their label distribution using EMD or JSD.
    Updates client partition IDs based on clustering results so that clients
    within the same cluster have similar label distributions.
    
    Args:
        num_clusters: Number of clusters (typically number of edge servers)
        distance_metric: 'emd' for Earth Mover's Distance or 'jsd' for Jensen-Shannon Divergence
        save_dir: Directory to save CSV files (if None, uses logs/{EXPERIMENT_NAME})
        topology_info: {'num_edge_servers', 'edge_to_clients', 'client_info', 'central_server', 'pid_to_clientname'}
    
    Returns:
        dict: {
            'partition_mapping': {logical_id: assigned_id},
            'cluster_assignments': {logical_id: cluster_id},
            'client_distributions_pre': DataFrame,
            'client_distributions_post': DataFrame,
            'cluster_distributions': DataFrame,
            'distance_matrix': ndarray,
            'linkage_matrix': ndarray
        }
    """
    print(f"\n🔄 Starting client clustering with {num_clusters} clusters using {distance_metric.upper()}...")
    
    n_clients = NUM_CLIENTS
    n_classes = NUM_CLASSES

    # Helper to map Edge Name -> ID for "DefaultClusterID"
    edge_server_list = topology_info.get('edge_servers', []) if topology_info else []
    edge_name_to_id = {name: i+1 for i, name in enumerate(edge_server_list)} # 1-based index to match cluster IDs often used

    # Step 1: Load all client data distributions
    print(f"📦 Loading distributions for {n_clients} clients...")
    partition_data = []
    pid_to_clientname = {}
    if topology_info is not None:
        pid_to_clientname = topology_info.get('pid_to_clientname', {})
    
    for pid in range(n_clients):
        trainloader, _, _ = load_datasets(partition_id=pid)
        summary = get_dataloader_summary(trainloader)
        dist_counts_map = summary['label_distribution']
        num_items = summary['num_items']
        
        # Create counts vector
        counts_vector = np.zeros(n_classes)
        if num_items > 0:
            for label_str, count in dist_counts_map.items():
                label_int = int(label_str)
                if 0 <= label_int < n_classes:
                    counts_vector[label_int] = count
        
        # Store both counts and normalized distribution
        distribution = counts_vector / num_items if num_items > 0 else counts_vector
        
        # Identify Default Cluster ID (Initial Topology)
        default_cluster_id = 0
        client_name = pid_to_clientname.get(pid)
        if client_name and topology_info:
            c_info = topology_info['client_info'].get(client_name)
            if c_info:
                edge_name = c_info.get('edge_server')
                if edge_name:
                    default_cluster_id = edge_name_to_id.get(edge_name, 0)
        
        partition_data.append({
            'pid': pid,
            'default_cluster_id': default_cluster_id,
            'counts': counts_vector,
            'distribution': distribution,
            'total': num_items
        })
    # --- CASE: NO CLUSTERING (Static) ---
    if distance_metric == 'none' or distance_metric is None:
        print("  Strategy is None: Using static topology mappings.")
        print(f"edge_to_clients: {topology_info['edge_to_clients']}")
        pprint.pprint(f"client_info: {topology_info['client_info']}")
        partition_mapping = {}
        for clients, config in topology_info['client_info'].items():
            pid = config['partition_id']
            partition_mapping[pid] = pid  # logical ID maps to same physical partition ID
        cluster_assignments = {}
        
        # Map original edge servers to fake ClusterIDs (1, 2, 3...)
        edge_servers = topology_info['edge_servers']
        print(f"Edge Servers: {edge_servers}")
        edge_to_cluster_id = {edge: i+1 for i, edge in enumerate(edge_servers)}
        
        # Assign clients to clusters based on their static edge connection
        for pid in range(n_clients):
            client_name = pid_to_clientname.get(pid)
            if client_name:
                edge = topology_info['client_info'][client_name]['edge_server']
                # If connected to central or unknown, assign Cluster 0
                cluster_assignments[pid] = edge_to_cluster_id.get(edge, 0) 
            else:
                cluster_assignments[pid] = 0

        print(f"cluster_assignments: {cluster_assignments}")
        # Dummy matrices for return values
        dist_matrix = np.zeros((n_clients, n_clients))
        linkage_matrix = np.zeros((n_clients, 4))
        cluster_labels = list(cluster_assignments.values())
        if (pid + 1) % 10 == 0 or pid == n_clients - 1:
            print(f"  Loaded {pid + 1}/{n_clients} partitions...")
    
    # --- CASE: DYNAMIC CLUSTERING ---
    else:
        # Step 2: Calculate pairwise distance matrix
        print(f"\n📏 Calculating pairwise distances using {distance_metric.upper()}...")
        
        dist_matrix = np.zeros((n_clients, n_clients))
        linkage_matrix = np.zeros((n_clients, 4))
        X = np.array([pd['distribution'] for pd in partition_data])  # (n_clients, n_classes)
        
        if distance_metric == 'emd':
            # EMD using Wasserstein distance (1D Earth Mover's Distance)
            class_indices = np.arange(n_classes)
            for i in range(n_clients):
                for j in range(i + 1, n_clients):
                    dist = wasserstein_distance(
                        class_indices, class_indices,
                        partition_data[i]['distribution'],
                        partition_data[j]['distribution']
                    )
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
        
        elif distance_metric == 'jsd':
            # Jensen-Shannon Divergence
            for i in range(n_clients):
                for j in range(i + 1, n_clients):
                    # Add small epsilon to avoid log(0)
                    p = partition_data[i]['distribution'] + 1e-10
                    q = partition_data[j]['distribution'] + 1e-10
                    dist = jensenshannon(p, q)
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
        
        elif distance_metric == 'cosine':
            for i in range(n_clients):
                for j in range(i + 1, n_clients):
                    dist = cosine(partition_data[i]['distribution'], 
                                 partition_data[j]['distribution'])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
        
        elif distance_metric == 'euclidean':
            for i in range(n_clients):
                for j in range(i + 1, n_clients):
                    dist = euclidean(partition_data[i]['distribution'], 
                                    partition_data[j]['distribution'])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
        
        elif distance_metric == 'manhattan':
            for i in range(n_clients):
                for j in range(i + 1, n_clients):
                    dist = cityblock(partition_data[i]['distribution'], 
                                    partition_data[j]['distribution'])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
        
        elif distance_metric == 'gmm':
            gmm = GaussianMixture(n_components=num_clusters, random_state=rand_seed, n_init=10)
            cluster_labels = gmm.fit_predict(X)
            print(f" GMM log-likelihood: {gmm.score(X):.4f}")
        
        elif distance_metric == 'kmeans':
            kmeans = KMeans(n_clusters=num_clusters, random_state=rand_seed, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            print(f" K-means inertia: {kmeans.inertia_:.4f}")
        
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        print(f"  Distance matrix shape: {dist_matrix.shape}")
        if distance_metric not in ['gmm', 'kmeans']:
            print(f" Distance range: [{dist_matrix[dist_matrix > 0].min():.4f}, {dist_matrix.max():.4f}]")
    
    # Step 3: Perform clustering
    print(f"\n🌳 Performing clustering...")
    if distance_metric in ['gmm', 'kmeans']:
            # Use pre-computed labels
            pass
    else:
        condensed_dist_matrix = dist_matrix[np.triu_indices(n_clients, k=1)]
        if len(condensed_dist_matrix) > 0 and distance_metric != 'none':
            linkage_matrix = linkage(condensed_dist_matrix, method='average')
            cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        elif distance_metric == 'none' or distance_metric is None:
            cluster_labels = cluster_assignments.values()
        else:
            cluster_labels = [1] * n_clients
    print(f"  Cluster labels assigned: {cluster_labels}")
    clusters = defaultdict(list)
    for pid, cluster_id in enumerate(cluster_labels):
        clusters[cluster_id].append(pid)
    
    partition_mapping = {}
    cluster_assignments = {}
    new_pid = 0
    
    for cluster_id in sorted(clusters.keys()):
        client_pids = sorted(clusters[cluster_id])
        for original_pid in client_pids:
            partition_mapping[new_pid] = original_pid
            cluster_assignments[new_pid] = cluster_id
            new_pid += 1

    # --- GENERATE REPORTS (Shared logic) ---
    print("  Generating distribution CSVs...")
    pre_cluster_rows = []
    for pid in range(n_clients):
        client_name = pid_to_clientname.get(pid, f"Client-{pid}")
        row = {
            'ClientName': client_name,
            'PartitionID': pid,
            'DefaultClusterID': partition_data[pid]['default_cluster_id'],
            'TotalSamples': int(partition_data[pid]['total']),
        }
        for class_idx in range(n_classes):
            row[f'Class_{class_idx}'] = int(partition_data[pid]['counts'][class_idx])
        pre_cluster_rows.append(row)
    
    df_pre = pd.DataFrame(pre_cluster_rows)
    
    # Post-clustering: new assignments based on clustering
    post_cluster_rows = []
    for logical_id in range(n_clients):
        assigned_id = partition_mapping.get(logical_id, logical_id)
        # If dynamic, logical ID implies position in sorted list, so cluster ID matches
        # If static, logical ID is just ID, lookup cluster assignment
        cluster_id = cluster_assignments.get(logical_id, 0)
        
        client_name = pid_to_clientname.get(logical_id, f"Client-{logical_id}")
        physical_client_name = pid_to_clientname.get(assigned_id, f"Client-{assigned_id}")

        row = {
            'ClientName': client_name,          # from topology (e.g., Client-1)
            'LogicalPartitionID': logical_id,
            'AssignedPartitionID': assigned_id,
            'ClusterID': cluster_id,
            'TotalSamples': int(partition_data[assigned_id]['total']),
        }
        num_classes = NUM_CLASSES
        for class_idx in range(num_classes):
            row[f'Class_{class_idx}'] = int(partition_data[assigned_id]['counts'][class_idx])
        post_cluster_rows.append(row)
    
    df_post = pd.DataFrame(post_cluster_rows)
    
    # Cluster-level aggregated distribution
    unique_clusters = sorted(list(set(cluster_assignments.values())))
    cluster_aggregate_rows = []
    for c_id in unique_clusters:
        cluster_counts = np.zeros(n_classes)
        total_samples = 0
        member_clients = []
        
        for log_id in range(n_clients):
            if cluster_assignments.get(log_id) == c_id:
                phys_id = partition_mapping.get(log_id, log_id)
                cluster_counts += partition_data[phys_id]['counts']
                total_samples += partition_data[phys_id]['total']
                member_clients.append(log_id)
        
        row = {
            'ClusterID': c_id,
            'NumClients': len(member_clients),
            'ClientIDs': str(member_clients),
            'TotalSamples': int(total_samples)
        }
        for class_idx in range(num_classes):
            row[f'Class_{class_idx}'] = int(cluster_counts[class_idx])
        cluster_aggregate_rows.append(row)
    
    df_cluster = pd.DataFrame(cluster_aggregate_rows)
    
    # Step 7: Save CSVs
    save_distributions_clusters(save_dir, df_pre, df_post, df_cluster)
    
    return {
        'partition_mapping': partition_mapping,
        'cluster_assignments': cluster_assignments,
        'client_distributions_pre': df_pre,
        'client_distributions_post': df_post,
        'cluster_distributions': df_cluster,
        'distance_matrix': dist_matrix,
        'linkage_matrix': linkage_matrix
    }

def save_distributions_clusters(save_dir, df_pre, df_post, df_cluster):
    if save_dir is None:
        save_dir = os.path.join('logs', EXPERIMENT_NAME)
    
    os.makedirs(save_dir, exist_ok=True)
    
    df_pre.to_csv(os.path.join(save_dir, 'distribution_pre_clustering.csv'), index=False)
    df_post.to_csv(os.path.join(save_dir, 'distribution_post_clustering.csv'), index=False)
    df_cluster.to_csv(os.path.join(save_dir, 'cluster_distribution.csv'), index=False)
    
    print(f"\n✅ Saved distribution CSVs:")
    print(f"✅- {os.path.join(save_dir, 'distribution_pre_clustering.csv')}")
    print(f"✅- {os.path.join(save_dir, 'distribution_post_clustering.csv')}")
    print(f"✅- {os.path.join(save_dir, 'cluster_distribution.csv')}")


def assign_clusters_to_edge_servers(topology_info, cluster_result):
    """
    Maps clusters to edge servers based on topology.
    
    Args:
        topology_info: Result from parse_topology_for_clustering()
        cluster_result: Result from cluster_clients_by_distribution()
    
    Returns:
        dict: Mapping of edge servers to cluster IDs
    """
    num_edge_servers = topology_info['num_edge_servers']
    num_clusters = len(set(cluster_result['cluster_assignments'].values()))
    
    if num_clusters != num_edge_servers:
        print(f"⚠️  Warning: Number of clusters ({num_clusters}) != number of edge servers ({num_edge_servers})")
    
    edge_servers = topology_info['edge_servers']
    cluster_ids = sorted(list(set(cluster_result['cluster_assignments'].values())))
    
    # Remove 0 if it exists (unassigned)
    if 0 in cluster_ids: cluster_ids.remove(0)	
    cluster_to_edge = {}

    # Simple mapping: Cluster 1 -> Edge 0, Cluster 2 -> Edge 1
    for i, c_id in enumerate(cluster_ids):
        if i < len(edge_servers):
            cluster_to_edge[c_id] = edge_servers[i]
    return cluster_to_edge


# Example usage and integration test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Clustering Utilities for Hierarchical FL")
    print("=" * 60)
    
    # Test 1: Parse topology
    topology_file = os.path.join("topologies", TOPOLOGY_FILE)
    if os.path.exists(topology_file):
        print("\nTest 1: Parsing topology...")
        topology_info = parse_topology_for_clustering(topology_file)
        print(f"Number of edge servers: {topology_info['num_edge_servers']}")
        
        # Test 2: Cluster clients
        print("\nTest 2: Clustering clients by distribution...")
        cluster_result = cluster_clients_by_distribution(
            num_clusters=topology_info['num_edge_servers'],
            distance_metric='none',  # or 'emd' or 'none'
            # save_dir=os.path.join(BASE_DIR, "logs", config.EXPERIMENT_NAME),
            topology_info=topology_info,
        )
        
        print("\nPartition mapping (first 5):")
        for k, v in list(cluster_result['partition_mapping'].items())[:5]:
            print(f"  Logical ID {k} -> Physical Partition {v}")
        
        # Test 3: Assign clusters to edge servers
        print("\nTest 3: Assigning clusters to edge servers...")
        cluster_to_edge = assign_clusters_to_edge_servers(topology_info, cluster_result)
    else:
        print(f"⚠️  Topology file not found: {topology_file}")
        print("Please ensure the topology file exists in the correct location.")
