import os
import pandas as pd
import glob
import argparse

def process_experiment(exp_path):
    """
    Aggregates traffic logs for a single experiment folder.
    Generates 'traffic_summary.xlsx' with detailed separations.
    """
    print(f"📦 Processing Experiment: {exp_path}")
    
    # Subfolders to scan
    subfolders = ["clients", "edge", "central"]
    
    all_dfs = []
    
    for sub in subfolders:
        # Pattern: exp_path/subfolder/*traffic.csv
        search_path = os.path.join(exp_path, sub, "*traffic.csv")
        files = glob.glob(search_path)
        
        for f in files:
            try:
                df = pd.read_csv(f)
                
                # Add context info
                filename = os.path.basename(f)
                node_name = filename.split('_')[0]  # e.g., "Client-1"
                
                df.insert(0, "Role", sub.capitalize())  # Client, Edge, Central
                df.insert(1, "Node", node_name)
                
                all_dfs.append(df)
            except Exception as e:
                print(f"  ⚠️ Error reading {f}: {e}")

    if not all_dfs:
        print(f"  ❌ No traffic logs found in {exp_path}")
        return

    # 1. Combine all raw data
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort for readability
    if "Round" in full_df.columns:
        full_df.sort_values(by=["Round", "Role", "Node"], inplace=True)

    # --- 2. AGGREGATION LOGIC (The "Summary by Node" tab) ---
    
    # Define aggregation rules (Summing up the specific columns you asked for)
    agg_rules = {
        "Round": "nunique",                  # Count active rounds
        
        # Traffic Volumes (Uncompressed vs Compressed)
        "model_wts_MB": "sum",
        "compressed_model_wts_MB": "sum",
        "Y_i_MB": "sum",
        "compressed_Y_i_MB": "sum",
        "Z_i_MB": "sum",
        "compressed_Z_i_MB": "sum",
        "Total_MB": "sum",                   # Total Data Volume
        
        # Time Metrics (Separated)
        "compression_time_s": "sum",
        "decompression_time_s": "sum"
    }

    # Only aggregate columns that actually exist in the CSVs
    # (This prevents errors if old logs are missing some columns)
    actual_agg_rules = {k: v for k, v in agg_rules.items() if k in full_df.columns}

    # Group by Node and calculate sums
    summary_df = full_df.groupby(["Role", "Node"]).agg(actual_agg_rules).reset_index()

    # --- 3. RENAME COLUMNS FOR CLARITY ---
    rename_map = {
        "Round": "Active Rounds",
        
        # Model
        "model_wts_MB": "Model Wts (Uncompressed MB)",
        "compressed_model_wts_MB": "Model Wts (Compressed MB)",
        
        # Yi (Global Control Variate)
        "Y_i_MB": "Yi (Uncompressed MB)",
        "compressed_Y_i_MB": "Yi (Compressed MB)",
        
        # Zi (Local Control Variate)
        "Z_i_MB": "Zi (Uncompressed MB)",
        "compressed_Z_i_MB": "Zi (Compressed MB)",
        
        # Totals
        "Total_MB": "Total Data Volume (MB)",
        
        # Timings
        "compression_time_s": "Total Compression Time (s)",
        "decompression_time_s": "Total Decompression Time (s)"
    }
    summary_df.rename(columns=rename_map, inplace=True)

    # --- 4. EXPORT TO EXCEL ---
    output_path = os.path.join(exp_path, "traffic_summary.xlsx")
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Tab 1: High-level Summary (The one you requested)
            summary_df.to_excel(writer, sheet_name="Summary by Node", index=False)
            
            # Tab 2: All Raw Data (For deep debugging)
            full_df.to_excel(writer, sheet_name="All Raw Traffic", index=False)
            
        print(f"  ✅ Generated Report: {output_path}")
    except Exception as e:
        print(f"  ❌ Failed to save Excel: {e}")

def main(root_dir):
    print(f"🚀 Scanning for experiments in: {root_dir}")
    
    experiment_paths = set()
    
    # Smart detection of experiment folders
    for root, dirs, files in os.walk(root_dir):
        # We look for folders containing the standard logger subdirectories
        if set(dirs).intersection({"clients", "edge", "central"}):
            experiment_paths.add(root)

    if not experiment_paths:
        print("  ⚠️ No experiment logs found. (Looking for 'clients', 'edge', 'central' folders)")
        return

    # Process each experiment found
    for exp_path in sorted(list(experiment_paths)):
        process_experiment(exp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="logs", help="Root logs directory (e.g., logs or logs/run_1)")
    args = parser.parse_args()
    
    if os.path.exists(args.dir):
        main(args.dir)
    else:
        print(f"❌ Directory not found: {args.dir}")