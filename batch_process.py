import os
import pandas as pd
import glob
import argparse
import re
import numpy as np

def parse_filename(filename):
    """Extracts node name from log filename."""
    basename = os.path.basename(filename)
    name_part = basename.split('_')[0].split('.')[0]
    return name_part

def extract_experiment_group(folder_name):
    """
    Normalizes experiment names for grouping.
    Example: 'mnist_lenet_iid_seed42' -> 'mnist_lenet_iid'
    Example: 'mnist_lenet_iid_run1'  -> 'mnist_lenet_iid'
    """
    # Remove _seedXXX or -seedXXX (case insensitive)
    name = re.sub(r'[_\W]seed\d+', '', folder_name, flags=re.IGNORECASE)
    # Remove _runX or -runX
    name = re.sub(r'[_\W]run\d+', '', name, flags=re.IGNORECASE)
    # Remove trailing/leading underscores or dashes
    return name.strip('_-')

def process_experiment_folder(folder_path):
    """
    Generates a summary for a SINGLE experiment folder.
    """
    # print(f"  Processing: {folder_path}...")
    
    traffic_data = []
    
    # 1. READ TRAFFIC
    traffic_files = glob.glob(os.path.join(folder_path, "**", "*traffic.csv"), recursive=True)
    total_volume_mb = 0.0
    total_uplink_mb = 0.0
    total_downlink_mb = 0.0
    
    for f in traffic_files:
        try:
            df = pd.read_csv(f)
            source = parse_filename(f)
            
            # Aggregate metrics
            if "Total_MB" in df.columns:
                total_volume_mb += df["Total_MB"].sum()
            
            if "Direction" in df.columns and "Total_MB" in df.columns:
                total_uplink_mb += df[df["Direction"] == "Uplink"]["Total_MB"].sum()
                total_downlink_mb += df[df["Direction"].str.contains("Downlink")]["Total_MB"].sum()
                
        except Exception:
            pass

    # 2. READ EVALUATION (Central Server Logs)
    central_log = glob.glob(os.path.join(folder_path, "**", "central_server.log"), recursive=True)
    final_accuracy = 0.0
    final_loss = 0.0
    max_accuracy = 0.0
    rounds_completed = 0
    
    if central_log:
        try:
            df = pd.read_csv(central_log[0])
            if not df.empty:
                rounds_completed = df["round"].max()
                # Get metrics from the last recorded round
                last_row = df.iloc[-1]
                final_accuracy = last_row.get("accuracy", 0.0)
                final_loss = last_row.get("loss", 0.0)
                max_accuracy = df["accuracy"].max()
        except Exception:
            pass

    return {
        "Final_Accuracy": final_accuracy,
        "Max_Accuracy": max_accuracy,
        "Final_Loss": final_loss,
        "Total_Traffic_MB": total_volume_mb,
        "Total_Uplink_MB": total_uplink_mb,
        "Total_Downlink_MB": total_downlink_mb,
        "Rounds": rounds_completed
    }

def main(root_dir, output_file):
    print(f"🚀 Scanning for experiments in: {os.path.abspath(root_dir)}")
    
    all_experiments = []
    
    # Walk through directory tree
    for root, dirs, files in os.walk(root_dir):
        # Heuristic: It's an experiment folder if it contains logs
        has_logs = any(f.endswith('traffic.csv') or f.endswith('.log') for f in files)
        
        if has_logs:
            exp_name = os.path.basename(root)
            
            # Identify Parent Run Folder (e.g., "run_1" from "logs/run_1/exp_name")
            parent_dir = os.path.basename(os.path.dirname(root))
            
            # If the parent is just "logs" or "compressed", label it "Root"
            run_id = parent_dir if "run" in parent_dir.lower() else "Default"

            # Process
            metrics = process_experiment_folder(root)
            
            # Add Identifiers
            metrics["Experiment_Folder"] = exp_name
            metrics["Run_ID"] = run_id
            metrics["Config_Group"] = extract_experiment_group(exp_name)
            
            all_experiments.append(metrics)
            print(f"  ✅ Found: {run_id} / {exp_name}")

    if not all_experiments:
        print("❌ No experiment logs found.")
        return

    # --- DATAFRAME CREATION ---
    df_raw = pd.DataFrame(all_experiments)
    
    # Sort for readability
    if "Run_ID" in df_raw.columns:
        df_raw.sort_values(by=["Config_Group", "Run_ID"], inplace=True)

    # --- AGGREGATION (STATISTICS) ---
    numeric_cols = ["Final_Accuracy", "Max_Accuracy", "Final_Loss", 
                    "Total_Traffic_MB", "Total_Uplink_MB", "Total_Downlink_MB", "Rounds"]
    
    # Group by the clean config name (averaging across runs/seeds)
    grouped = df_raw.groupby("Config_Group")[numeric_cols]
    
    df_mean = grouped.mean().reset_index()
    df_std = grouped.std().reset_index()
    df_min = grouped.min().reset_index()
    df_max = grouped.max().reset_index()
    df_count = grouped.count().reset_index()[["Config_Group", "Final_Accuracy"]].rename(columns={"Final_Accuracy": "Count"})

    # Combine Mean + Count for the main summary
    df_summary = pd.merge(df_mean, df_count, on="Config_Group")

    # --- SAVE TO EXCEL ---
    print(f"\n💾 Saving Global Statistics to {output_file}...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name="Average (Stats)", index=False)
        df_raw.to_excel(writer, sheet_name="All Raw Data", index=False)
        df_std.to_excel(writer, sheet_name="Std Dev", index=False)
        df_max.to_excel(writer, sheet_name="Maximums", index=False)
        df_min.to_excel(writer, sheet_name="Minimums", index=False)

    print("✨ Done! Open 'Average (Stats)' tab to compare experiments.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="logs/compressed", help="Root directory containing run folders")
    parser.add_argument("--out", type=str, default="global_statistics.xlsx", help="Output file name")
    args = parser.parse_args()
    
    if os.path.exists(args.dir):
        main(args.dir, args.out)
    else:
        print(f"❌ Directory not found: {args.dir}")