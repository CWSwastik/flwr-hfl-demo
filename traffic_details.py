import os
import pandas as pd
import glob

# --- CONFIGURATION ---
ROOT_DIR = "logs/new_communication_new_mtgc_mnist"  # Change this to your logs folder path
OUTPUT_FILE = os.path.join(ROOT_DIR, "experiment_compression_details.csv")

def calculate_layer_stats(file_pattern):
    """
    Reads all CSVs matching the pattern and sums up Total and Compressed sizes.
    Returns: (total_mb, compressed_mb, file_count)
    """
    files = glob.glob(file_pattern)
    total_mb_agg = 0.0
    compressed_mb_agg = 0.0
    count = 0
    
    for f in files:
        try:
            df = pd.read_csv(f)
            # Ensure columns exist before summing
            if 'Total_MB' in df.columns and 'Compressed_Total_MB' in df.columns:
                total_mb_agg += df['Total_MB'].sum()
                compressed_mb_agg += df['Compressed_Total_MB'].sum()
                count += 1
        except Exception as e:
            print(f"Warning: Could not read {f}. Error: {e}")

    return total_mb_agg, compressed_mb_agg, count

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory '{ROOT_DIR}' not found.")
        return

    # Get list of experiment folders
    experiments = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    experiments.sort()
    
    results = []

    print(f"Processing {len(experiments)} experiments...")
    
    for exp_name in experiments:
        exp_path = os.path.join(ROOT_DIR, exp_name)
        
        # Define file patterns
        patterns = {
            "Central": os.path.join(exp_path, "central", "traffic.csv"),
            "Edge": os.path.join(exp_path, "edge", "Edge-*_traffic.csv"),
            "Client": os.path.join(exp_path, "clients", "Client-*_traffic.csv")
        }
        
        row = {"Experiment": exp_name}
        
        for layer, pattern in patterns.items():
            total, comp, count = calculate_layer_stats(pattern)
            
            # Calculate Ratio
            if comp > 0:
                ratio = total / comp
            elif count > 0:
                ratio = 1.0 # Default if compressed size is 0 but files exist
            else:
                ratio = None # No files found
            
            # Add columns for this layer
            prefix = layer
            row[f"{prefix}_Uncompressed_MB"] = round(total, 4)
            row[f"{prefix}_Compressed_MB"] = round(comp, 4)
            row[f"{prefix}_Ratio"] = round(ratio, 2) if ratio is not None else "N/A"

        # Add to results if at least one layer had data
        if any(row[k] != "N/A" for k in row if "Ratio" in k):
            results.append(row)

    # Create DataFrame
    if results:
        df_results = pd.DataFrame(results)
        
        # Reorder columns
        columns_order = ["Experiment"]
        for layer in ["Central", "Edge", "Client"]:
            columns_order.extend([
                f"{layer}_Uncompressed_MB", 
                f"{layer}_Compressed_MB", 
                f"{layer}_Ratio"
            ])
            
        df_results = df_results[columns_order]
        
        # Save to CSV
        df_results.to_csv(OUTPUT_FILE, index=False)
        
        # Display Summary Table in Console
        print("\n--- Experiment Compression Summary ---")
        print(df_results.to_string(index=False))
        print(f"\nSaved detailed results to: {OUTPUT_FILE}")
    else:
        print("No valid traffic data found.")

if __name__ == "__main__":
    main()