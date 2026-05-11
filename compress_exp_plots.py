import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
ROOT_DIR = "./logs/new_compression"  # Directory containing experiment subfolders
OUTPUT_FILE = "compression_comparison.png"
Y_LIMITS = (0, 100)  # Adjust accuracy limits if needed

# --- COLORS FROM PROVIDED FILE ---
# Index 0 (Red) is reserved for Baseline (No Compression)
COLORS = [
    [1.0, 0.0, 0.0, 1.0],  # Red 0 (Baseline)
    [0.6627451, 0.8505867, 0.53165947, 1.0],  # Green 1
    [0.07254902, 0.88292761, 0.9005867, 1.0],  # Light Blue 2
    [0.0, 0.0, 1.0, 1.0],  # Blue 3
    [1.0, 0.49803922, 0.05490196, 1.0],  # Orange 4
    [0.58039216, 0.40392157, 0.74117647, 1.0],  # Purple 5
    [0.54901961, 0.3372549, 0.29411765, 1.0],  # Brown 6
    [0.89019608, 0.46666667, 0.76078431, 1.0],  # Pink 7
    [0.49803922, 0.49803922, 0.49803922, 1.0],  # Gray 8
    [0.7372549, 0.74117647, 0.13333333, 1.0],  # Yellow 9
    [0.99215686, 0.70588235, 0.38431373, 1.0],  # Gold 10
    [0.15294118, 0.16078431, 0.16078431, 1.0],  # Black 11
    [0.97254902, 0.70588235, 0.09411765, 1.0],  # Apricot 12
    [0.85098039, 0.37254902, 0.00784314, 1.0],  # Cinnamon 13
    [0.90196078, 0.90196078, 0.98039216, 1.0],  # Lavender 14
    [0.90588235, 0.16078431, 0.54117647, 1.0],  # Magenta 15
    [0.4, 0.65098039, 0.11764706, 1.0],  # Lime 16
    [0.65098039, 0.4627451, 0.11372549, 1.0],  # Sienna 17
    [0.71764706, 0.81960784, 0.54607843, 1.0],  # Lime 18
    [0.83137255, 0.68627451, 0.21568627, 1.0],  # Sienna 19
    [0.0, 0.4, 0.4, 1.0],  # Dark Teal 20
    [0.6, 0.0, 1.0, 1.0],  # Electric Purple 21
    [0.0, 0.74, 0.67, 1.0],  # Turquoise Blue 22
    [0.15, 0.2, 0.05, 1.0],  # Dark Olive 23
]

def load_experiment_data(folder_path):
    """
    Reads accuracy data. Tries 'global_acc_loss.csv' first, then 'central_server.log'.
    Returns: (rounds, accuracy) or (None, None) if failed.
    """
    # 1. Try parsed CSV (from your post-processing script)
    csv_path = os.path.join(folder_path, 'global_acc_loss.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'acc' in df.columns and 'round_numbers' in df.columns:
                return df['round_numbers'].values, df['acc'].values
        except:
            pass

    # 2. Try raw log (from central_server.py)
    log_path = os.path.join(folder_path, 'central', 'central_server.log')
    if os.path.exists(log_path):
        try:
            # Assumes CSV format: round,loss,accuracy (or similar with headers)
            # If header exists, pd.read_csv handles it. 
            # If your log has no header, you might need names=['round','loss','accuracy']
            with open(log_path, 'r') as f:
                header = f.readline()
                has_header = 'accuracy' in header.lower() or 'round' in header.lower()
            
            if has_header:
                df = pd.read_csv(log_path)
            else:
                # Fallback for standard Logger format
                df = pd.read_csv(log_path, names=['round', 'loss', 'accuracy'])
            
            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            
            if 'accuracy' in df.columns and 'round' in df.columns:
                return df['round'].values, df['accuracy'].values * 100 # usually 0-1 in log, 0-100 in plot
            elif 'acc' in df.columns:
                 return df['round'].values, df['acc'].values
                 
        except Exception as e:
            print(f"Failed to read log {log_path}: {e}")
            pass
            
    return None, None

def parse_experiment_info(folder_name):
    """
    Parses folder name to determine:
    1. Compression Method (e.g., 'quant_8bit', 'topk_0.1')
    2. Target (Yi, Zi, YiZi, or None)
    """
    # Check for target keywords based on config.py logic
    target = "None" # Default to None (Baseline)
    
    # Check for specific compression flags
    if "_YiZi_" in folder_name:
        target = "YiZi"
    elif "_Yi_" in folder_name:
        target = "Yi"
    elif "_Zi_" in folder_name:
        target = "Zi"
    
    # Extract Compression Method Name
    # We assume naming format: ...-compress_[METHOD]_[TARGET]...
    # e.g., "mnist-lenet-compress_quant_8bit_Yi_Zi_..."
    
    if "compress_" in folder_name:
        # Regex to capture text between 'compress_' and the next target/end
        # Matches "quant_8bit" from "...compress_quant_8bit_Yi..."
        match = re.search(r"compress_([a-zA-Z0-9_]+)_(?:Yi|Zi|YiZi)", folder_name)
        if match:
            method_name = match.group(1)
        else:
            # Fallback for simple names
            method_name = "Compressed"
    else:
        method_name = "No Compression"
        target = "None"

    # Clean up formatting for display
    method_display = method_name.replace("_", " ").title()
    if "Quant" in method_display: method_display = method_display.replace("Quant", "Quantization")
    if "Pct" in method_display: method_display = method_display.replace("Pct", "%")
    
    return method_display, target

def get_color(method_name, method_color_map):
    """
    Returns consistent color for a compression method.
    "No Compression" is ALWAYS Red (Index 0).
    New methods get assigned the next available color.
    """
    if method_name == "No Compression":
        return COLORS[0]
    
    if method_name not in method_color_map:
        # Assign next available color (skipping 0)
        idx = (len(method_color_map) + 1) % len(COLORS)
        # Ensure we don't wrap around to 0 (Red)
        if idx == 0: idx = 1 
        method_color_map[method_name] = COLORS[idx]
    
    return method_color_map[method_name]

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory '{ROOT_DIR}' not found.")
        return

    # Data structure: data[target][method] = (rounds, acc)
    # targets: "Yi", "Zi", "YiZi", "None" (Baseline)
    data_store = {
        "Yi": {},
        "Zi": {},
        "YiZi": {},
        "None": {}
    }
    
    method_color_map = {}

    # 1. Scan and Load Data
    print(f"Scanning {ROOT_DIR}...")
    for folder in os.listdir(ROOT_DIR):
        full_path = os.path.join(ROOT_DIR, folder)
        if not os.path.isdir(full_path):
            continue
            
        method, target = parse_experiment_info(folder)
        rounds, acc = load_experiment_data(full_path)
        
        if rounds is not None and len(rounds) > 0:
            print(f"  Loaded: {folder} -> Method: {method}, Target: {target}")
            data_store[target][method] = (rounds, acc)
        else:
            print(f"  Skipped (No Data): {folder}")

    # 2. Setup Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    targets_order = ["Yi", "Zi", "YiZi"]
    titles = ["Compressing Yi", "Compressing Zi", "Compressing Yi & Zi"]

    baseline_data = data_store["None"].get("No Compression")
    
    # 3. Plotting Loop
    for i, target_key in enumerate(targets_order):
        ax = axes[i]
        
        # A. Plot Baseline (No Compression)
        if baseline_data:
            b_rounds, b_acc = baseline_data
            ax.plot(b_rounds, b_acc, label="No Compression", color=COLORS[0], linewidth=2, linestyle="--")
        else:
            print("Warning: No 'No Compression' baseline found.")

        # B. Plot Compressed Experiments for this Target
        experiments = data_store[target_key]
        for method_name, (rounds, acc) in experiments.items():
            color = get_color(method_name, method_color_map)
            ax.plot(rounds, acc, label=method_name, color=color, linewidth=2, marker='o', markersize=4)

        # Styling
        ax.set_title(titles[i], fontsize=14, weight='bold')
        ax.set_xlabel("Rounds", fontsize=12)
        if i == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        
        ax.set_ylim(Y_LIMITS)
        ax.set_xlim(0,len(rounds))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="lower right", fontsize=10)

    save_path = os.path.join(ROOT_DIR,OUTPUT_FILE)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")
    # plt.show()

if __name__ == "__main__":
    main()
