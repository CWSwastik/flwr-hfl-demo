import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
ROOT_DIR = "logs"  # Directory containing experiment subfolders
exp_dir = 'new_communication_new_mtgc_mnist'
# exp_dir = 'new_comm_paper_mtgc_mnist'
ROOT_DIR = os.path.join(ROOT_DIR, exp_dir)
OUTPUT_FILE = f"compression_acc_{exp_dir}.png"
Y_LIMITS = (0, 100)  # Adjust accuracy limits if needed

# --- COLORS (Converted to Tuples for Hashing) ---
# Index 0 (Red) -> FedAvg-GC (Baseline)
# Index 3 (Blue) -> FedProx (Baseline)
COLORS = [
    (1.0, 0.0, 0.0, 1.0),  # Red 0 (FedAvg-GC)
    (0.6627451, 0.8505867, 0.53165947, 1.0),  # Green 1
    (0.07254902, 0.88292761, 0.9005867, 1.0),  # Light Blue 2
    (0.0, 0.0, 1.0, 1.0),  # Blue 3 (FedProx)
    (1.0, 0.49803922, 0.05490196, 1.0),  # Orange 4
    (0.58039216, 0.40392157, 0.74117647, 1.0),  # Purple 5
    (0.54901961, 0.3372549, 0.29411765, 1.0),  # Brown 6
    (0.89019608, 0.46666667, 0.76078431, 1.0),  # Pink 7
    (0.49803922, 0.49803922, 0.49803922, 1.0),  # Gray 8
    (0.7372549, 0.74117647, 0.13333333, 1.0),  # Yellow 9
    (0.99215686, 0.70588235, 0.38431373, 1.0),  # Gold 10
    (0.15294118, 0.16078431, 0.16078431, 1.0),  # Black 11
    (0.97254902, 0.70588235, 0.09411765, 1.0),  # Apricot 12
    (0.85098039, 0.37254902, 0.00784314, 1.0),  # Cinnamon 13
    (0.90196078, 0.90196078, 0.98039216, 1.0),  # Lavender 14
    (0.90588235, 0.16078431, 0.54117647, 1.0),  # Magenta 15
    (0.4, 0.65098039, 0.11764706, 1.0),  # Lime 16
    (0.65098039, 0.4627451, 0.11372549, 1.0),  # Sienna 17
    (0.71764706, 0.81960784, 0.54607843, 1.0),  # Lime 18
    (0.83137255, 0.68627451, 0.21568627, 1.0),  # Sienna 19
    (0.0, 0.4, 0.4, 1.0),  # Dark Teal 20
    (0.6, 0.0, 1.0, 1.0),  # Electric Purple 21
    (0.0, 0.74, 0.67, 1.0),  # Turquoise Blue 22
    (0.15, 0.2, 0.05, 1.0),  # Dark Olive 23
]

def load_experiment_data(folder_path):
    """
    Reads accuracy data. Tries 'global_acc_loss.csv' first, then 'central_server.log'.
    """
    csv_path = os.path.join(folder_path, 'global_acc_loss.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'acc' in df.columns and 'round_numbers' in df.columns:
                return df['round_numbers'].values, df['acc'].values
        except:
            pass

    possible_paths = [
        os.path.join(folder_path, 'central_server.log'),
        os.path.join(folder_path, 'central', 'central_server.log')
    ]
    
    for log_path in possible_paths:
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    header = f.readline()
                    has_header = 'accuracy' in header.lower() or 'round' in header.lower()
                
                if has_header:
                    df = pd.read_csv(log_path)
                else:
                    df = pd.read_csv(log_path, names=['round', 'loss', 'accuracy'])
                
                df.columns = [c.strip().lower() for c in df.columns]
                
                if 'accuracy' in df.columns and 'round' in df.columns:
                    return df['round'].values, df['accuracy'].values * 100 
                elif 'acc' in df.columns:
                     return df['round'].values, df['acc'].values
            except Exception as e:
                print(f"Failed to read log {log_path}: {e}")
                pass
            
    return None, None

def parse_experiment_info(folder_name):
    """
    Identifies: FedProx, FedAvg-GC, or Compression Experiments
    """
    lower_name = folder_name.lower()
    
    if "fedprox" in lower_name:
        return "FedProx", "Baseline"

    target = "None"
    if "_YiZi_" in folder_name: target = "YiZi"
    elif "_Yi_" in folder_name: target = "Yi"
    elif "_Zi_" in folder_name: target = "Zi"
    
    if "compress_" in folder_name:
        match = re.search(r"compress_([a-zA-Z0-9_]+)_(?:Yi|Zi|YiZi)", folder_name)
        if match:
            method_name = match.group(1)
        else:
            method_name = "Compressed"
            
        method_display = method_name.replace("_", " ").title()
        if "Quant" in method_display: method_display = method_display.replace("Quant", "Quantization")
        if "Pct" in method_display: method_display = method_display.replace("Pct", "%")
        return method_display, target

    if "fedavg-gc" in lower_name:
        #return "MTGC (No Comp)", "Baseline"
        return "Hier-Scaffold (No Comp)", "Baseline"
    else:
        return "FedAvg", "Baseline"

# --- HELPER: NATURAL SORT ---
def natural_sort_key(s):
    """
    Sorts strings containing numbers naturally (e.g., '2' < '10').
    Splits string into list of integers and text.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

# --- PARAMETER-BASED COLOR MAPPING ---
def get_color(method_name, param_color_map):
    # Fixed Baseline Colors
    # if "MTGC" in method_name:
    if "Hier-Scaffold" in method_name:
        return COLORS[0]  # Red
    if "FedProx" in method_name:
        return COLORS[3]  # Blue
    
    # 1. Extract the Parameter (the number) from the name
    match = re.search(r"(\d+(?:\.\d+)?)", method_name)
    param_key = match.group(1) if match else method_name

    # 2. Assign Color based on Parameter
    if param_key not in param_color_map:
        used_colors = set(param_color_map.values())
        # Avoid Red (0) and Blue (3)
        reserved_indices = {0, 3}
        
        found = False
        for i in range(len(COLORS)):
            if i not in reserved_indices and COLORS[i] not in used_colors:
                param_color_map[param_key] = COLORS[i]
                found = True
                break
        
        # Fallback if we run out of colors
        if not found:
            param_color_map[param_key] = COLORS[(len(param_color_map) + 5) % len(COLORS)]

    return param_color_map[param_key]

def get_family_name(method_name):
    return method_name.split()[0]

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory '{ROOT_DIR}' not found.")
        return

    data_store = {
        "Yi": {},
        "Zi": {},
        "YiZi": {},
        "Baseline": {}
    }
    
    # Shared map to ensure '10%' is same color everywhere
    param_color_map = {}

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

    # 2. Identify Compression Families
    all_methods = set()
    for t in ["Yi", "Zi", "YiZi"]:
        for m in data_store[t]:
            all_methods.add(m)
            
    if not all_methods:
        print("No compressed experiments found to plot.")
        return

    families = sorted(list(set(get_family_name(m) for m in all_methods)))
    nrows = 3
    ncols = len(families)
    
    # 3. Setup Plot
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 12), sharey=True, squeeze=False)
    
    targets_order = ["Yi", "Zi", "YiZi"]
    row_titles = ["Compressing Yi", "Compressing Zi", "Compressing Yi & Zi"]
    
    # Retrieve Baselines
    baselines = data_store["Baseline"]
    
    print(f"\nGenerating {nrows}x{ncols} grid for families: {families}")

    # 4. Plotting Loop
    for r, target_key in enumerate(targets_order):
        for c, family in enumerate(families):
            ax = axes[r, c]
            
            # A. Plot ALL Baselines (FedProx & FedAvg-GC)
            for b_name, (b_rounds, b_acc) in baselines.items():
                b_color = get_color(b_name, param_color_map)
                ax.plot(b_rounds, b_acc, label=b_name, color=b_color, linewidth=2, linestyle="--")

            # B. Plot Compression Methods for this Family & Target
            experiments = data_store[target_key]
            
            # Filter methods belonging to this family
            family_methods = [m for m in experiments.keys() if get_family_name(m) == family]
            
            # Sort Naturally (2 < 10)
            family_methods.sort(key=natural_sort_key)

            for method_name in family_methods:
                rounds, acc = experiments[method_name]
                color = get_color(method_name, param_color_map)
                ax.plot(rounds, acc, label=method_name, color=color, linewidth=2, marker='o', markersize=4)

            # C. Styling
            if r == 0:
                ax.set_title(f"{family} Techniques", fontsize=14, weight='bold')
            if c == 0:
                ax.set_ylabel(f"{row_titles[r]}\nAccuracy (%)", fontsize=12, weight='bold')
            if r == nrows - 1:
                ax.set_xlabel("Rounds", fontsize=12)

            ax.set_ylim(Y_LIMITS)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc="lower right", fontsize=10)

    save_path = os.path.join(ROOT_DIR, OUTPUT_FILE)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")

if __name__ == "__main__":
    main()
