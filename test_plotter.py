import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import re

class Plotter:
    """
    Plotter for central server, client, and edge logs using Regex grouping and renaming.
    """

    colors = [
    [1.0, 0.0, 0.0, 1.0],  # Red 1
    [0.0, 0.0, 1.0, 1.0],  # Blue 2
    [0.6, 0.0, 1.0, 1.0],  # Electric Purple 3
    [1.0, 0.49803922, 0.05490196, 1.0],  # Orange 4 
    [0.0, 0.4, 0.4, 1.0],  # Dark Teal 5
    [0.54901961, 0.3372549, 0.29411765, 1.0],  # Brown 6
    [0.89019608, 0.46666667, 0.76078431, 1.0],  # Pink 7
    [0.00000000, 0.85000000, 0.02881356, 1.0],  # Lime 8
    [0.54444444, 0.00000000, 0.00000000, 1.0],  # Darkred 9
    [0.0, 0.74, 0.67, 1.0],  # Turquoise Blue 10
    [0.90588235, 0.16078431, 0.54117647, 1.0],  # Magenta 11
    [0.7372549, 0.74117647, 0.13333333, 1.0],  # Yellow 12
    [0.15294118, 0.16078431, 0.16078431, 1.0],  # Black 13
    [0.71764706, 0.81960784, 0.54607843, 1.0],  # Lime 14
    [0.85098039, 0.37254902, 0.00784314, 1.0],  # Cinnamon 15
    [0.15, 0.2, 0.05, 1.0],  # Dark Olive 16
    
    [0.4, 0.65098039, 0.11764706, 1.0],  # Lime 17
    [0.6627451, 0.8505867, 0.53165947, 1.0],  # Green 18
    [0.99215686, 0.70588235, 0.38431373, 1.0],  # Peach 19
    [0.07254902, 0.88292761, 0.9005867, 1.0],  # Light Blue 20
    [0.99215686, 0.70588235, 0.38431373, 1.0],  # Gold 21
    [0.97254902, 0.70588235, 0.09411765, 1.0],  # Apricot 22
    [0.90196078, 0.90196078, 0.98039216, 1.0],  # Lavender 23
    [0.65098039, 0.4627451, 0.11372549, 1.0],  # Sienna 24
    [0.83137255, 0.68627451, 0.21568627, 1.0],  # Sienna 25
    [0.49803922, 0.49803922, 0.49803922, 1.0],  # Gray 26
    ]
    
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':', (5, (10, 3)), (0, (5, 5)), (0, (5, 1))]

    def __init__(self, logs_dir='logs', exp_groupers=None):
        self.logs_dir = logs_dir
        # Default to capturing everything if no grouper provided
        self.exp_groupers = exp_groupers if exp_groupers else { 'All': r'.*'}
        self.plot_dir = 'plots'
        
        self.exp_style_map = {} 
        self.color_idx = 0
        self.linestyle_idx = 0
        self.marker_idx = 0
        
        os.makedirs(self.plot_dir, exist_ok=True)

    def generate_legend_label(self, folder_name):
        """
        Renames folder_name to <dataset_partition>-cluster_<strategy>-<gc>
        Removes prefix: <dataset>-<num_clients>c-<model>-
        """
        # Regex: Start -> (anything not -) -> - -> (digits)c -> - -> (anything not -) -> - -> (CAPTURE REST)
        pattern = r"^[^-]+-\d+c-[^-]+-(.+)$"
        match = re.search(pattern, folder_name)
        if match:
            return match.group(1)
        return folder_name

    def get_exp_style(self, exp_name):
        if exp_name not in self.exp_style_map:
            color = self.colors[self.color_idx % len(self.colors)]
            linestyle = self.linestyles[self.linestyle_idx % len(self.linestyles)]
            marker = self.markers[self.marker_idx % len(self.markers)]
            self.exp_style_map[exp_name] = (color, linestyle, marker)
            self.color_idx += 1
            self.linestyle_idx += 1
            self.marker_idx += 1
        return self.exp_style_map[exp_name]

    def load_central_server_data(self, exp_name):
        log_path = os.path.join(self.logs_dir, exp_name, 'central', 'central_server.log')
        if os.path.exists(log_path):
            return pd.read_csv(log_path)
        return None

    def load_edge_data_filtered(self, exp_list):
        """Load edge logs only for experiments in the current list."""
        edge_data = {}
        for exp_name in exp_list:
            edge_dir = os.path.join(self.logs_dir, exp_name, 'edge')
            if os.path.exists(edge_dir):
                edge_files = glob.glob(os.path.join(edge_dir, '*.log'))
                for edge_file in edge_files:
                    edge_name = os.path.basename(edge_file).replace('.log', '')
                    df = pd.read_csv(edge_file)
                    if edge_name not in edge_data:
                        edge_data[edge_name] = {}
                    edge_data[edge_name][exp_name] = df
        return edge_data

    # ---------------- PLOTTING FUNCTIONS ---------------- #

    def plot_central_server_accuracy(self, exp_list, group_name):
        """Plot central server accuracy for a specific group of experiments."""
        if not exp_list: return

        plt.figure(figsize=(12, 6))
        has_data = False
        
        # Sort based on the renamed label for clean legend order
        exp_list_sorted = sorted(exp_list, key=lambda x: self.generate_legend_label(x))

        for exp_name in exp_list_sorted:
            df = self.load_central_server_data(exp_name)
            if df is not None and 'round' in df.columns and 'accuracy' in df.columns:
                has_data = True
                color, linestyle, marker = self.get_exp_style(exp_name)
                label = self.generate_legend_label(exp_name)
                
                plt.plot(df['round'], df['accuracy'], label=label,
                        color=color, linestyle='-', linewidth=2, markersize=5)

        if has_data:
            plt.xlabel('Round', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.title(f'Central Server Accuracy ({group_name})', fontsize=16)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.ylim(0, 1)
            plt.legend(loc='lower right', fontsize=12)
            plt.tight_layout()
            
            out_file = f"{group_name}_central_accuracy.png"
            plt.savefig(os.path.join(self.plot_dir, out_file), dpi=300)
            print(f"[INFO] Saved {out_file}")
            plt.close()

    def plot_edge_accuracy_subplots(self, exp_list, group_name):
        """Plot edge accuracy subplots for a specific group."""
        edge_data = self.load_edge_data_filtered(exp_list)
        if not edge_data: return

        sorted_edges = sorted(edge_data.keys())
        num_edges = len(sorted_edges)
        if num_edges == 0: return
        
        ncols = int(np.ceil(np.sqrt(num_edges)))
        nrows = int(np.ceil(num_edges / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if num_edges == 1: axes = [axes]
        else: axes = axes.flatten()

        # Shared Y-axis limits
        all_acc = []
        for en in edge_data:
            for ex in edge_data[en]:
                if 'accuracy' in edge_data[en][ex]:
                    all_acc.extend(edge_data[en][ex]['accuracy'].values)
        y_min, y_max = (0, 1)
        if all_acc:
            y_min = max(0, min(all_acc) - 0.05)
            y_max = min(1, max(all_acc) + 0.05)

        exp_list_sorted = sorted(exp_list, key=lambda x: self.generate_legend_label(x))

        for idx, edge_name in enumerate(sorted_edges):
            ax = axes[idx]
            exp_dict = edge_data[edge_name]
            
            for exp_name in exp_list_sorted:
                if exp_name in exp_dict:
                    df = exp_dict[exp_name]
                    if 'accuracy' in df.columns:
                        color, _, _ = self.get_exp_style(exp_name)
                        label = self.generate_legend_label(exp_name)
                        ax.plot(df['round'], df['accuracy'], label=label,
                                color=color, linestyle='-', linewidth=1.5)

            ax.set_ylim(y_min, y_max)
            ax.set_title(edge_name, fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)
            if idx == 0: ax.legend(loc='lower right', fontsize=8) # Legend only on first

        # Hide unused
        for idx in range(num_edges, len(axes)): axes[idx].set_visible(False)
        
        plt.suptitle(f'Edge Accuracy ({group_name})', fontsize=16)
        plt.tight_layout()
        out_file = f"{group_name}_edge_accuracy.png"
        plt.savefig(os.path.join(self.plot_dir, out_file), dpi=300)
        print(f"[INFO] Saved {out_file}")
        plt.close()

    def process_groups(self):
        """Iterates through defined groups, filters experiments, and generates plots."""
        all_dirs = [d for d in os.listdir(self.logs_dir) if os.path.isdir(os.path.join(self.logs_dir, d))]
        
        for group_name, regex_pattern in self.exp_groupers.items():
            # Filter experiments for this group
            matched_exps = [d for d in all_dirs if re.search(regex_pattern, d)]
            
            if not matched_exps:
                print(f"[WARN] No experiments found for group: {group_name}")
                continue
            
            print(f"--- Processing Group: {group_name} ({len(matched_exps)} experiments) ---")
            
            # Reset styles so colors recycle per group (optional, remove if you want global consistency)
            self.exp_style_map = {} 
            self.color_idx = 0
            
            self.plot_central_server_accuracy(matched_exps, group_name)
            self.plot_edge_accuracy_subplots(matched_exps, group_name)

if __name__ == '__main__':
    # Update logs directory
    LOGS_DIR = 'logs/mnist' 

    # Define your regex groups here (Similar to partition_based script)
    EXP_GROUPERS = {
        'Dirichlet_0.1': r'dirichlet_0.1',
        'Dirichlet_0.5': r'dirichlet_0.5',
        'IID': r'iid',
        # 'All_Experiments': r'.*' 
    }

    plotter = Plotter(logs_dir=LOGS_DIR, exp_groupers=EXP_GROUPERS)
    plotter.process_groups()
