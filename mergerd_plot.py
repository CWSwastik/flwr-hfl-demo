import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import re
import math
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

class Plotter:
    """
    Unified Plotter for Central, Client, Edge logs and Grid Comparisons.
    Merges functionality from plot_results.py and test_plotter.py.
    """

    # --- 1. Style Definitions ---
    # Consistent cycling colors for individual clients/edges or strategies
    colors = [
    [1.0, 0.0, 0.0, 1.0],  # Red 0
    [0.0, 0.0, 1.0, 1.0],  # Blue 1
    [0.6, 0.0, 1.0, 1.0],  # Electric Purple 2
    [1.0, 0.49803922, 0.05490196, 1.0],  # Orange 3
    [0.0, 0.4, 0.4, 1.0],  # Dark Teal 4
    [0.54901961, 0.3372549, 0.29411765, 1.0],  # Brown 5
    [0.89019608, 0.46666667, 0.76078431, 1.0],  # Pink 6
    [0.00000000, 0.85000000, 0.02881356, 1.0],  # Lime 7
    [0.54444444, 0.00000000, 0.00000000, 1.0],  # Darkred 8
    [0.0, 0.74, 0.67, 1.0],  # Turquoise Blue 9
    [0.90588235, 0.16078431, 0.54117647, 1.0],  # Magenta 10
    [0.7372549, 0.74117647, 0.13333333, 1.0],  # Yellow 11
    [0.15294118, 0.16078431, 0.16078431, 1.0],  # Black 12
    [0.71764706, 0.81960784, 0.54607843, 1.0],  # Lime 13
    [0.85098039, 0.37254902, 0.00784314, 1.0],  # Cinnamon 14
    [0.15, 0.2, 0.05, 1.0],  # Dark Olive 15
    
    [0.4, 0.65098039, 0.11764706, 1.0],  # Lime 16
    [0.6627451, 0.8505867, 0.53165947, 1.0],  # Green 17
    [0.99215686, 0.70588235, 0.38431373, 1.0],  # Peach 18
    [0.07254902, 0.88292761, 0.9005867, 1.0],  # Light Blue 19
    [0.99215686, 0.70588235, 0.38431373, 1.0],  # Gold 20
    [0.97254902, 0.70588235, 0.09411765, 1.0],  # Apricot 21
    [0.90196078, 0.90196078, 0.98039216, 1.0],  # Lavender 22
    [0.65098039, 0.4627451, 0.11372549, 1.0],  # Sienna 23
    [0.83137255, 0.68627451, 0.21568627, 1.0],  # Sienna 24
    [0.49803922, 0.49803922, 0.49803922, 1.0],  # Gray 25
]
    
    # Updated Semantic colors using the list above
    semantic_colors = {
        'no_gc': colors[0],  # Red
        'gc':    colors[1],  # Blue
        'fedprox_gc': colors[9], # Turquoise Blue
        'fedprox': colors[3], # Orange
        'fedmut': colors[7], # Lime
        'none': [0.5, 0.5, 0.5, 1.0], # Gray
        'dissimilar': colors[4] # Teal
    }

    # Markers for lines
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    
    # Regex Maps for Automatic Grouping
    PARTITION_MAP = {
        'Dirichlet 0.1': r'dirichlet_0.1',
        'Dirichlet 0.5': r'dirichlet_0.5', 
        'IID': r'iid', 
    }

    # Strategy Regex Map
    STRATEGY_MAP = {
        'No Clustering': r'cluster_none',
        'Cluster - (EMD)':  r'cluster_emd',   
        'Cluster - (JSD)':  r'cluster_jsd',
        'Cluster - (Cosine)':  r'cluster_cosine',
    }

    linestyles_map = {
        'fedavg': '-',                # Solid
        'fedprox': '--',              # Dashed
        'fedmut': '-.',               # Dash-Dot
        'dissimilar': ':',            # Dotted (for Dissimilar variants)
        'cluster': (0, (3, 1, 1, 1)), # Densely Dash-Dotted
        'default': ':'                # Fallback
    }

    def __init__(self, logs_dir='logs'):
        self.logs_dir = logs_dir
        
        # Auto-detect dataset for plot directory
        self.dataset_name = os.path.basename(os.path.normpath(logs_dir))
        if self.dataset_name == 'logs': 
            self.dataset_name = 'experiment'
            
        self.plot_dir = os.path.join('plots', self.dataset_name)
        os.makedirs(self.plot_dir, exist_ok=True)
        print(f"[Init] Plots will be saved to: {self.plot_dir}")

    # ---------------- DATA LOADING ---------------- #

    def load_central_data(self, exp_name):
        path = os.path.join(self.logs_dir, exp_name, 'central', 'central_server.log')
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    def load_distribution_data(self, exp_name, filename):
        path = os.path.join(self.logs_dir, exp_name, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return None
    
    def get_linestyle(self, exp_name):
        if 'dissimilar_cluster' in exp_name: return self.linestyles_map['dissimilar']
        if 'fedavg' in exp_name: return self.linestyles_map['fedavg']
        if 'fedprox' in exp_name: return self.linestyles_map['fedprox']
        if 'fedmut' in exp_name: return self.linestyles_map['fedmut']
        return self.linestyles_map['default']

    # ---------------- PLOTTING: ORIGINAL GRID ---------------- #

    def plot_grid_comparison(self, output_file="central_acc_grid_gc_vs_nogc", include_dissimilar=False):
        """
        Original 3x4 Grid.
        include_dissimilar: If True, plots 'fedavg-dissimilar' as dotted lines. 
                            If False, explicitly filters them out.
        """
        print(f"Generating Grid (GC vs No-GC)... Dissimilar included: {include_dissimilar}")
        
        rows = list(self.PARTITION_MAP.keys())
        cols = list(self.STRATEGY_MAP.keys())
        
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(20, 12), sharey=True, sharex=True)
        if len(rows) == 1: axes = np.array([axes])
            
        all_dirs = [d for d in os.listdir(self.logs_dir) if os.path.isdir(os.path.join(self.logs_dir, d))]

        for i, row_label in enumerate(rows):
            part_regex = self.PARTITION_MAP[row_label]
            
            for j, col_label in enumerate(cols):
                strat_regex = self.STRATEGY_MAP[col_label]
                ax = axes[i, j]
                
                cell_exps = [d for d in all_dirs if re.search(part_regex, d) and re.search(strat_regex, d)]
                
                # Filter out dissimilar if not requested
                if not include_dissimilar:
                    cell_exps = [d for d in cell_exps if 'dissimilar_cluster' not in d]
                
                for exp_name in cell_exps:
                    df = self.load_central_data(exp_name)
                    if df is None or 'accuracy' not in df: continue
                    
                    color = None
                    label = None
                    style = '-' # Default solid
                    
                    is_dissimilar = 'dissimilar_cluster' in exp_name
                    is_gc = '-gc' in exp_name
                    
                    # Logic for labels and styles
                    if 'fedavg' in exp_name and is_gc:
                        color = self.semantic_colors['gc']
                        base_label = "FedAvg With GC"
                        # print("fedavg (GC): ",exp_name)
                    if 'fedprox' in exp_name and is_gc:
                        color = self.semantic_colors['fedprox_gc']
                        base_label = "Fedprox With GC"
                        # print("fedprox (GC): ",exp_name)
                    elif 'fedprox' in exp_name:
                        color = self.semantic_colors['fedprox']
                        base_label = "Fedprox Without GC"
                        # print("fedprox (Without GC): ",exp_name)
                    elif 'fedmut' in exp_name:
                        color = self.semantic_colors['fedmut']
                        base_label = "Fedmut Without GC"
                    elif 'fedavg' in exp_name and not is_gc:
                        color = self.semantic_colors['no_gc']
                        base_label = "FedAvg Without GC"
                        # print("fedavg (Without GC): ",exp_name)


                    if is_dissimilar:
                        base_label += "FedAvg (GC) (Dissimilar)"
                        style = self.linestyles_map['dissimilar'] # Dotted for dissimilar
                    elif 'fedprox' in exp_name: style = self.linestyles_map['fedprox']
                    elif 'fedmut' in exp_name: style = self.linestyles_map['fedmut']

                    zorder = 10 if '-gc' in exp_name else 5
                    
                    ax.plot(df['round'], df['accuracy'], label=base_label, 
                            color=color, linestyle=style, linewidth=1, zorder=zorder)
                    ax.yaxis.set_major_locator(MultipleLocator(0.1))
                    ax.xaxis.set_major_locator(MultipleLocator(10))

                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_ylim(0.0, 1.0)
                
                if i == 0: ax.set_title(col_label, fontsize=14, fontweight='bold')
                if j == 0: ax.set_ylabel(f"{row_label}\nAccuracy", fontsize=14, fontweight='bold')
                if i == len(rows) - 1: ax.set_xlabel("Rounds", fontsize=12)

        # Legend
        custom_lines = [
            Line2D([0], [0], color=self.semantic_colors['no_gc'], lw=2, label='FedAvg (No GC)'),
            Line2D([0], [0], color=self.semantic_colors['gc'], lw=2, label='FedAvg (GC)'),
            Line2D([0], [0], color=self.semantic_colors['fedprox'], lw=2, linestyle=self.linestyles_map['fedprox'], label='FedProx (No GC)'),
            Line2D([0], [0], color=self.semantic_colors['fedprox_gc'], lw=2, linestyle=self.linestyles_map['fedprox'], label='FedProx (GC)'),
            Line2D([0], [0], color=self.semantic_colors['fedmut'], lw=2, linestyle=self.linestyles_map['fedmut'], label='FedMut'),
        ]
        if include_dissimilar:
            custom_lines.append(Line2D([0], [0], color=self.semantic_colors['dissimilar'], lw=2, linestyle=self.linestyles_map['dissimilar'], label='Dissimilar Cluster FedAvg (GC)'))

        fig.legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncols=len(custom_lines), fontsize=12)
        
        plt.suptitle(f"Comparision of Central Accuracy Across Algorithms: {self.dataset_name.upper()}", fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.92)
        
        tag = "_with_dissimilar" if include_dissimilar else ""
        out_path = os.path.join(self.plot_dir, f'{output_file}_{self.dataset_name}{tag}.png')
        plt.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")
        plt.close()
    # ---------------- PLOTTING: NEW CLUSTERING IMPACT GRID ---------------- #
    
    def plot_clustering_impact_grid(self, 
                                  ymin=0.0, 
                                  output_file="central_acc_grid_clustering_impact",
                                  partition_filter=None, 
                                  training_strategy_filter=None):
        print(f"Generating Clustering Impact Grid (Hiding Dissimilar)...")

        rows = list(self.PARTITION_MAP.keys())
        if partition_filter: rows = [r for r in rows if r in partition_filter]
            
        cols = ["Without GC", "With GC", "Difference (FedAvg GC - FedAvg)"]
        strategies = list(self.STRATEGY_MAP.keys())
        
        strategy_colors = {
            'No Clustering': self.colors[12],      # Black
            'Cluster - (EMD)': self.colors[4],    # Teal
            'Cluster - (JSD)': self.colors[3],    # Orange
            'Cluster - (Cosine)': self.colors[2], # Purple
        }

        fig, axes = plt.subplots(len(rows), len(cols), figsize=(20, 4.5 * len(rows)), sharex=True, sharey=False)
        if len(rows) == 1: axes = np.array([axes])
            
        all_dirs = [d for d in os.listdir(self.logs_dir) if os.path.isdir(os.path.join(self.logs_dir, d))]
        if training_strategy_filter:
            all_dirs = [d for d in all_dirs if re.search(training_strategy_filter, d)]

        # EXPLICIT FILTER: Remove 'dissimilar' to keep this plot clean
        all_dirs = [d for d in all_dirs if 'dissimilar_cluster' not in d]

        legend_handles = {} 

        for i, row_label in enumerate(rows):
            part_regex = self.PARTITION_MAP[row_label]

            for s_idx, strat_label in enumerate(strategies):
                strat_regex = self.STRATEGY_MAP[strat_label]
                
                exps_nogc = []
                exps_gc = []

                for d in all_dirs:
                    if re.search(part_regex, d) and re.search(strat_regex, d):
                        if d.endswith('-gc'): exps_gc.append(d)
                        else: exps_nogc.append(d)

                color = strategy_colors.get(strat_label, [0.5, 0.5, 0.5, 1.0])
                width = 1.5 if strat_label == 'No Clustering' else 1.2

                def plot_set(ax, exp_list):
                    main_df = None
                    for exp_name in exp_list:
                        df = self.load_central_data(exp_name)
                        if df is not None and 'accuracy' in df:
                            if main_df is None: main_df = df
                            linestyle = self.get_linestyle(exp_name)
                            
                            subtype = "cluster"
                            for key in ['fedavg', 'fedprox', 'fedmut']:
                                if key in exp_name: subtype = key; break
                            
                            label_key = f"{strat_label} ({subtype})"
                            
                            ax.plot(df['round'], df['accuracy'], label=label_key, 
                                    color=color, linestyle=linestyle, linewidth=width)
                            ax.yaxis.set_major_locator(MultipleLocator(0.1))
                            ax.xaxis.set_major_locator(MultipleLocator(10))

                            if label_key not in legend_handles:
                                legend_handles[label_key] = Line2D([0], [0], color=color, linestyle=linestyle, lw=width)
                    return main_df

                df_nogc = plot_set(axes[i, 0], exps_nogc)
                df_gc = plot_set(axes[i, 1], exps_gc)

                # Diff Plot (FedAvg only)
                fedavg_gc_exp = next((exp for exp in exps_gc if 'fedavg' in exp), None)
                fedavg_nogc_exp = next((exp for exp in exps_nogc if 'fedavg' in exp), None)

                if fedavg_gc_exp and fedavg_nogc_exp:
                    df_gc_f = self.load_central_data(fedavg_gc_exp)
                    df_nogc_f = self.load_central_data(fedavg_nogc_exp)
                    if df_gc_f is not None and df_nogc_f is not None:
                        merged = pd.merge(df_gc_f[['round', 'accuracy']], df_nogc_f[['round', 'accuracy']], 
                                          on='round', suffixes=('_gc', '_nogc'))
                        merged['diff'] = merged['accuracy_gc'] - merged['accuracy_nogc']
                        lstyle = self.linestyles_map['fedavg']
                        axes[i, 2].plot(merged['round'], merged['diff'], color=color, linestyle=lstyle, linewidth=width)

            # Styling (Same as before)
            for j in range(3):
                ax = axes[i, j]
                ax.grid(which='major', linestyle=':', linewidth=0.8, alpha=0.8)
                ax.minorticks_on()
                ax.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.3)
                if j < 2: ax.set_ylim(ymin, 1.0)
                else: ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                if i == 0: ax.set_title(cols[j], fontsize=16, fontweight='bold')
                if j == 0: ax.set_ylabel(f"{row_label}\nAccuracy", fontsize=14, fontweight='bold')
                if i == len(rows) - 1: ax.set_xlabel("Rounds", fontsize=12)

        # Legend Logic (Same as before)
        type_priority = ['fedavg', 'fedprox', 'fedmut']
        strat_priority = ['No Clustering', 'Cluster - (EMD)', 'Cluster - (JSD)', 'Cluster - (Cosine)']
        
        def get_sort_key(label):
            match = re.match(r"(.+) \((.+)\)", label)
            if match:
                s_name, t_name = match.groups()
                s_rank = strat_priority.index(s_name) if s_name in strat_priority else 99
                t_rank = type_priority.index(t_name) if t_name in type_priority else 99
                return s_rank * 100 + t_rank
            return 999

        sorted_labels = sorted(legend_handles.keys(), key=get_sort_key)
        sorted_handles = [legend_handles[l] for l in sorted_labels]

        fig.legend(handles=sorted_handles, labels=sorted_labels, 
                   loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12, frameon=False)

        title_suffix = f" ({training_strategy_filter})" if training_strategy_filter else ""
        plt.suptitle(f"Clustering Impact{title_suffix}: {self.dataset_name.upper()}", fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.14, top=0.92)

        filter_tag = re.sub(r'[^a-zA-Z0-9]', '', training_strategy_filter) if training_strategy_filter else "all"
        out_path = os.path.join(self.plot_dir, f'{output_file}_{self.dataset_name}_{filter_tag}.png')
        plt.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")
        plt.close()

    def plot_dissimilar_impact_grid(self, output_file="central_acc_grid_dissimilar_impact"):
        """
        New 3x4 Grid: Impact of Dissimilar Clustering on FedAvg.
        """
        print(f"Generating Dissimilar Clustering Impact Grid...")
        
        rows = list(self.PARTITION_MAP.keys())
        cols = list(self.STRATEGY_MAP.keys())
        
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(20, 4.5 * len(rows)), sharey=True, sharex=True)
        if len(rows) == 1: axes = np.array([axes])
            
        all_dirs = [d for d in os.listdir(self.logs_dir) if os.path.isdir(os.path.join(self.logs_dir, d))]

        for i, row_label in enumerate(rows):
            part_regex = self.PARTITION_MAP[row_label]
            
            for j, col_label in enumerate(cols):
                strat_regex = self.STRATEGY_MAP[col_label]
                ax = axes[i, j]
                
                # Get relevant experiments
                cell_exps = [d for d in all_dirs if re.search(part_regex, d) and re.search(strat_regex, d)]
                
                # 1. Filter for 'fedavg' base only (ignore fedprox/fedmut for clarity)
                cell_exps = [d for d in cell_exps if 'fedavg' in d]
                
                # 2. Filter out any that are NOT 'dissimilar' OR 'standard fedavg'
                # (e.g. if you had other variants)
                
                for exp_name in cell_exps:
                    df = self.load_central_data(exp_name)
                    if df is None or 'accuracy' not in df: continue
                    
                    is_gc = '-gc' in exp_name
                    color = self.semantic_colors['gc'] if is_gc else self.semantic_colors['no_gc']
                    
                    is_dissimilar = 'dissimilar_cluster' in exp_name
                    linestyle = '--' if is_dissimilar else '-' 
                    
                    mode_label = "Dissimilar" if is_dissimilar else "Standard"
                    gc_label = "GC" if is_gc else "No GC"
                    label = f"FedAvg {mode_label} ({gc_label})"
                    
                    zorder = 10 if is_gc else 5
                    
                    ax.plot(df['round'], df['accuracy'], label=label, 
                            color=color, linestyle=linestyle, linewidth=1.5, zorder=zorder)
                    ax.yaxis.set_major_locator(MultipleLocator(0.1))
                    ax.xaxis.set_major_locator(MultipleLocator(10))

                ax.grid(which='major', linestyle=':', linewidth=0.8, alpha=0.8)
                ax.minorticks_on()
                ax.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.3)
                ax.set_ylim(0.0, 1.0)
                
                if i == 0: ax.set_title(col_label, fontsize=14, fontweight='bold')
                if j == 0: ax.set_ylabel(f"{row_label}\nAccuracy", fontsize=14, fontweight='bold')
                if i == len(rows) - 1: ax.set_xlabel("Rounds", fontsize=12)

        # Legend
        custom_lines = [
            Line2D([0], [0], color=self.semantic_colors['no_gc'], linestyle='-', lw=2, label='Standard (No GC)'),
            Line2D([0], [0], color=self.semantic_colors['gc'],    linestyle='-', lw=2, label='Standard (+ GC)'),
            Line2D([0], [0], color=self.semantic_colors['no_gc'], linestyle='--', lw=2, label='Dissimilar (No GC)'),
            Line2D([0], [0], color=self.semantic_colors['gc'],    linestyle='--', lw=2, label='Dissimilar (+ GC)'),
        ]
        
        fig.legend(handles=custom_lines, loc='lower center', 
                   bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=12, frameon=False)
        
        plt.suptitle(f"Dissimilar Clustering Impact: {self.dataset_name.upper()}", fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.14, top=0.92)
        
        out_path = os.path.join(self.plot_dir, f'{output_file}_{self.dataset_name}.png')
        plt.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")
        plt.close()
    
    # ---------------- PLOTTING: DISTRIBUTIONS ---------------- #

    def plot_distributions(self):
        """Generates distribution plots for every experiment."""
        print(f"Generating Distribution Plots...")
        exp_names = [d for d in os.listdir(self.logs_dir) if os.path.isdir(os.path.join(self.logs_dir, d))]

        for exp_name in sorted(exp_names):
            df = self.load_distribution_data(exp_name, "distribution_post_clustering.csv")
            if df is not None:
                self._plot_stacked_bar(df, x_col="ClientName", sort_by="ClusterID", 
                                     title=f"{exp_name} - Client Dist", 
                                     filename=f"{exp_name}_dist_clients.png")
            
            df_edge = self.load_distribution_data(exp_name, "cluster_distribution.csv")
            if df_edge is not None:
                 self._plot_stacked_bar(df_edge, x_col="ClusterID", sort_by="ClusterID", 
                                     title=f"{exp_name} - Edge Dist", 
                                     filename=f"{exp_name}_dist_edges.png")

    def _plot_stacked_bar(self, df, x_col, sort_by, title, filename):
        class_cols = [c for c in df.columns if c.startswith("Class_")]
        class_cols.sort(key=lambda x: int(x.split('_')[1]))
        
        if sort_by in df.columns:
            df = df.sort_values(by=[sort_by, x_col])

        data = df[class_cols].values
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 
        data_pct = data / row_sums
        labels = df[x_col].tolist()

        fig, ax = plt.subplots(figsize=(max(10, len(labels)*0.5), 6))
        bottom = np.zeros(len(labels))
        cmap = plt.get_cmap("tab20")

        for i, col in enumerate(class_cols):
            ax.bar(labels, data_pct[:, i], bottom=bottom, label=f"C{i}", color=cmap(i % 20), width=0.8)
            bottom += data_pct[:, i]

        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction")
        plt.xticks(rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Class")
        plt.tight_layout()
        os.makedirs(os.path.join(self.plot_dir, 'distribution'), exist_ok=True)
        plt.savefig(os.path.join(self.plot_dir, 'distribution', filename), dpi=150)
        plt.close()

if __name__ == '__main__':
    # POINT THIS TO YOUR DATASET LOGS
    exps = [
        {'log_dir':'logs/mnist', 'ymin':0.4}, 
        {'log_dir':'logs/fashion_mnist', 'ymin':0.0},
        {'log_dir':'logs/cifar10', 'ymin':0.0}
        ]

    for exp in exps:
        plotter = Plotter(logs_dir=exp['log_dir'])
        
        # 1. Original Grid: GC vs No-GC
        plotter.plot_grid_comparison(include_dissimilar=True)

        # 2. New Grid: Clustering Impact (No Cluster vs Algos)
        # plotter.plot_clustering_impact_grid(ymin=exp['ymin'])
        plotter.plot_clustering_impact_grid(
        ymin=exp['ymin'],
        training_strategy_filter=r'fedavg|fedprox'
        )

        plotter.plot_dissimilar_impact_grid()

        # 3. Distributions
        # plotter.plot_distributions()
    
