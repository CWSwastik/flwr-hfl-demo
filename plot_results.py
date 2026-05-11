import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import concurrent.futures

class Plotter:
    """
    Plotter for central server, client, and edge logs following the style of the reference script.
    Uses consistent colors and linestyles per experiment.
    """

    colors = [
        [1.0, 0.0, 0.0, 1.0],  # Red 1
        [0.0, 0.0, 1.0, 1.0],  # Blue 2
        [0.6, 0.0, 1.0, 1.0],  # Electric Purple 3
        [1.0, 0.49803922, 0.05490196, 1.0],  # Orange 4 
        [0.0, 0.4, 0.4, 1.0],  # Dark Teal 5
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
        [0.0, 0.74, 0.67, 1.0],  # Turquoise Blue 20
        [0.15, 0.2, 0.05, 1.0],  # Dark Olive 21
        [0.6627451, 0.8505867, 0.53165947, 1.0],  # Green 22
        [0.99215686, 0.70588235, 0.38431373, 1.0],  # Peach 23
        [0.07254902, 0.88292761, 0.9005867, 1.0],  # Light Blue 24
    ]

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':', (5, (10, 3)), (0, (5, 5)), (0, (5, 1))]

    def __init__(self, logs_dir='logs'):
        self.logs_dir = logs_dir
        self.exp_style_map = {}  # Maps exp_name -> (color, linestyle, marker)
        self.client_style_map = {}  # Maps client_name -> color
        self.color_idx = 0
        self.linestyle_idx = 0
        self.marker_idx = 0
        self.plot_dir = 'plots'
        os.makedirs(self.plot_dir, exist_ok=True)

    def get_exp_style(self, exp_name):
        """Get or create consistent style for an experiment across all plots."""
        if exp_name not in self.exp_style_map:
            color = self.colors[self.color_idx % len(self.colors)]
            linestyle = self.linestyles[self.linestyle_idx % len(self.linestyles)]
            marker = self.markers[self.marker_idx % len(self.markers)]

            self.exp_style_map[exp_name] = (color, linestyle, marker)

            self.color_idx += 1
            self.linestyle_idx += 1
            self.marker_idx += 1

        return self.exp_style_map[exp_name]

    def get_client_style(self, client_name, idx=None):
        """Get or create style for a client within a single plot."""
        if client_name not in self.client_style_map:
            if idx is not None:
                color = self.colors[idx % len(self.colors)]
            else:
                color = self.colors[len(self.client_style_map) % len(self.colors)]
            self.client_style_map[client_name] = color

        return self.client_style_map[client_name]

    def load_central_server_data(self, exp_name):
        """Load central server log for a given experiment."""
        log_path = os.path.join(self.logs_dir, exp_name, 'central', 'central_server.log')
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            return df
        return None

    def load_client_data(self, exp_name, data_type='test'):
        """
        Load all client logs for a given experiment and data type.
        data_type: 'test' or 'train'
        Returns a dict of {client_name: dataframe}
        """
        clients_dir = os.path.join(self.logs_dir, exp_name, 'clients')
        if not os.path.exists(clients_dir):
            return {}

        pattern = f'*_{data_type}.log'
        client_files = glob.glob(os.path.join(clients_dir, pattern))

        client_data = {}
        for client_file in client_files:
            client_name = os.path.basename(client_file).replace(f'_lenet_mnist_{data_type}.log', '')
            df = pd.read_csv(client_file)
            client_data[client_name] = df

        return client_data

    def load_edge_data(self):
        """
        Load all edge logs across all experiments.
        Returns a dict of {edge_name: {exp_name: dataframe}}
        """
        edge_data = {}
        exp_names = [d for d in os.listdir(self.logs_dir) 
                     if os.path.isdir(os.path.join(self.logs_dir, d))]

        for exp_name in sorted(exp_names):
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

    def plot_central_server_accuracy(self, output_file='central_accuracy_vs_rounds.png'):
        """Plot central server accuracy vs rounds for all experiments."""
        exp_names = [d for d in os.listdir(self.logs_dir)
                     if os.path.isdir(os.path.join(self.logs_dir, d))]

        plt.figure(figsize=(12, 6))

        for exp_name in sorted(exp_names):
            df = self.load_central_server_data(exp_name)
            if df is not None and 'round' in df.columns and 'accuracy' in df.columns:
                color, linestyle, marker = self.get_exp_style(exp_name)
                plt.plot(df['round'], df['accuracy'],
                        label=exp_name,
                        color=color,
                        #linestyle=linestyle,
                        linestyle='-',
                        linewidth=2,
                        markersize=5)

        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Central Server Accuracy vs Round (all clients selected)', fontsize=16)
        plt.grid(which='major', linestyle='-', linewidth=0.5)
        plt.grid(which='minor', linestyle='dotted', linewidth=0.2)
        plt.minorticks_on()
        plt.ylim(0, 1)
        plt.legend(loc='lower right', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, output_file), dpi=600)
        print(f"Saved central server plot to {output_file}")
        plt.close()

    def plot_client_accuracy(self, exp_name, data_type='test', output_file=None):
        """Plot client accuracy for a specific experiment."""
        if output_file is None:
            output_file = f'{exp_name}_clients_{data_type}_accuracy.png'

        client_data = self.load_client_data(exp_name, data_type)

        if not client_data:
            print(f"No {data_type} data found for {exp_name}")
            return

        plt.figure(figsize=(12, 6))

        for i, (client_name, df) in enumerate(sorted(client_data.items())):
            if 'round' in df.columns and 'accuracy' in df.columns:
                color = self.get_client_style(client_name, i)

                plt.plot(df['round'], df['accuracy'],
                        label=client_name,
                        color=color,
                        linestyle='-',
                        linewidth=1.5,
                        markersize=5)

        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.ylim(0, 1)
        plt.title(f'{exp_name} - Client\'s {data_type.capitalize()} Accuracy vs Round', fontsize=16)
        plt.grid(which='major', linestyle='-', linewidth=0.5)
        plt.grid(which='minor', linestyle='dotted', linewidth=0.2)
        plt.minorticks_on()
        plt.legend(loc='lower right', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, output_file), dpi=600)
        print(f"Saved client {data_type} plot to {output_file}")
        plt.close()

    def plot_all_clients_for_all_experiments(self, data_type='test'):
        """Plot client accuracy for all experiments."""
        exp_names = [d for d in os.listdir(self.logs_dir)
                     if os.path.isdir(os.path.join(self.logs_dir, d))]

        for exp_name in sorted(exp_names):
            self.plot_client_accuracy(exp_name, data_type)

    def plot_edge_accuracy_subplots(self, output_file='edge_accuracy_subplots.png'):
        """
        Plot edge accuracy vs rounds in subplots.
        Each subplot corresponds to one edge (e.g., Edge1, Edge2, etc.)
        Lines within each subplot represent different experiments with consistent linestyles.
        Y-axis labels displayed only on right edge subplots with constant scale across all subplots.
        """
        edge_data = self.load_edge_data()

        if not edge_data:
            print("No edge data found")
            return

        # Sort edge names for consistent ordering
        sorted_edges = sorted(edge_data.keys())

        # Calculate grid dimensions
        num_edges = len(sorted_edges)
        ncols = int(np.ceil(np.sqrt(num_edges)))
        nrows = int(np.ceil(num_edges / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = axes.flatten()  # Flatten to 1D for easier indexing

        # First pass: collect all accuracy data to determine y-axis limits
        all_accuracies = []
        for idx, edge_name in enumerate(sorted_edges):
            exp_dict = edge_data[edge_name]
            for exp_name in sorted(exp_dict.keys()):
                df = exp_dict[exp_name]
                if 'accuracy' in df.columns:
                    all_accuracies.extend(df['accuracy'].values)

        # Compute shared y-axis limits with some padding
        if all_accuracies:
            y_min = min(all_accuracies)
            y_max = max(all_accuracies)
            y_padding = (y_max - y_min) * 0.05
            y_min = max(0, y_min - y_padding)
            y_max = min(1, y_max + y_padding)
        else:
            y_min, y_max = 0, 1

        # Second pass: plot data with shared y-axis
        for idx, edge_name in enumerate(sorted_edges):
            ax = axes[idx]
            exp_dict = edge_data[edge_name]

            for exp_name in sorted(exp_dict.keys()):
                df = exp_dict[exp_name]
                if 'round' in df.columns and 'accuracy' in df.columns:
                    color, linestyle, marker = self.get_exp_style(exp_name)
                    ax.plot(df['round'], df['accuracy'],
                           label=exp_name,
                           color=color,
                           #linestyle=linestyle,
                           linestyle='-',
                           linewidth=1.5,
                           markersize=4)

            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'{edge_name}', fontsize=12, fontweight='bold')
            ax.grid(which='major', linestyle='-', linewidth=0.5)
            ax.grid(which='minor', linestyle='dotted', linewidth=0.2)
            ax.minorticks_on()
            ax.legend(loc='lower right', fontsize=10)

            # Remove y-axis labels from all subplots initially
            # ax.set_yticklabels([])

            # Only show y-axis labels and ylabel on left edge subplots
            is_left_edge = (idx % ncols == 0)
            if is_left_edge:
                ax.yaxis.tick_left()
                ax.yaxis.set_label_position('left')
                ax.set_ylabel('Accuracy', fontsize=11)
                # ax.set_yticklabels([f'{y:.1f}' for y in ax.get_yticks()])

        # Hide unused subplots
        for idx in range(num_edges, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Edge Accuracy vs Rounds Across Experiments (all clients selected)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, output_file), dpi=600, bbox_inches='tight')
        print(f"Saved edge accuracy subplots to {output_file}")
        plt.close()

    def load_distribution_data(self, exp_name, file_name):
        """Loads a distribution CSV file."""
        path = os.path.join(self.logs_dir, exp_name, file_name)
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    def plot_label_distributions(self):
        """
        Plots label distributions for clients and edges for all experiments.
        Generates:
        1. Stacked Bar Chart for Clients (Post-Clustering)
        2. Stacked Bar Chart for Edge Clusters
        3. Combined Comparison (Pre vs Post)
        """
        exp_names = [d for d in os.listdir(self.logs_dir) 
                     if os.path.isdir(os.path.join(self.logs_dir, d))]

        for exp_name in sorted(exp_names):
            print(f"Plotting distributions for {exp_name}...")
            
            # --- 1. Client Distribution (Post-Clustering) ---
            df_clients = self.load_distribution_data(exp_name, "distribution_post_clustering.csv")
            fig, ax = plt.subplots(figsize=(max(10, len(df_clients)*0.5), 6))
            if df_clients is not None:
                self._plot_stacked_bar(
                    ax,
                    df_clients, 
                    x_col="ClientName", 
                    title=f"{exp_name} - Client Label Distribution (Post-Clustering)",
                    sort_by="EdgeServer" # Group clients by their edge
                )
                plt.tight_layout()
                filename = f"{exp_name}_dist_clients.png"
                plt.savefig(os.path.join(self.plot_dir, filename), dpi=600)
                plt.close()
                print(f"  Saved {filename}")

            # --- 2. Edge Distribution ---
            df_edges = self.load_distribution_data(exp_name, "cluster_distribution.csv")
            fig, ax = plt.subplots(figsize=(max(10, len(df_clients)*0.5), 6))
            if df_edges is not None:
                self._plot_stacked_bar(
                    ax,
                    df_edges, 
                    x_col="ClusterID", 
                    title=f"{exp_name} - Edge Cluster Label Distribution",
                )
                plt.tight_layout()
                filename = f"{exp_name}_dist_edges.png"
                plt.savefig(os.path.join(self.plot_dir, filename), dpi=600)
                plt.close()
                print(f"  Saved {filename}")
                
            # --- 3. Combined Comparison (Pre vs Post) ---
            self.plot_clustering_comparison(exp_name)

    def _plot_stacked_bar(self, ax, df, x_col, title, sort_by=None):
        """Helper to create a stacked bar chart on a given axis."""
        
        # Filter only class columns
        class_cols = [c for c in df.columns if c.startswith("Class_")]
        class_cols.sort(key=lambda x: int(x.split('_')[1])) # Ensure numeric order
        
        if sort_by and sort_by in df.columns:
            df = df.sort_values(by=[sort_by, x_col])
        # else:
        #     df = df.sort_values(by=[x_col])
        
        # Normalize data to percentages (optional, but good for comparing distributions)
        # If you want raw counts, remove the division.
        data = df[class_cols].values
        row_sums = data.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1 
        data_pct = data / row_sums

        labels = df[x_col].tolist()
        
        # Create Plot
        # fig, ax = plt.subplots(figsize=(max(10, len(labels)*0.5), 6))
        
        # Plot stacked bars
        bottom = np.zeros(len(labels))
        # Use a distinct colormap for classes (e.g., tab10 or tab20)
        cmap = plt.get_cmap("tab10")
        
        for i, col_name in enumerate(class_cols):
            class_label = col_name.replace("Class_", "")
            values = data_pct[:, i]
            ax.bar(labels, values, bottom=bottom, label=f"Class {class_label}", 
                   color=cmap(i % 10), edgecolor='white', width=0.8)
            bottom += values

        ax.set_ylabel("Fraction of Samples")
        ax.set_xlabel(x_col)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        
        # Rotate x-labels if there are many
        plt.xticks(rotation=45, ha='right')
        
        # Legend outside
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")
        ax.legend(title="Classes", loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
        
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.plot_dir, filename), dpi=600)
        # plt.close()
        # print(f"  Saved {filename}")
    
    def plot_clustering_comparison(self, exp_name):
        """
        Plots a side-by-side comparison of client distributions before and after clustering.
        """
        pre_path = "distribution_pre_clustering.csv"
        post_path = "distribution_post_clustering.csv"
        
        df_pre = self.load_distribution_data(exp_name, pre_path)
        df_post = self.load_distribution_data(exp_name, post_path)
        
        if df_pre is not None and df_post is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), sharey=True)
            
            # Plot Pre: Sorted by Default Cluster (Original Edge) if available
            sort_pre = "DefaultClusterID" if "DefaultClusterID" in df_pre.columns else "ClientName"
            self._plot_stacked_bar(
                ax1, df_pre, 
                x_col="ClientName", 
                sort_by=sort_pre, 
                title="Pre-Clustering Distribution (Sorted by Initial Topology)"
            )
            
            # Plot Post: Sorted by New Cluster ID
            sort_post = "ClusterID" if "ClusterID" in df_post.columns else "ClientName"
            self._plot_stacked_bar(
                ax2, df_post, 
                x_col="ClientName", 
                sort_by=sort_post, 
                title="Post-Clustering Distribution (Sorted by Assigned Cluster)"
            )
            
            plt.suptitle(f"Clustering Impact - {exp_name}", fontsize=16)
            plt.tight_layout()
            filename = f"{exp_name}_clustering_comparison.png"
            plt.savefig(os.path.join(self.plot_dir, filename), dpi=600)
            plt.close()
            print(f"  Saved comparison plot {filename}")

# Usage example
if __name__ == '__main__':
    plotter = Plotter(logs_dir='logs')

    # 1. Plot central server accuracy for all experiments
    plotter.plot_central_server_accuracy()

    # 2. Plot test accuracy for clients in all experiments
    plotter.plot_all_clients_for_all_experiments(data_type='test')

    # 3. Plot train accuracy for clients in all experiments
    plotter.plot_all_clients_for_all_experiments(data_type='train')

    # 4. Plot edge accuracy in subplots
    plotter.plot_edge_accuracy_subplots()

    # 5. Plot label distributions for clients and edges
    plotter.plot_label_distributions()
