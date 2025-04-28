import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


# Helper to plot loss vs round and accuracy vs round
def plot_loss_accuracy(ax_loss, ax_acc, df, label):
    ax_loss.plot(df["round"], df["loss"], marker="o", label=label)
    ax_acc.plot(df["round"], df["accuracy"], marker="o", label=label)


# Set up figure
fig, axs = plt.subplots(
    3, 2, figsize=(16, 18)
)  # 3 rows (Central, Edge, Client) × 2 columns (Loss and Accuracy)
fig.tight_layout(pad=5.0)

# Plot for Central Server
current_dir = os.path.dirname(os.path.abspath(__file__))
central_log = os.path.join(current_dir, "logs", "central", "central_server.log")
central_df = pd.read_csv(central_log)
axs[0][0].set_title("Central Server: Loss vs Round")
axs[0][1].set_title("Central Server: Accuracy vs Round")
plot_loss_accuracy(axs[0][0], axs[0][1], central_df, "Central Server")
axs[0][0].set_xlabel("Round")
axs[0][0].set_ylabel("Loss")
axs[0][0].grid(True)
axs[0][0].legend()
axs[0][1].set_xlabel("Round")
axs[0][1].set_ylabel("Accuracy")
axs[0][1].grid(True)
axs[0][1].legend()

# Plot for Edges
edge_logs = glob.glob(os.path.join(current_dir, "logs", "edge", "*.log"))
axs[1][0].set_title("Edges: Loss vs Round")
axs[1][1].set_title("Edges: Accuracy vs Round")
for log_file in edge_logs:
    name = os.path.basename(log_file).replace(".log", "")
    df = pd.read_csv(log_file)
    plot_loss_accuracy(axs[1][0], axs[1][1], df, name)
axs[1][0].set_xlabel("Round")
axs[1][0].set_ylabel("Loss")
axs[1][0].grid(True)
axs[1][0].legend()
axs[1][1].set_xlabel("Round")
axs[1][1].set_ylabel("Accuracy")
axs[1][1].grid(True)
axs[1][1].legend()

# Plot for Clients
client_logs = glob.glob(os.path.join(current_dir, "logs", "clients", "*.log"))
clients = {}

# Group client train and test files
for log_file in client_logs:
    base = os.path.basename(log_file)
    client_name = "_".join(base.split("_")[:1])  # e.g., Client1
    if client_name not in clients:
        clients[client_name] = {}
    if "train" in base:
        clients[client_name]["train"] = log_file
    else:
        clients[client_name]["test"] = log_file

axs[2][0].set_title("Clients: Loss vs Round")
axs[2][1].set_title("Clients: Accuracy vs Round")

# Now plot both train and test for each client
for client, paths in clients.items():
    if "train" in paths:
        train_df = pd.read_csv(paths["train"])
        plot_loss_accuracy(axs[2][0], axs[2][1], train_df, f"{client} Train")
    if "test" in paths:
        test_df = pd.read_csv(paths["test"])
        plot_loss_accuracy(axs[2][0], axs[2][1], test_df, f"{client} Test")

axs[2][0].set_xlabel("Round")
axs[2][0].set_ylabel("Loss")
axs[2][0].grid(True)
axs[2][0].legend()
axs[2][1].set_xlabel("Round")
axs[2][1].set_ylabel("Accuracy")
axs[2][1].grid(True)
axs[2][1].legend()

# Show all plots
# plt.show()
plt.savefig("logs/plots.png")
