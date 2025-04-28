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

os.makedirs(os.path.join(current_dir, "plots"), exist_ok=True)
plt.savefig(os.path.join(current_dir, "plots", "accuracy_loss_vs_rounds.png"))


### Load dataset and display partitions

from config import NUM_CLIENTS
import json


# Load data and prepare summaries
summaries = []
for client_id in range(NUM_CLIENTS):
    client_name = f"Client{client_id+1}"
    # Match file pattern: logs/clients/{client_name}_{partition_id}_data_dist.json
    pattern = os.path.join(
        current_dir, "logs", "clients", f"{client_name}_*_data_dist.json"
    )
    matching_files = glob.glob(pattern)
    if not matching_files:
        continue  # or handle missing file

    # Assuming one matching file per client
    file_path = matching_files[0]
    with open(file_path, "r") as f:
        data = json.load(f)

    train_summary = data["trainloader"]
    val_summary = data["valloader"]
    test_summary = data["testloader"]

    summaries.append((client_name, train_summary, val_summary, test_summary))

# Set up one big figure
fig, axs = plt.subplots(NUM_CLIENTS, 3, figsize=(15, 5 * NUM_CLIENTS))
fig.tight_layout(pad=5.0)

# If only 1 client, axs is not 2D; fix it
if NUM_CLIENTS == 1:
    axs = axs.reshape(1, 3)

for idx, (client_id, train_summary, val_summary, test_summary) in enumerate(summaries):
    # Train
    ax = axs[idx, 0]
    train_dist = train_summary["label_distribution"]
    sorted_train = sorted(train_dist.items(), key=lambda x: int(x[0]))
    labels, counts = zip(*sorted_train)
    ax.bar(labels, counts)
    ax.set_title(f"{client_id} Train Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Size")

    # Validation
    ax = axs[idx, 1]
    val_dist = val_summary["label_distribution"]
    sorted_val = sorted(val_dist.items(), key=lambda x: int(x[0]))
    labels, counts = zip(*sorted_val)
    ax.bar(labels, counts, color="orange")
    ax.set_title(f"{client_id} Validation Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Size")

    # Test
    ax = axs[idx, 2]
    test_dist = test_summary["label_distribution"]
    sorted_test = sorted(test_dist.items(), key=lambda x: int(x[0]))
    labels, counts = zip(*sorted_test)
    ax.bar(labels, counts, color="green")
    ax.set_title(f"{client_id} Test Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Size")

# Adjust layout
plt.tight_layout()

output_path = os.path.join(current_dir, "plots", "data_distribution.png")
plt.savefig(output_path)
plt.close()
