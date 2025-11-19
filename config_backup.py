NUM_ROUNDS = 100
TOPOLOGY_FILE = "topo.yml"

NUM_CLIENTS = 4
MIN_CLIENTS_PER_EDGE = 2

MODEL = "lenet_mnist"
DATASET = "mnist"

BATCH_SIZE = 16
PARTITIONER = "dirichlet"  # "iid" or "dirichlet" or "pathological"
DIRICHLET_ALPHA = 0.1
NUM_CLASSES_PER_PARTITION = 3  # used in pathological partitioning (limit label)
NUM_CLASSES = 10  # total number of classes in the dataset

CLUSTER_STRATEGY="emd"  # Options: "emd", "jsd", none # jsd is Jensen-Shannon Divergence

GRADIENT_CORRECTION_BETA = 1

TRAINING_LEARNING_RATE = 5 * 1e-4
TRAINING_WEIGHT_DECAY = 1e-4

TRAINING_SCHEDULER_STEP_SIZE = 10
TRAINING_SCHEDULER_GAMMA = 0.1

DASHBOARD_SERVER_URL = "https://f772957a48fe.ngrok-free.app"
SPLIT=PARTITIONER
if PARTITIONER == "dirichlet":
    SPLIT=f"{SPLIT}_{DIRICHLET_ALPHA}"
elif PARTITIONER == "pathological":
    SPLIT=f"{SPLIT}_{NUM_CLASSES_PER_PARTITION}"

if CLUSTER_STRATEGY != "none":
    SPLIT=f"{SPLIT}-{CLUSTER_STRATEGY}"

if GRADIENT_CORRECTION_BETA == 1:
    SPLIT=f"{SPLIT}-gc"

EXPERIMENT_NAME = f"{DATASET}-{NUM_CLIENTS}c-{SPLIT}"
