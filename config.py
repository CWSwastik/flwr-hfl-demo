NUM_ROUNDS = 2
TOPOLOGY_FILE = "topo-20c.yml"

NUM_CLIENTS = 20
MIN_CLIENTS_PER_EDGE = 10

MODEL = "lenet_mnist"
DATASET = "mnist"

BATCH_SIZE = 16
PARTITIONER = "dirichlet" # options: "iid", "dirichlet", "pathological"
DIRICHLET_ALPHA = 0.1
NUM_CLASSES_PER_PARTITION = 3  # used in pathological partitioning (limit label)
NUM_CLASSES = 10  # total number of classes in the dataset

CLUSTER_STRATEGY = "none" # options: "none", "emd", "jsd"

GRADIENT_CORRECTION_BETA = 1

TRAINING_LEARNING_RATE = 5 * 1e-4
TRAINING_WEIGHT_DECAY = 1e-4

TRAINING_SCHEDULER_STEP_SIZE = 10
TRAINING_SCHEDULER_GAMMA = 0.1

DASHBOARD_SERVER_URL = "https://f772957a48fe.ngrok-free.app"
ENABLE_DASHBOARD = False
SPLIT=PARTITIONER

if PARTITIONER == "dirichlet":
    SPLIT=f"{SPLIT}_{DIRICHLET_ALPHA}"
elif PARTITIONER == "pathological":
    SPLIT=f"{SPLIT}_{NUM_CLASSES_PER_PARTITION}"

SPLIT=f"{SPLIT}-cluster_{CLUSTER_STRATEGY}"

if GRADIENT_CORRECTION_BETA == 1:
    SPLIT=f"{SPLIT}-gc"

EXPERIMENT_NAME = f"{DATASET}-{NUM_CLIENTS}c-{MODEL}-{SPLIT}"
