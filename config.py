NUM_ROUNDS = 100
TOPOLOGY_FILE = "topo-20c.yml"

NUM_CLIENTS = 20
MIN_CLIENTS_PER_EDGE = 10

MODEL = "lenet_mnist"
DATASET = "mnist"

SEED = 42

LOCAL_EPOCHS = 1
BATCH_SIZE = 16
PARTITIONER = "iid"
DIRICHLET_ALPHA = 0.1
NUM_CLASSES_PER_PARTITION = 3  # used in pathological partitioning (limit label)
NUM_CLASSES = 10  # total number of classes in the dataset

CLUSTER_STRATEGY = "none" # options: "emd", "jsd", "cosine", "euclidean", "manhattan", "gmm", "kmeans", "mahalanobis", "none"
DISSIMILAR_CLUSTERING = True  # if True, clients in the same cluster are dissimilar

GRADIENT_CORRECTION_BETA = 0

TRAINING_LEARNING_RATE = 5 * 1e-4
TRAINING_WEIGHT_DECAY = 1e-4

TRAINING_SCHEDULER_STEP_SIZE = 10
TRAINING_SCHEDULER_GAMMA = 0.1
TRAINING_STRATEGY = "fedavg"  # options: "fedavg", "fedprox", "fedmut"
FedProx_MU = 0.01
# FedMut Configuration
# 0 = Off, 1 = On
FEDMUT_CENTRAL = 0  # Mutate Global Model before sending to Edges
FEDMUT_EDGE = 1     # Mutate Edge Model before sending to Clients
FEDMUT_ALPHA = 4.0  # Mutation strength (hyperparameter from paper)

DASHBOARD_SERVER_URL = "https://f772957a48fe.ngrok-free.app"
ENABLE_DASHBOARD = False
SPLIT=PARTITIONER

if PARTITIONER == "dirichlet":
    SPLIT=f"{SPLIT}_{DIRICHLET_ALPHA}"
elif PARTITIONER == "pathological":
    SPLIT=f"{SPLIT}_{NUM_CLASSES_PER_PARTITION}"

fedmut_type = ""
if FEDMUT_CENTRAL and FEDMUT_EDGE and TRAINING_STRATEGY == "fedmut":
    fedmut_type = "_Central&Edge"
elif FEDMUT_CENTRAL and TRAINING_STRATEGY == "fedmut":
    fedmut_type = "_Central"
elif FEDMUT_EDGE and TRAINING_STRATEGY == "fedmut":
    fedmut_type = "_Edge"

SPLIT=f"{SPLIT}-cluster_{CLUSTER_STRATEGY}-{TRAINING_STRATEGY}{fedmut_type}{"-dissimilar_cluster" if DISSIMILAR_CLUSTERING else ""}"

if GRADIENT_CORRECTION_BETA == 1:
    SPLIT=f"{SPLIT}-gc"

EXPERIMENT_NAME = f"{DATASET}-{NUM_CLIENTS}c-{MODEL}-{SPLIT}"
