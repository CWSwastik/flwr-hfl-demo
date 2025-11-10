NUM_ROUNDS = 100
TOPOLOGY_FILE = "topo-16clients.yml"

NUM_CLIENTS = 16
MIN_CLIENTS_PER_EDGE = 3

MODEL = "lenet_mnist"  # vgg, lenet_mnist, lenet_cifar10
DATASET = "mnist"  # mnist, cifar10, fashion_mnist

BATCH_SIZE = 16
PARTITIONER = "iid"  # "iid" or "dirichlet" or "pathological"
DIRICHLET_ALPHA = 0.1
NUM_CLASSES_PER_PARTITION = 3  # used in pathological partitioning (limit label)

GRADIENT_CORRECTION_BETA = 1

TRAINING_LEARNING_RATE = 5 * 1e-4
TRAINING_WEIGHT_DECAY = 1e-4

TRAINING_SCHEDULER_STEP_SIZE = 10
TRAINING_SCHEDULER_GAMMA = 0.1

DASHBOARD_SERVER_URL = "https://f772957a48fe.ngrok-free.app"
SPLIT=PARTITIONER
if PARTITIONER == 'dirichlet':
    SPLIT=f"{SPLIT}_{DIRICHLET_ALPHA}"
elif PARTITIONER == 'pathological':
    SPLIT=f"{SPLIT}_{NUM_CLASSES_PER_PARTITION}"

EXPERIMENT_NAME = f"{DATASET}-{NUM_CLIENTS}c-{SPLIT}"
