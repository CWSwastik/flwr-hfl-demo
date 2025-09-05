NUM_ROUNDS = 100
TOPOLOGY_FILE = "topo-10c.yml"

NUM_CLIENTS = 10
MIN_CLIENTS_PER_EDGE = 3

MODEL = "lenet_mnist"  # vgg, lenet_mnist, lenet_cifar10
DATASET = "fashion_mnist"  # mnist, cifar10, fashion_mnist

BATCH_SIZE = 32
PARTITIONER = "pathological"  # "iid" or "dirichlet" or "pathological"
DIRICHLET_ALPHA = 0.5
NUM_CLASSES_PER_PARTITION = 3  # used in pathological partitioning (limit label)

TRAINING_LEARNING_RATE = 5 * 1e-4
TRAINING_WEIGHT_DECAY = 1e-4

TRAINING_SCHEDULER_STEP_SIZE = 10
TRAINING_SCHEDULER_GAMMA = 0.1

DASHBOARD_SERVER_URL = "https://f772957a48fe.ngrok-free.app"
