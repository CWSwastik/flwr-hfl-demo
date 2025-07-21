NUM_ROUNDS = 100
TOPOLOGY_FILE = "topo.yml"
NUM_CLIENTS = 4

MODEL = "lenet_mnist"  # vgg, lenet_mnist, lenet_cifar10
DATASET = "fashion_mnist"  # mnist, cifar10, fashion_mnist

BATCH_SIZE = 32
PARTITIONER = "iid"  # "iid" or "dirichlet"
DIRICHLET_ALPHA = 0.5

TRAINING_LEARNING_RATE = 5 * 1e-4
TRAINING_WEIGHT_DECAY = 1e-4

TRAINING_SCHEDULER_STEP_SIZE = 10
TRAINING_SCHEDULER_GAMMA = 0.1

DASHBOARD_SERVER_URL = "https://f64e2d22cc67.ngrok-free.app"
