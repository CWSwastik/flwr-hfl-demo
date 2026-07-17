import os
import json

# --- CONTROL FLAGS ---
# DEBUG = True  : Runs a single experiment (Legacy mode). Saves logs directly to 'logs/experiment_name'.
#                 Ignores 'FL_RUN_ID' and uses BASE_SEED.
# DEBUG = False : Runs in Batch Mode. Saves logs to 'logs/run_X/experiment_name'.
#                 Expects 'FL_RUN_ID' from run_experiments.py to set dynamic seeds.
DEBUG = True

NUM_ROUNDS = 100
TOPOLOGY_FILE = "topo-100c.yml"

NUM_CLIENTS = 100
MIN_CLIENTS_PER_EDGE = 10

MODEL = "lenet_mnist"
DATASET = "mnist"

# Debug/run 1 uses BASE_SEED; batch run r uses BASE_SEED + (r - 1)
BASE_SEED = 42

LOCAL_EPOCHS = 1
BATCH_SIZE = 16
PARTITIONER = "dirichlet"
DIRICHLET_ALPHA = 0.1
NUM_CLASSES_PER_PARTITION = 3  # used in pathological partitioning (limit label)
NUM_CLASSES = 10  # total number of classes in the dataset

CLUSTER_STRATEGY = "none"
DISSIMILAR_CLUSTERING = False

GRADIENT_CORRECTION_BETA = 0

TRAINING_LEARNING_RATE = 0.0005
TRAINING_WEIGHT_DECAY = 1e-4

TRAINING_SCHEDULER_STEP_SIZE = 10
TRAINING_SCHEDULER_GAMMA = 0.1
TRAINING_STRATEGY = "fedavg"
FedProx_MU = 0.01

# Options: "none", "quantization", "topk", "shap", "fisher"
COMPRESSION_METHOD = "none"

# Fine-grained control: Only compress these specific control variates?
COMPRESS_YI = False
COMPRESS_ZI = False

# Compression Hyperparameters
QUANTIZATION_BITS = 8
TOPK_RATIO = 0.1

# FedMut Configuration
# 0 = Off, 1 = On
FEDMUT_CENTRAL = 0
FEDMUT_EDGE = 0
FEDMUT_ALPHA = 4

DASHBOARD_SERVER_URL = "https://f772957a48fe.ngrok-free.app"
ENABLE_DASHBOARD = False

# --- PER-JOB OVERRIDES ---
# run_experiments.py writes each experiment's dict to a job-private JSON file
# and points FL_CONFIG_OVERRIDES at it. The env var is inherited by every
# spawned process (simulate/edge/client/server/monitor), so parallel jobs on
# one machine each see their own config without ever rewriting this file.
_overrides = {}
_override_file = os.environ.get("FL_CONFIG_OVERRIDES")
if _override_file and os.path.exists(_override_file):
    with open(_override_file) as f:
        _overrides = json.load(f)
    globals().update(_overrides)
    print(f"[Config] 📦 Applied {len(_overrides)} overrides from {_override_file}")

# --- DYNAMIC CONFIGURATION LOGIC ---
if DEBUG:
    # [DEBUG MODE] Single Execution, Standard Path
    RUN = 0
    SEED = BASE_SEED

    # No folder prefix -> saves to logs/dataset-model...
    PATH_PREFIX = ""

    print(f"[Config] 🟡 DEBUG MODE: Single Run | Seed: {SEED}")

else:
    # [BATCH MODE] Multi-Run, Isolated Folders
    # run_experiments.py sets this environment variable
    if "FL_RUN_ID" in os.environ:
        RUN = int(os.environ["FL_RUN_ID"])
    else:
        # Default fallback if running simulate.py manually with DEBUG=False
        RUN = 1

    # Dynamic Seed Calculation: Run 1->BASE_SEED, Run 2->BASE_SEED+1, etc.
    SEED = BASE_SEED + (RUN - 1)

    # Path Prefix -> saves to logs/run_1/dataset-model...
    PATH_PREFIX = f"run_{RUN}/"

    print(f"[Config] 🟢 BATCH MODE: Run {RUN} | Seed {SEED} | Path: logs/{PATH_PREFIX}...")

# An explicit SEED in the overrides pins every run to that seed
if "SEED" in _overrides:
    SEED = _overrides["SEED"]

# -----------------------------------

SPLIT = PARTITIONER

if PARTITIONER == "dirichlet":
    SPLIT = f"{SPLIT}_{DIRICHLET_ALPHA}"
elif PARTITIONER == "pathological":
    SPLIT = f"{SPLIT}_{NUM_CLASSES_PER_PARTITION}"

fedmut_type = ""
if FEDMUT_CENTRAL and FEDMUT_EDGE and TRAINING_STRATEGY == "fedmut":
    fedmut_type = "_Central&Edge"
elif FEDMUT_CENTRAL and TRAINING_STRATEGY == "fedmut":
    fedmut_type = "_Central"
elif FEDMUT_EDGE and TRAINING_STRATEGY == "fedmut":
    fedmut_type = "_Edge"

compression_log = ""
compress_end= ""
if COMPRESSION_METHOD != "none":
    if COMPRESS_YI and COMPRESS_ZI:
        compress_end = "_YiZi_"
    elif COMPRESS_YI:
        compress_end = "_Yi_"
    elif COMPRESS_ZI:
        compress_end = "_Zi_"
    if COMPRESSION_METHOD == "quantization":
        compression_log += f"-compress_quant_{QUANTIZATION_BITS}bit{compress_end}"
    elif COMPRESSION_METHOD == "topk":
        compression_log += f"-compress_topk_{int(TOPK_RATIO*100)}pct{compress_end}"
    elif COMPRESSION_METHOD == "shap":
        compression_log += f"-compress_shap_{int(TOPK_RATIO*100)}pct{compress_end}"
    elif COMPRESSION_METHOD == "fisher":
        compression_log += f"-compress_fisher_{int(TOPK_RATIO*100)}pct{compress_end}"

SPLIT = f"{SPLIT}-cluster_{CLUSTER_STRATEGY}-{TRAINING_STRATEGY}{fedmut_type}{'-dissimilar_cluster' if DISSIMILAR_CLUSTERING else ''}{compression_log}"

if GRADIENT_CORRECTION_BETA == 1:
    SPLIT = f"{SPLIT}-gc"

# Apply the prefix (either "run_X/" or empty string)
EXPERIMENT_NAME = f"{PATH_PREFIX}{DATASET}-{NUM_CLIENTS}c-{MODEL}-{SPLIT}"
