from collections import Counter, OrderedDict
import pickle
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr
from typing import Optional
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    PathologicalPartitioner,
)
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

print("Available devices:")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

from config import (
    NUM_CLIENTS,
    TRAINING_LEARNING_RATE,
    BATCH_SIZE,
    PARTITIONER,
    DIRICHLET_ALPHA,
    DATASET,
    DASHBOARD_SERVER_URL,
    NUM_CLASSES_PER_PARTITION,
    COMPRESSION_METHOD, 
    QUANTIZATION_BITS, 
    TOPK_RATIO
)


def load_datasets(partition_id: Optional[int] = None):

    if partition_id is None:
        num_partitions = 1
        pid = 0
    else:
        num_partitions = NUM_CLIENTS
        pid = partition_id

    if PARTITIONER == "iid":
        partitioner = IidPartitioner(
            num_partitions=num_partitions,
        )
    elif PARTITIONER == "dirichlet":
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=DIRICHLET_ALPHA,  # 0.9
            self_balancing=True,
        )
    else:
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            num_classes_per_partition=NUM_CLASSES_PER_PARTITION,
        )

    fds = FederatedDataset(dataset=DATASET, partitioners={"train": partitioner})
    partition = fds.load_partition(pid)
    # No need for validation
    partition_train_test = partition.train_test_split(test_size=0.1, seed=42)

    def apply_transforms(batch):
        imgs = batch.get("img", batch.get("image"))

        if "fashion_mnist" in DATASET:
            pytorch_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),  # Fashion_MNIST’s mean/std
                ]
            )
        elif "mnist" in DATASET:
            pytorch_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST’s mean/std
                ]
            )
        else:
            pytorch_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                    ),  # CIFAR10’s mean/std
                ]
            )
        batch["img"] = [pytorch_transforms(img) for img in imgs]
        if "image" in batch:
            del batch["image"]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    train_partition = partition.with_transform(apply_transforms) 
    trainloader = DataLoader(
        train_partition, batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader


def train(net, trainloader, optimizer: torch.optim.Adam, epochs: int, verbose=False):
    """Train the network on the training set."""
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    losses = []
    accuracies = []
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        losses.append(epoch_loss.item())
        accuracies.append(epoch_acc)
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    return losses, accuracies

def train_with_zi_yi(net, trainloader, optimizer, epochs, beta, zi, yi, verbose=False):
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    losses, accuracies = [], []

    # accumulator for average gradients
    grad_accumulator = {name: torch.zeros_like(p, device=DEVICE) 
                        for name, p in net.named_parameters() if p.requires_grad}
    num_batches = 0

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # --- apply gradient correction per-parameter ---
            with torch.no_grad():
                for name, p in net.named_parameters():
                    if p.grad is not None:
                        correction = beta * (zi.get(name, 0.0) + yi.get(name, 0.0))
                        p.grad += correction

            # --- accumulate gradients ---
            with torch.no_grad():
                for name, p in net.named_parameters():
                    if p.grad is not None:
                        grad_accumulator[name] += p.grad.clone()

            optimizer.step()
            num_batches += 1

            # metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

    # average gradients
    for name in grad_accumulator:
        grad_accumulator[name] /= num_batches

    return losses, accuracies, grad_accumulator

def train_fedprox_with_zi_yi(net, trainloader, optimizer: torch.optim.Adam, epochs: int, beta, zi, yi, verbose=False, mu=0.01):
    """Train the network on the training set."""
    net.to(DEVICE)
    global_params = [p.clone().detach() for p in net.parameters()]
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    losses = []
    accuracies = []
    # accumulator for average gradients
    grad_accumulator = {name: torch.zeros_like(p, device=DEVICE) 
                        for name, p in net.named_parameters() if p.requires_grad}
    
    num_batches = 0
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            # Fedprox
            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)**2
            
            loss += (mu / 2) * proximal_term

            loss.backward()
            # --- apply gradient correction per-parameter ---
            with torch.no_grad():
                for name, p in net.named_parameters():
                    if p.grad is not None:
                        correction = beta * (zi.get(name, 0.0) + yi.get(name, 0.0))
                        p.grad += correction

            # --- accumulate gradients ---
            with torch.no_grad():
                for name, p in net.named_parameters():
                    if p.grad is not None:
                        grad_accumulator[name] += p.grad.clone()

            optimizer.step()
            num_batches += 1
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        losses.append(epoch_loss.item())
        accuracies.append(epoch_acc)
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    # average gradients
    for name in grad_accumulator:
        grad_accumulator[name] /= num_batches

    del global_params

    return losses, accuracies, grad_accumulator

def train_fedprox(net, trainloader, optimizer: torch.optim.Adam, epochs: int, verbose=False, mu=0.01):
    """Train the network on the training set."""
    net.to(DEVICE)
    global_params = [p.clone().detach() for p in net.parameters()]
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    losses = []
    accuracies = []
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            # Fedprox
            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)**2
            
            loss += (mu / 2) * proximal_term

            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        losses.append(epoch_loss.item())
        accuracies.append(epoch_acc)
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    
    del global_params

    return losses, accuracies

def generate_mutated_models(current_weights, old_weights, num_variations, alpha=4.0):
    """
    Implements FedMut Algorithm 2 (Balanced Mutation).
    
    Ensures that for every layer, the sum of mutation directions across 
    all 'num_variations' models is exactly 0. This guarantees that 
    Agg(Mutated_Models) == Global_Model.
    """
    # 1. Calculate Gradients (Direction: Current - Old)
    gradients = [c - o for c, o in zip(current_weights, old_weights)]
    
    # 2. Prepare the Balanced Mutation Matrix
    # If K=4, we want directions [-1, -1, 1, 1]
    # If K=5, we keep one model original (0), and mutate 4: [-1, -1, 1, 1]
    
    num_pairs = num_variations // 2
    # Create the base list of directions for one layer: [1, 1, ..., -1, -1, ...]
    base_directions = [1.0] * num_pairs + [-1.0] * num_pairs
    
    # If odd, we will handle the extra client separately (keep it as original)
    
    mutated_models = [[] for _ in range(num_variations)]
    
    # 3. Apply Layer-wise Balanced Mutation
    for l_idx, (layer_w, layer_g) in enumerate(zip(current_weights, gradients)):
        
        # Shuffle directions specifically for this layer
        # This ensures diversity: Client A might get +1 on Layer 1 and -1 on Layer 2
        np.random.shuffle(base_directions)
        
        # Assign mutated weights to the paired clients
        for i in range(num_pairs * 2):
            direction = base_directions[i]
            
            # Formula: W_new = W + alpha * direction * Gradient
            mutation_term = (alpha * direction * layer_g)
            new_layer = (layer_w + mutation_term)
            
            mutated_models[i].append(new_layer)
            
        # Handle the odd client out (if any)
        # The paper says: "set w_K = w_glb when K%2=1"
        if num_variations % 2 != 0:
            mutated_models[-1].append(layer_w) # No mutation
            
    return mutated_models

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def get_gradient_shap_importance(model, data_loader, device):
    """
    Calculates importance based on |weight * gradient| (Gradient SHAP approximation).
    Used when COMPRESSION_METHOD == "shap".
    """
    model.eval()
    model.zero_grad()
    
    # We only need one batch to estimate the gradient sensitivity
    try:
        for i, batch in enumerate(data_loader):
            inputs, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            if i >= 1: break        
            # Forward pass
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # Backward pass to get gradients
            loss.backward()
        
            importance_scores = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Importance = |weight * gradient|
                    # Represents: "How much does setting this param to 0 change the loss?"
                    imp = torch.abs(param.data * param.grad.data)
                    importance_scores[name] = imp.cpu().numpy()
                else:
                    importance_scores[name] = torch.zeros_like(param.data).cpu().numpy()
                    
            return importance_scores
    except Exception as e:
        print(f"⚠️ Error calculating SHAP importance: {e}")
        return {}

def get_fisher_importance(model, data_loader, device, num_batches=1):
    """
    Calculates importance based on Diagonal Fisher Information.
    F_ii = E[ (grad_i)^2 ]
    """
    model.eval()
    fisher_info = {}
    
    # Initialize zero tensors
    for name, param in model.named_parameters():
        fisher_info[name] = torch.zeros_like(param.data)

    total_samples = 0
    
    # Accumulate squared gradients
    for i, batch in enumerate(data_loader):
        inputs, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
        if i >= num_batches: break
        
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Fisher = Sum of (grad^2)
                fisher_info[name] += torch.square(param.grad.data)
        
        total_samples += 1

    # Normalize
    final_scores = {}
    for name, val in fisher_info.items():
        final_scores[name] = (val / total_samples).cpu().numpy()
        
    return final_scores

def compress_model_update(model_diff: Dict[str, np.ndarray], importance_scores: Dict[str, np.ndarray] = None) -> Dict:
    """
    Compresses the difference (gradient/update/yi/zi).
    Supports: 'quantization', 'topk', 'shap', 'fisher'.
    """
    if COMPRESSION_METHOD == "none":
        return model_diff
        
    compressed_layers = []
    
    for name, layer in model_diff.items():
        layer_t = torch.tensor(layer)
        original_shape = layer_t.shape
        
        if COMPRESSION_METHOD == "quantization":
            min_val, max_val = layer_t.min(), layer_t.max()
            if max_val == min_val:
                scale = 1.0
            else:
                scale = (2**QUANTIZATION_BITS - 1) / (max_val - min_val)    
            q_layer = torch.round((layer_t - min_val) * scale).to(torch.uint8)
            
            compressed_layers.append({
                "name": name,
                "q": q_layer.numpy(), 
                "min": min_val.item(), 
                "max": max_val.item(), 
                "scale": scale.item() if isinstance(scale, torch.Tensor) else scale
            })

        elif COMPRESSION_METHOD == "topk":
            layer_flat = layer_t.flatten()
            k = max(1, int(layer_flat.numel() * TOPK_RATIO))
            vals, idxs = torch.topk(torch.abs(layer_flat), k)
            original_vals = layer_flat[idxs]
            
            compressed_layers.append({
                "name": name,
                "vals": original_vals.numpy(), 
                "idxs": idxs.numpy(), 
                "shape": original_shape
            })
        
        # SHAP and FISHER share the same logic: 
        # Use external importance scores to pick indices, but send actual gradient values.
        elif COMPRESSION_METHOD == "shap" or COMPRESSION_METHOD == "fisher":
            layer_flat = layer_t.flatten()
            k = max(1, int(layer_flat.numel() * TOPK_RATIO))
            
            # 1. Get Importance Scores
            if importance_scores and name in importance_scores:
                imp_flat = torch.tensor(importance_scores[name]).flatten()
            else:
                # Fallback to Magnitude if importance_scores is missing
                imp_flat = torch.abs(layer_flat)
            
            # 2. Select Top K indices based on IMPORTANCE
            _, idxs = torch.topk(imp_flat, k)
            
            # 3. Retrieve ACTUAL values
            original_vals = layer_flat[idxs]
            
            compressed_layers.append({
                "name": name,
                "vals": original_vals.numpy(), 
                "idxs": idxs.numpy(), 
                "shape": original_shape
            })
    
    return {"method": COMPRESSION_METHOD, "layers": compressed_layers}
       
def decompress_model_update(compressed_payload: Union[Dict, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Reconstructs the dictionary {layer_name: gradient}.
    """
    if not isinstance(compressed_payload, dict) or "method" not in compressed_payload:
        return compressed_payload
        
    method = compressed_payload.get("method", "none")
    decompressed_dict = {}

    if method == "quantization":
        for layer_data in compressed_payload["layers"]:
            name = layer_data["name"]
            q = torch.tensor(layer_data["q"]).float()
            rec = (q / layer_data["scale"]) + layer_data["min"]
            decompressed_dict[name] = rec.numpy()
            
    elif method in ["topk", "shap", "fisher"]:
        # All sparse methods use the same reconstruction logic
        for layer_data in compressed_payload["layers"]:
            name = layer_data["name"]
            shape = layer_data["shape"]
            
            flat = torch.zeros(int(np.prod(shape)))
            flat[layer_data["idxs"]] = torch.tensor(layer_data["vals"]).float()
            
            decompressed_dict[name] = flat.reshape(shape).numpy()

    return decompressed_dict

def get_payload_size(payload):
    """
    Calculates the approximate network size of the payload in bytes.
    Supports Lists, Raw Dictionaries, and Compressed Dictionaries.
    """
    total_bytes = 0
    
    # Case 1: Uncompressed List (Legacy / Fallback)
    if isinstance(payload, list):
        for layer in payload:
            if isinstance(layer, np.ndarray):
                total_bytes += layer.nbytes
            elif isinstance(layer, list):
                # [FIX] Convert list to numpy to get the TRUE uncompressed data size
                # sys.getsizeof(list) only counts pointers (~8KB), ignoring the data (~240KB)
                try:
                    total_bytes += np.array(layer).nbytes
                except:
                    total_bytes += sys.getsizeof(layer)
            else:
                total_bytes += sys.getsizeof(layer)
            
    # Case 2: Dictionary (Could be Raw {name: array} OR Compressed structure)
    elif isinstance(payload, dict):
        # Check if it's our Compressed Structure (has "method" and "layers")
        if "method" in payload and "layers" in payload:
            # Compressed Dictionary
            method = payload["method"]
            
            for layer in payload["layers"]:
                # 1. Count the Layer Name string size (approx)
                if "name" in layer:
                    total_bytes += len(layer["name"].encode('utf-8'))

                # 2. Count Data based on method
                if method == "quantization":
                    total_bytes += layer["q"].nbytes
                    # Metadata overhead
                    for key in ["min", "max", "scale"]:
                        val = layer[key]
                        if isinstance(val, (np.generic, np.ndarray)):
                            total_bytes += val.nbytes
                        else:
                            total_bytes += 8 # Python float
                            
                elif method in ["topk", "shap", "fisher"]:
                    total_bytes += layer["vals"].nbytes
                    total_bytes += layer["idxs"].nbytes
                    # Shape tuple (approx 8 bytes per dim)
                    total_bytes += len(layer["shape"]) * 8 
        
        # Case 3: Raw Dictionary { "layer_name": np.ndarray } (Before Compression)
        else:
            # Uncompressed Dictionary
            for name, layer in payload.items():
                # Count Name size
                total_bytes += len(name.encode('utf-8'))
                
                # Count Array size
                if isinstance(layer, np.ndarray):
                    total_bytes += layer.nbytes
                elif isinstance(layer, list):
                    # [FIX] Handle Lists in Dicts
                    try:
                        total_bytes += np.array(layer).nbytes
                    except:
                        total_bytes += sys.getsizeof(layer)
                else:
                    total_bytes += sys.getsizeof(layer)

    # Case 4: Raw Numpy Array (The compressed blob from pack_compressed_data)
    elif isinstance(payload, np.ndarray):
        total_bytes = payload.nbytes
        
    # Case 5: Raw Bytes (Pickled streams for Yi/Zi)
    elif isinstance(payload, bytes):
        total_bytes = len(payload)

    # mb_size = total_bytes / (1024**2) # converting to MB in get_traffic_metrics
    # # Optional: Only print if significant to reduce log spam
    # if mb_size > 0.0001:
    #     print(f"Payload Size: {total_bytes} bytes ({mb_size:.4f} MB)")
    
    return total_bytes

def pack_compressed_data(compressed_dict: dict) -> np.ndarray:
    """Serializes compressed dict to a uint8 numpy array for transport."""
    serialized = pickle.dumps(compressed_dict)
    # Convert bytes to uint8 array
    return np.frombuffer(serialized, dtype=np.uint8)

def unpack_compressed_data(packed_array: Union[np.ndarray, bytes]) -> dict:
    """Deserializes uint8 numpy array OR raw bytes back to compressed dict."""
    if isinstance(packed_array, bytes):
        serialized = packed_array
    else:
        # It's a numpy array (uint8), so we extract bytes
        serialized = packed_array.tobytes()
        
    return pickle.loads(serialized)

def get_traffic_metrics(round_num, direction, model_tuple=(0,0), yi_tuple=(0,0), zi_tuple=(0,0), grad_tuple=(0,0), comp_time=0.0, decomp_time=0.0):
    """
    Returns a dictionary for Logger including sizes (MB) and timings (seconds).
    """
    m_u, m_c = model_tuple
    y_u, y_c = yi_tuple
    z_u, z_c = zi_tuple
    g_u, g_c = grad_tuple

    # Calculate Totals
    total_u = m_u + y_u + z_u + g_u
    total_c = m_c + y_c + z_c + g_c

    def to_mb(val): return float(f"{val / (1024**2):.7f}")

    return {
        "Round": round_num,
        "Direction": direction,
        "model_wts_MB": to_mb(m_u),
        "compressed_model_wts_MB": to_mb(m_c),
        "Y_i_MB": to_mb(y_u),
        "compressed_Y_i_MB": to_mb(y_c),
        "Z_i_MB": to_mb(z_u),
        "compressed_Z_i_MB": to_mb(z_c),
        "Total_MB": to_mb(total_u),
        "Compressed_Total_MB": to_mb(total_c),
        "compression_time_s": float(f"{comp_time:.5f}"),
        "decompression_time_s": float(f"{decomp_time:.5f}")
    }

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    # torch.Tensor -> float and torch.tensor -> int or long; preserves type so that num_batches_tracked BatchNorm could work 
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def get_dataloader_summary(dataloader):
    dataset = dataloader.dataset

    # pull labels out correctly
    labels = []
    for sample in dataset:
        # if it's a dict with a "label" key
        if isinstance(sample, dict) and "label" in sample:
            labels.append(sample["label"])
        # else if it’s a tuple (x, y)
        else:
            _, y = sample
            labels.append(y)

    num_items = len(labels)
    counts = Counter(labels)
    # dist = {cls: cnt / num_items for cls, cnt in counts.items()}

    # print("raw counts:", dict(counts))
    return {"label_distribution": counts, "num_items": num_items}


def post_to_dashboard(url: str, payload: Dict):
    try:
        res = requests.post(
            url, json=payload, headers={"ngrok-skip-browser-warning": "true"}
        )
        if res.status_code != 200:
            print(f"Error posting to {url}: {res.text}")
    except Exception as e:
        print(f"Failed to post to {url}: {e}")


def log_to_dashboard(exp_id: str, role: str, payload: Dict):
    url = f"{DASHBOARD_SERVER_URL}/experiment/{exp_id}/log/{role}"
    post_to_dashboard(url, payload)


if __name__ == "__main__":
    # Example usage
    trainloader, valloader, testloader = load_datasets()
    print(get_dataloader_summary(trainloader))
    print(get_dataloader_summary(valloader))
    print(get_dataloader_summary(testloader))
