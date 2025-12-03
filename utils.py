from collections import Counter, OrderedDict
from typing import Dict, List, Tuple

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
