import flwr as fl
import numpy as np
import time
import argparse

from utils import (
    load_datasets,
    get_parameters,
    set_parameters,
    train,
    test,
    DEVICE,
    get_dataset_summary,
)
from config import NUM_ROUNDS

import importlib
from logger import Logger
import os
import sys, traceback


parser = argparse.ArgumentParser(description="Start a Flower client.")
parser.add_argument(
    "server_address",
    help="Server address in the format host:port (e.g., localhost:8081)",
)
parser.add_argument("--partition_id", type=int, default=0, help="Partition ID")
parser.add_argument(
    "--model", type=str, default="lenet", help="Model name (default: lenet)"
)
parser.add_argument(
    "--name", type=str, default="client", help="Client name (default: client)"
)
args = parser.parse_args()

test_logger = Logger(
    subfolder="clients",
    file_path=f"{args.name}_{args.model}_test.log",
    headers=["round", "loss", "accuracy", "data_samples"],
)

train_logger = Logger(
    subfolder="clients",
    file_path=f"{args.name}_{args.model}_train.log",
    headers=["round", "loss", "accuracy", "data_samples"],
)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.round = 0

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):

        if not np.all(parameters[0] == 0):
            print(f"Received new global model from server")
            set_parameters(self.net, parameters)
        else:
            print("Received initial model from server, starting training...")

        losses, accuracies = train(self.net, self.trainloader, epochs=1)
        train_logger.log(
            {
                "round": self.round,
                "loss": losses[0],
                "accuracy": accuracies[0],
                "data_samples": len(self.trainloader.dataset),
            }
        )
        # print(get_parameters(self.net)[0][0][0][0])
        return get_parameters(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        test_logger.log(
            {
                "round": self.round,
                "loss": loss,
                "accuracy": accuracy,
                "data_samples": len(self.valloader.dataset),
            }
        )
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


def create_client(partition_id, model) -> fl.client.Client:

    model_module = importlib.import_module(f"models.{model}")
    net = model_module.Net().to(DEVICE)

    trainloader, valloader, _ = load_datasets(partition_id=partition_id)
    print("Trainloader size:", len(trainloader.dataset))
    print("Valloader size:", len(valloader.dataset))
    print("Trainloader summary:", get_dataset_summary(trainloader))
    print("Valloader summary:", get_dataset_summary(valloader))

    return FlowerClient(net, trainloader, valloader)


if __name__ == "__main__":
    print(
        f"Starting client {args.name} with partition_id {args.partition_id} and connecting to {args.server_address}"
    )
    client = create_client(args.partition_id, model=args.model)
    while client.round <= NUM_ROUNDS:
        try:
            print(f"Starting client {args.name} for Round {client.round}")
            fl.client.start_client(
                server_address=args.server_address, client=client.to_client()
            )
            client.round += 1
        except Exception as e:
            # traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
            print(f"Error: {type(e)}, Couldn't run client. Retrying in 5 seconds...")

        time.sleep(5)
