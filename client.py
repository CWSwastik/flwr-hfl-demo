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
)
from config import NUM_ROUNDS

import importlib
from logger import Logger
import os


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

logger = Logger(
    subfolder="clients",
    file_path=f"{args.name}_{args.partition_id}_{args.model}.log",
    headers=["round", "loss", "accuracy", "data_samples"],
)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        if not np.all(parameters[0] == 0):
            print(f"Received new global model from server")
            set_parameters(self.net, parameters)
        else:
            print("Received initial model from server, starting training...")

        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        global rounds
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        logger.log(
            {
                "round": rounds + 1,
                "loss": loss,
                "accuracy": accuracy,
                "data_samples": len(self.valloader.dataset),
            }
        )
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def create_client(partition_id, model) -> fl.client.Client:

    model_module = importlib.import_module(f"models.{model}")
    net = model_module.Net().to(DEVICE)

    trainloader, valloader, _ = load_datasets(partition_id=partition_id)

    return FlowerClient(net, trainloader, valloader).to_client()


if __name__ == "__main__":
    print(
        f"Starting client {args.name} with partition_id {args.partition_id} and connecting to {args.server_address}"
    )
    rounds = 0
    client = create_client(args.partition_id, model=args.model)
    while rounds < NUM_ROUNDS:
        try:
            print(f"Starting client {args.name} for Round {rounds}")
            fl.client.start_client(server_address=args.server_address, client=client)
            rounds += 1
        except Exception as e:
            print(f"Error: {type(e)}, Couldn't run client. Retrying in 10 seconds...")

        time.sleep(3)
