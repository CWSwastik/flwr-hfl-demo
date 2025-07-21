import json
import flwr as fl
import numpy as np
import time
import argparse
import requests

from utils import (
    load_datasets,
    get_parameters,
    set_parameters,
    train,
    test,
    DEVICE,
    get_dataloader_summary,
    post_to_dashboard,
    log_to_dashboard,
)
from config import (
    NUM_ROUNDS,
    MODEL,
    TRAINING_LEARNING_RATE,
    TRAINING_WEIGHT_DECAY,
    TRAINING_SCHEDULER_GAMMA,
    TRAINING_SCHEDULER_STEP_SIZE,
    DASHBOARD_SERVER_URL,
)

import importlib
from logger import Logger
import os
import sys, traceback


from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


parser = argparse.ArgumentParser(description="Start a Flower client.")
parser.add_argument(
    "server_address",
    help="Server address in the format host:port (e.g., localhost:8081)",
)
parser.add_argument("--partition_id", type=int, default=0, help="Partition ID")
parser.add_argument(
    "--name", type=str, default="client", help="Client name (default: client)"
)
parser.add_argument(
    "--exp_id",
    type=str,
    help="The experiment ID for the dashboard",
)

args = parser.parse_args()

test_logger = Logger(
    subfolder="clients",
    file_path=f"{args.name}_{MODEL}_test.log",
    headers=["round", "loss", "accuracy", "data_samples"],
)

train_logger = Logger(
    subfolder="clients",
    file_path=f"{args.name}_{MODEL}_train.log",
    headers=["round", "loss", "accuracy", "data_samples"],
)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.round = 0

        self.optimizer = Adam(
            self.net.parameters(),
            lr=TRAINING_LEARNING_RATE,
            weight_decay=TRAINING_WEIGHT_DECAY,
        )

        self.scheduler = StepLR(
            self.optimizer,
            step_size=TRAINING_SCHEDULER_STEP_SIZE,
            gamma=TRAINING_SCHEDULER_GAMMA,
        )

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):

        if not np.all(parameters[0] == 0):
            print(f"Received new global model from server")
            set_parameters(self.net, parameters)
        else:
            print("Received initial model from server, starting training...")

        losses, accuracies = train(self.net, self.trainloader, self.optimizer, epochs=1)
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
        log_to_dashboard(
            args.exp_id,
            "client",
            {
                "device": args.name,
                "round": self.round,
                "loss": loss,
                "accuracy": accuracy,
                "data_samples": len(self.valloader.dataset),
            },
        )
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


def create_client(partition_id, model) -> fl.client.Client:

    model_module = importlib.import_module(f"models.{model}")
    net = model_module.Net().to(DEVICE)

    trainloader, valloader, testloader = load_datasets(partition_id=partition_id)
    print("Trainloader size:", len(trainloader.dataset))
    print("Valloader size:", len(valloader.dataset))
    print("Testloader size:", len(testloader.dataset))
    print("Trainloader summary:", get_dataloader_summary(trainloader))
    print("Valloader summary:", get_dataloader_summary(valloader))
    print("Testloader summary:", get_dataloader_summary(testloader))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(
            current_dir,
            "logs",
            "clients",
            f"{args.name}_{args.partition_id}_data_dist.json",
        ),
        "w",
    ) as f:
        json.dump(
            {
                "trainloader": get_dataloader_summary(trainloader),
                "valloader": get_dataloader_summary(valloader),
                "testloader": get_dataloader_summary(testloader),
            },
            f,
        )

    url = f"{DASHBOARD_SERVER_URL}/experiment/{args.exp_id}/distribution/client"
    dist = {
        "trainloader": get_dataloader_summary(trainloader),
        "valloader": get_dataloader_summary(valloader),
        "testloader": get_dataloader_summary(testloader),
    }
    payload = {"device": args.name, "distribution": dist}
    post_to_dashboard(url, payload)

    return FlowerClient(net, trainloader, valloader)


if __name__ == "__main__":
    print(
        f"Starting client {args.name} with partition_id {args.partition_id} and connecting to {args.server_address}"
    )
    client = create_client(args.partition_id, model=MODEL)
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
