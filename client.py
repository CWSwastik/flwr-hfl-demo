import flwr as fl
import numpy as np
import time
import argparse

from model import Net, load_datasets, get_parameters, set_parameters, train, test, DEVICE
from config import NUM_ROUNDS


parser = argparse.ArgumentParser(description="Start a Flower client.")
parser.add_argument("server_address", help="Server address in the format host:port (e.g., localhost:8081)")
parser.add_argument("--partition_id", type=int, help="Partition ID")
args = parser.parse_args()

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
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def create_client(partition_id) -> fl.client.Client:
    net = Net().to(DEVICE)

    trainloader, valloader, _ = load_datasets(partition_id=partition_id)

    return FlowerClient(net, trainloader, valloader).to_client()

if __name__ == "__main__":

    rounds = 0
    client = create_client(args.partition_id)
    while rounds < NUM_ROUNDS:
        try:
            print(f"Starting client for Round {rounds} and connecting to {args.server_address}, partition_id: {args.partition_id}")
            fl.client.start_client(server_address=args.server_address, client=client)
            rounds += 1
        except Exception as e:
            print(f"Error: {type(e)}, Couldn't run client. Retrying in 10 seconds...")
    
        time.sleep(3)
