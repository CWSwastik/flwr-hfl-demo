import flwr as fl
import numpy as np
import time
import argparse
import random

from model import Net, load_datasets, get_parameters, set_parameters, train, test, DEVICE

parser = argparse.ArgumentParser(description="Start a Flower client.")
parser.add_argument("server_address", help="Server address in the format host:port (e.g., localhost:8081)")
parser.add_argument("--partition_id", type=int, help="Partition ID")
args = parser.parse_args()

# class SimpleClient(fl.client.NumPyClient):
#     def __init__(self):
#         self.weights = np.array([1.0, 2.0, 3.0])

#     def get_parameters(self, config):
#         return [self.weights]

#     def fit(self, parameters, config):
#         self.weights = np.array(parameters[0]) + random.randrange(10, 100)/100
#         print(f"Client updated weights: {self.weights}")
#         return [self.weights], len(self.weights), {}

#     def evaluate(self, parameters, config):
#         loss = np.sum((np.array(parameters[0]) - 1.0) ** 2)
#         num_examples = 1
#         return loss, num_examples, {"accuracy": 0.9}



class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        if not np.all(parameters[0] == 0):
            print(f"Received new global model from server: {parameters}")
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
    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    trainloader, valloader, _ = load_datasets(partition_id=partition_id)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, trainloader, valloader).to_client()

if __name__ == "__main__":


    max_rounds = 3
    rounds = 0
    client = create_client(args.partition_id)
    while rounds < max_rounds:
        try:
            print(f"Starting client and connecting to {args.server_address}, partition_id: {args.partition_id}")
            fl.client.start_client(server_address=args.server_address, client=client)
            rounds += 1
        except Exception as e:
            print(f"Error: {type(e)}, Couldn't run client. Retrying in 10 seconds...")
    
        time.sleep(10)
