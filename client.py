import flwr as fl
import numpy as np
import time
import argparse
import random


parser = argparse.ArgumentParser(description="Start a Flower client.")
parser.add_argument("server_address", help="Server address in the format host:port (e.g., localhost:8081)")
args = parser.parse_args()

class SimpleClient(fl.client.NumPyClient):
    def __init__(self):
        self.weights = np.array([1.0, 2.0, 3.0])

    def get_parameters(self, config):
        return [self.weights]

    def fit(self, parameters, config):
        self.weights = np.array(parameters[0]) + random.randrange(10, 100)/100
        print(f"Client updated weights: {self.weights}")
        return [self.weights], len(self.weights), {}

    def evaluate(self, parameters, config):
        loss = np.sum((np.array(parameters[0]) - 1.0) ** 2)
        num_examples = 1
        return loss, num_examples, {"accuracy": 0.9}

if __name__ == "__main__":
    max_rounds = 3
    rounds = 0
    while rounds < max_rounds:
        try:
            print(f"Starting client and connecting to {args.server_address}")
            fl.client.start_client(server_address=args.server_address, client=SimpleClient().to_client())
            rounds += 1
        except Exception as e:
            print(f"Error: {type(e)}, Couldn't run client. Retrying in 10 seconds...")
    
        time.sleep(10)
