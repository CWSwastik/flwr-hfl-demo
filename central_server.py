import flwr as fl
from flwr.server import ServerConfig
import argparse


parser = argparse.ArgumentParser(description="Start the Flower central server.")
parser.add_argument("address", help="Server address in the format host:port (e.g., 0.0.0.0:8081)")
args = parser.parse_args()

server_address = args.address

strategy = fl.server.strategy.FedAvg(min_fit_clients=2, min_available_clients=2)

if __name__ == "__main__":
    config = ServerConfig(num_rounds=3)
    print(f"Starting central server at {server_address}")
    fl.server.start_server(server_address=server_address, strategy=strategy, config=config)
