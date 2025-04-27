import flwr as fl
from flwr.server import ServerConfig
import argparse
import matplotlib.pyplot as plt
from config import NUM_ROUNDS
from logger import Logger

from models.lenet import Net
from utils import set_parameters, test, load_datasets
from flwr.common import parameters_to_ndarrays

parser = argparse.ArgumentParser(description="Start the Flower central server.")
parser.add_argument(
    "address", help="Server address in the format host:port (e.g., 0.0.0.0:8081)"
)
args = parser.parse_args()

logger = Logger(
    subfolder="central",
    file_path="central_server.log",
    headers=["round", "aggregated_loss", "aggregated_accuracy"],
)

server_address = args.address


class FedAvgWithLogging(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            min_fit_clients=2,
            min_available_clients=2,
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
        )

    def evaluate(self, server_round, parameters):
        if server_round == 0:
            print("Skipping evaluation for round 0")
            return super().evaluate(server_round, parameters)

        print(f"[Central Server] Evaluate round {server_round}")
        net = Net()
        # print(parameters_to_ndarrays(parameters)[0][0][0][0])
        set_parameters(net, parameters_to_ndarrays(parameters))
        _, _, testloader = load_datasets()  # full dataset for evaluation
        loss, accuracy = test(net, testloader)
        logger.log(
            {
                "round": server_round,
                "aggregated_loss": loss,
                "aggregated_accuracy": accuracy,
            }
        )

        print(
            f"[Central Server] Evaluate Round {server_round}: Loss = {loss}, Accuracy = {accuracy}"
        )
        return super().evaluate(server_round, parameters)

    def aggregate_evaluate(self, server_round, results, failures):
        """Log loss values after each round."""

        if not results:
            return None, {}

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        aggregated_accuracy = sum(accuracies) / sum(examples)
        # print(list(zip(accuracies, examples)))

        print(
            f"[Central Server] Round {server_round}: Average Loss = {aggregated_loss}"
        )
        print(
            f"[Central Server] Round {server_round}: Average Accuracy = {sum(accuracies) / sum(examples)}"
        )

        return float(aggregated_loss), {"accuracy": float(aggregated_accuracy)}


strategy = FedAvgWithLogging()

if __name__ == "__main__":
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    print(f"Starting central server at {server_address}")
    fl.server.start_server(
        server_address=server_address, strategy=strategy, config=config
    )
