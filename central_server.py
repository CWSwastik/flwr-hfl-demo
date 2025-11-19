import importlib
import json
import flwr as fl
from flwr.server import ServerConfig
import argparse
import matplotlib.pyplot as plt
import numpy as np
from config import NUM_ROUNDS, MODEL
from logger import Logger
from utils import log_to_dashboard

from utils import set_parameters, test, load_datasets
from flwr.common import parameters_to_ndarrays, FitIns

parser = argparse.ArgumentParser(description="Start the Flower central server.")
parser.add_argument(
    "address", help="Server address in the format host:port (e.g., 0.0.0.0:8081)"
)
parser.add_argument(
    "--exp_id",
    type=str,
    help="The experiment ID for the dashboard",
)
args = parser.parse_args()

logger = Logger(
    subfolder="central",
    file_path="central_server.log",
    headers=["round", "loss", "accuracy"],
)

server_address = args.address

class FedAvgWithGradientCorrection(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            min_fit_clients=2,
            min_available_clients=2,
            on_fit_config_fn=lambda rnd: {"round": rnd},
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
        )
        self.yi_per_group = {}  # store yi for each group/edge

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        # print(aggregated_parameters, rnd, results, failures)

        if aggregated_parameters is not None and results:
            # Each result.metrics["gradients"] contains group_avg_grad from edge
            group_grads = [json.loads(r.metrics["gradients"]) for _, r in results]

            # Average across all groups (edges)
            global_avg_grad = {}
            for name in group_grads[0]:
                global_avg_grad[name] = np.mean([gg[name] for gg in group_grads], axis=0)

            # Convert global_avg_grad to lists (JSON-safe)
            global_avg_grad_serializable = {name: grad.tolist() for name, grad in global_avg_grad.items()}

            # Compute yi for each group: yi_j = global_avg_grad - group_avg_grad_j
            yi_per_group = {}
            for (client, _), group_grad in zip(results, group_grads):
                client_id = getattr(client, "cid", None)
                yi_per_group[client_id] = {name: (global_avg_grad[name] - group_grad[name]).tolist()
                                        for name in group_grad}

            # Save yi for next round
            self.yi_per_group = yi_per_group
            self.global_avg_grad = global_avg_grad_serializable

            print(f"[Central Server] Computed yi for {len(yi_per_group)} groups.")

        return aggregated_parameters

    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        """
        Configure per-client fit instructions with yi.
        """
        # Get default instructions from FedAvg
        fit_instructions = super().configure_fit(
            server_round, parameters, client_manager, **kwargs
        )

        # fit_instructions is a list of (ClientProxy, FitIns)
        new_fit_instructions = []

        for client, fit_ins in fit_instructions:
            cfg = fit_ins.config.copy()  # make a copy

            # Get client_id from config or ClientProxy
            cid = cfg.get("cid", getattr(client, "cid", None))

            # Fetch yi from yi_per_group
            yi = self.yi_per_group.get(
                cid, {name: 0.0 for name in cfg.get("zi", {})}
            )

            # Add yi to config
            cfg["yi"] = json.dumps(yi)

            # Re-wrap as FitIns and append with ClientProxy
            new_fit_instructions.append((client, FitIns(fit_ins.parameters, cfg)))

        return new_fit_instructions

    def evaluate(self, server_round, parameters):
        if server_round == 0:
            print("Skipping evaluation for round 0")
            return super().evaluate(server_round, parameters)

        print(f"[Central Server] Evaluate round {server_round}")

        param_arrays = parameters_to_ndarrays(parameters)
        if all(np.allclose(p, 0) for p in param_arrays):
            print("[Warning] All parameters are zero! Skipping evaluation.")
            return super().evaluate(server_round, parameters)

        model_module = importlib.import_module(f"models.{MODEL}")
        net = model_module.Net()

        # print(parameters_to_ndarrays(parameters)[0][0][0][0])
        set_parameters(net, param_arrays)
        _, _, testloader = load_datasets()  # full dataset for evaluation
        loss, accuracy = test(net, testloader)
        logger.log(
            {
                "round": server_round,
                "loss": loss,
                "accuracy": accuracy,
            }
        )

        # Log to dashboard
        log_to_dashboard(
            args.exp_id,
            "central",
            {
                "device": "central_server",
                "round": server_round,
                "loss": loss,
                "accuracy": accuracy,
            },
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
    
strategy = FedAvgWithGradientCorrection()

if __name__ == "__main__":
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    print(f"Starting central server at {server_address}")
    fl.server.start_server(
        server_address=server_address, strategy=strategy, config=config
    )
