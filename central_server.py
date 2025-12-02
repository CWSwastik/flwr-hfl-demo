import importlib
import json
import flwr as fl
from flwr.server import ServerConfig
import argparse
import matplotlib.pyplot as plt
import numpy as np
from config import NUM_ROUNDS, MODEL, SEED
from logger import Logger
from utils import log_to_dashboard

from utils import set_parameters, test, load_datasets, log_to_dashboard, get_parameters
from flwr.common import parameters_to_ndarrays, FitIns, ndarrays_to_parameters, FitRes

parser = argparse.ArgumentParser(description="Start the Flower central server.")
parser.add_argument(
    "address", help="Server address in the format host:port (e.g., 0.0.0.0:8081)"
)
parser.add_argument(
    "--exp_id",
    type=str,
    help="The experiment ID for the dashboard",
)
parser.add_argument(
    "--min_edges",
    type=int,
    help="Minimum number of edge servers needed",
    default=2,
)
parser.add_argument(
    "--enable_dashboard",
    type=bool,
    help="Enable logging to dashboard",
    default=False,
)
args = parser.parse_args()
min_edges = args.min_edges

logger = Logger(
    subfolder="central",
    file_path="central_server.log",
    headers=["round", "loss", "accuracy"],
)

server_address = args.address

np.random.seed(seed=SEED)

class FedAvgWithGradientCorrection(fl.server.strategy.FedAvg):
    def __init__(self, min_fit_clients, min_available_clients):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=lambda rnd: {"round": rnd},
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
        )
        self.yi_per_group = {}  # store yi for each group/edge
        # Calculate the split index for weights vs gradients
        model_module = importlib.import_module(f"models.{MODEL}")
        ref_net = model_module.Net()
        self.num_model_layers = len(get_parameters(ref_net))
        self.grad_names = [n for n, p in ref_net.named_parameters() if p.requires_grad]

    def aggregate_fit(self, rnd, results, failures):

        valid_results = []
        group_grads = [] # Stores the gradient dictionaries from edges
        clients_list = []

        for client, fit_res in results:
            # 1. Unpack
            packed_params = parameters_to_ndarrays(fit_res.parameters)
            
            # 2. Slice: Weights [0 : N] | Gradients [N : end]
            weights = packed_params[:self.num_model_layers]
            raw_grads = packed_params[self.num_model_layers:]
            
            # 3. Reconstruct Gradient Dictionary
            # The edge server sent its 'group_avg_grad' as the packed gradients
            edge_grad_dict = dict(zip(self.grad_names, raw_grads))
            group_grads.append(edge_grad_dict)
            clients_list.append(client)

            # 4. Create CLEAN FitRes (Weights only) for standard FedAvg
            new_fit_res = FitRes(
                status=fit_res.status,
                parameters=ndarrays_to_parameters(weights),
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
            )
            valid_results.append((client, new_fit_res))
        
        # --- STANDARD AGGREGATION (Weights Only) ---
        aggregated_parameters = super().aggregate_fit(rnd, valid_results, failures)


        if aggregated_parameters is not None and results:

            # Average across all groups (edges)
            global_avg_grad = {}
            for name in self.grad_names:
            # for name in group_grads[0]:
                global_avg_grad[name] = np.mean([gg[name] for gg in group_grads], axis=0)

            # Convert global_avg_grad to lists (JSON-safe)
            global_avg_grad_serializable = {name: grad.tolist() for name, grad in global_avg_grad.items()}

            # Compute yi for each group: yi_j = global_avg_grad - group_avg_grad_j
            yi_per_group = {}
            for client, group_grad in zip(clients_list, group_grads):
            # for (client, _), group_grad in zip(results, group_grads):
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
                cid, {name: 0.0 for name in self.grad_names} #cfg.get("zi", {})}
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
        if args.enable_dashboard:
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
            f"[Central Server] Round {server_round}: Average Accuracy = {aggregated_accuracy}"
        )

        return float(aggregated_loss), {"accuracy": float(aggregated_accuracy)}
    
strategy = FedAvgWithGradientCorrection(min_fit_clients=min_edges, min_available_clients=min_edges)

if __name__ == "__main__":
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    print(f"Starting central server at {server_address}")
    fl.server.start_server(
        server_address=server_address, strategy=strategy, config=config
    )
