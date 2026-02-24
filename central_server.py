import importlib
import json
import flwr as fl
from flwr.server import ServerConfig
import argparse
import matplotlib.pyplot as plt
import numpy as np
from config import NUM_ROUNDS, MODEL, SEED, GRADIENT_CORRECTION_BETA, FEDMUT_CENTRAL, FEDMUT_ALPHA
from logger import Logger

from utils import (set_parameters, test, load_datasets, 
                   log_to_dashboard, get_parameters, generate_mutated_models,
                   unpack_compressed_data, decompress_model_update,
                   compress_model_update, pack_compressed_data, get_traffic_metrics, 
                   get_payload_size,)
from flwr.common import parameters_to_ndarrays, FitIns, ndarrays_to_parameters, FitRes, GetPropertiesIns
import time
import pickle

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
    def __init__(self, min_fit_clients, min_available_clients, initial_parameters=None):
        """
        Initializes the Central Server strategy for Hierarchical Federated Learning.
        
        Functionality:
        1. Calls the parent FedAvg constructor to handle basic client sampling and tracking.
        2. Stores the initial global parameters for mutation history tracking (if FedMut is enabled).
        3. Instantiates a reference neural network to dynamically extract layer names 
           and calculate the exact number of layers (num_model_layers).
        4. Initializes a Traffic Logger to record the Downlink bandwidth usage (Cloud -> Edge).
        """
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=lambda rnd: {"round": rnd},
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
            initial_parameters=initial_parameters,
        )
        self.prev_global_weights = None 
        if initial_parameters is not None:
            self.prev_global_weights = parameters_to_ndarrays(initial_parameters)
        self.yi_per_group = {}  # store yi for each group/edge
        # Calculate the split index for weights vs gradients
        model_module = importlib.import_module(f"models.{MODEL}")
        ref_net = model_module.Net()
        # self.num_model_layers = len(get_parameters(ref_net))
        self.grad_names = [n for n, p in ref_net.named_parameters()]
        self.num_model_layers = len(self.grad_names)
        self.grad_shapes = {n: p.shape for n, p in ref_net.named_parameters()}

        self.state_keys = list(ref_net.state_dict().keys())
        self.param_index = {name: self.state_keys.index(name) for name in self.grad_names}

        self.traffic_logger = Logger(
            subfolder="central",
            file_path="traffic.csv",
            headers=[
                "Round", "Direction", 
                "model_wts_MB", "compressed_model_wts_MB",
                "Y_i_MB", "compressed_Y_i_MB", 
                "Z_i_MB", "compressed_Z_i_MB", 
                "Total_MB", "Compressed_Total_MB",
                "compression_time_s", "decompression_time_s"
            ]
        )

    def aggregate_fit(self, rnd, results, failures):
        """
        Aggregates the group models received from Edge Servers to form the new Global Model.
        
        Functionality:
        1. Bypasses custom logic if Gradient Correction is disabled (Beta == 0).
        2. Unpacks the results received from each Edge Server.
        3. Slices the payload to extract ONLY the model weights, ignoring any extra 
           trailing data (which handles backward compatibility if an edge sent gradients).
        4. Filters out any Edge Servers that failed (returned 0 examples) to prevent 
           ZeroDivisionError crashes and mathematical poisoning.
        5. Passes the clean, filtered weights to standard FedAvg for global aggregation.
        """

        if GRADIENT_CORRECTION_BETA == 0:
            # Standard aggregation only
            return super().aggregate_fit(rnd, results, failures)
        
        valid_results = []
        clients_list = []

        for client, fit_res in results:
            # 1. Unpack
            packed_params = parameters_to_ndarrays(fit_res.parameters)
            edge_name = fit_res.metrics.get("client_name", getattr(client, "cid", "unknown"))
            print(f"Received update from Edge-{edge_name}")
            
            # 2. Slice: Weights [0 : N] | Gradients [N : end]
            model_len = fit_res.metrics.get("model_length", len(packed_params))
            weights = packed_params[:model_len]
            packed_tail = packed_params[model_len:]
            
            clients_list.append(client)

            # 4. Create CLEAN FitRes (Weights only) for standard FedAvg
            if fit_res.num_examples > 0:
                new_fit_res = FitRes(
                    status=fit_res.status,
                    parameters=ndarrays_to_parameters(weights),
                    num_examples=fit_res.num_examples,
                    metrics=fit_res.metrics,
                )
                valid_results.append((client, new_fit_res))
            else:
                failure_reason = fit_res.metrics.get("status", "UNKNOWN_ERROR")
                print(f"🚨 Edge-{edge_name} failed! Reason: {failure_reason}")
        
        # --- STANDARD AGGREGATION (Weights Only) ---
        aggregated_parameters = super().aggregate_fit(rnd, valid_results, failures)

        return aggregated_parameters

    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        """
        Prepares and sends the new Global Model (and configuration) down to the Edge Servers.
        
        Functionality:
        1. Uses standard FedAvg sampling to select which Edge Servers participate in this round.
        2. Applies FedMut (Federated Mutation) if enabled, generating diverse variations 
           of the global model to send to different edges to increase exploration.
        3. Calculates the exact size in bytes of the downlink payload.
        4. Logs the Downlink traffic metrics to the dashboard/CSV.
        5. Packages the Global Model (or mutated models) and configurations into FitIns 
           objects and dispatches them to the Edge Servers to start the new round.
        """
        # Get default instructions from FedAvg
        fit_instructions = super().configure_fit(
            server_round, parameters, client_manager, **kwargs
        )

        if not fit_instructions:
            return []

        # 2. Prepare for Mutation (if enabled)
        current_weights = parameters_to_ndarrays(parameters)
        mutated_weights_list = []
        use_mutation = False

        # Check config and ensure we have history (prev_weights) to calculate direction
        if FEDMUT_CENTRAL and self.prev_global_weights is not None and server_round > 1:
            print(f"[Central Server] 🧬 FedMut: Mutating Global Model for {len(fit_instructions)} Edges.")
            mutated_weights_list = generate_mutated_models(
                current_weights, 
                self.prev_global_weights, 
                len(fit_instructions), 
                FEDMUT_ALPHA
            )
            use_mutation = True
        
        # Update history for next round
        self.prev_global_weights = current_weights

        # fit_instructions is a list of (ClientProxy, FitIns)
        new_fit_instructions = []

        for i, (client, fit_ins) in enumerate(fit_instructions):
            c_name = "unknown"
            try:
                res = client.get_properties(GetPropertiesIns(config={}), timeout=10.0, group_id=0)
                c_name = res.properties.get("client_name", "unknown")
            except:
                print(f"[Central] Could not query name for client {i}")
            cfg = fit_ins.config.copy()  # make a copy

            # A. FedMut Logic: Assign specific mutated model
            if use_mutation:
                # Wrap numpy weights back to Parameters object
                client_parameters = ndarrays_to_parameters(mutated_weights_list[i])
            else:
                # Use standard global model
                client_parameters = fit_ins.parameters

            # --- Metrics Logic ---
            # A. Model Size
            model_payload = parameters_to_ndarrays(fit_ins.parameters)
            model_u = get_payload_size(model_payload)
            model_c = model_u # No compression on model weights

            target_id = c_name if c_name != "unknown" else f"Edge_Index_{i}"
            # --- Log Traffic Metrics ---
            metrics = get_traffic_metrics(
            round_num=server_round,
            direction=f"Downlink_to_{target_id}",
                model_tuple=(model_u, model_c),
                yi_tuple=(0, 0),
                comp_time=0.0 
            )
            self.traffic_logger.log(metrics)

            del model_payload                        

            # Re-wrap as FitIns and append with ClientProxy
            new_fit_instructions.append((client, FitIns(client_parameters, cfg)))

        return new_fit_instructions

    def evaluate(self, server_round, parameters):
        """
        Performs Centralized Evaluation of the aggregated Global Model.
        
        Functionality:
        1. Skips evaluation on round 0 (initialization).
        2. Safety Check: Verifies that the aggregated parameters are not all zeros 
           (which would indicate a total aggregation failure).
        3. Loads a fresh instance of the model and injects the new Global Weights.
        4. Loads the global test dataset and runs a full test pass to determine 
           the true global loss and accuracy.
        5. Logs the metrics to the local logger and the external Dashboard (if enabled).
        """
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
        # return super().evaluate(server_round, parameters) # This is returning None
        return float(loss), {"accuracy": float(accuracy)}

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggregates distributed evaluation metrics returned by the Edge Servers.
        
        Functionality:
        1. Collects the local evaluation results (loss and accuracy) computed by the Edge Servers.
        2. Uses the standard FedAvg logic to compute a weighted average of the loss.
        3. Manually computes a weighted average of the accuracies based on the number 
           of examples (num_examples) each Edge Server represents.
        4. Prints and returns the aggregated metrics for the current round.
        """
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
    
# strategy = FedAvgWithGradientCorrection(min_fit_clients=min_edges, min_available_clients=min_edges)

if __name__ == "__main__":
    model_module = importlib.import_module(f"models.{MODEL}")
    net = model_module.Net()
    init_params = ndarrays_to_parameters(get_parameters(net))
    strategy = FedAvgWithGradientCorrection(
        min_fit_clients=min_edges, 
        min_available_clients=min_edges,
        initial_parameters=init_params
    )
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    print(f"Starting central server at {server_address}")
    fl.server.start_server(
        server_address=server_address, strategy=strategy, config=config
    )
