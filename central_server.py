import importlib
import json
import flwr as fl
from flwr.server import ServerConfig
import argparse
import matplotlib.pyplot as plt
import numpy as np
from config import COMPRESS_YI, NUM_ROUNDS, MODEL, SEED, GRADIENT_CORRECTION_BETA, FEDMUT_CENTRAL, FEDMUT_ALPHA, COMPRESSION_METHOD
from logger import Logger

from utils import (set_parameters, test, load_datasets, 
                   log_to_dashboard, get_parameters, generate_mutated_models,
                   unpack_compressed_data, decompress_model_update,
                   compress_model_update, pack_compressed_data, get_traffic_metrics, 
                   get_payload_size,)
from flwr.common import parameters_to_ndarrays, FitIns, ndarrays_to_parameters, FitRes
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
        self.num_model_layers = len(get_parameters(ref_net))
        self.grad_names = [n for n, p in ref_net.named_parameters()]
        self.grad_shapes = {n: p.shape for n, p in ref_net.named_parameters()}
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

        if GRADIENT_CORRECTION_BETA == 0:
            # Standard aggregation only
            return super().aggregate_fit(rnd, results, failures)
        
        valid_results = []
        group_grads = [] # Stores the gradient dictionaries from edges
        clients_list = []

        for client, fit_res in results:
            # 1. Unpack
            packed_params = parameters_to_ndarrays(fit_res.parameters)
            is_compressed = fit_res.metrics.get("is_compressed", False)
            edge_name = fit_res.metrics.get("client_name", "Unknown_Edge")
            print(f"Received update from Edge-{edge_name}")
            
            # 2. Slice: Weights [0 : N] | Gradients [N : end]
            weights = packed_params[:self.num_model_layers]
            packed_tail = packed_params[self.num_model_layers:]
            
            edge_grad_dict = {}

            # Decompression Logic
            if is_compressed and len(packed_tail) > 0:
                # print(f"[Central Server] Decompressing update from Edge {getattr(client, 'cid', 'N/A')}")
                blob = packed_tail[0]
                compressed_dict = unpack_compressed_data(blob)
                edge_grad_dict = decompress_model_update(compressed_dict)
            else:
                # Standard raw list of gradients -> Convert to Dict
                raw_list = packed_tail

                if len(raw_list) != len(self.grad_names):
                    print(f"[Warning] Central: Gradient length mismatch! Expected {len(self.grad_names)}, got {len(raw_list)}")

                # 3. Reconstruct Gradient Dictionary
                # The edge server sent its 'group_avg_grad' as the packed gradients
                edge_grad_dict = dict(zip(self.grad_names, raw_list))
            
            # --- Safety Padding (Fill missing layers with Zeros) ---
            for name in self.grad_names:
                if name not in edge_grad_dict:
                    # Use stored shape to create zero tensor
                    edge_grad_dict[name] = np.zeros(self.grad_shapes[name])

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
            if group_grads:
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
                    # old yi
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
            cfg = fit_ins.config.copy()  # make a copy

            # Get client_id from config or ClientProxy
            cid = cfg.get("cid", getattr(client, "cid", None))

            # A. FedMut Logic: Assign specific mutated model
            if use_mutation:
                # Wrap numpy weights back to Parameters object
                client_parameters = ndarrays_to_parameters(mutated_weights_list[i])
            else:
                # Use standard global model
                client_parameters = fit_ins.parameters

            default_yi = {
                name: np.zeros(shape) 
                for name, shape in self.grad_shapes.items()
            }

            # Fetch yi from yi_per_group
            yi = self.yi_per_group.get(cid, default_yi)
            yi_blob = b""
            yi_is_compressed = False

            # --- Metrics Logic ---
            # A. Model Size
            model_payload = parameters_to_ndarrays(fit_ins.parameters)
            model_u = get_payload_size(model_payload)
            model_c = model_u # No compression on model weights

            # B. Yi Size & Compression Time
            yi_u = get_payload_size(yi)
            yi_c = yi_u
            comp_time = 0.0

            if COMPRESSION_METHOD != "none" and COMPRESS_YI and yi:
                # 1. Convert to dict for maintaining key names(layer names)
                yi_dict_to_send = {}

                for k in self.grad_names:
                    if k in yi:
                        val = yi[k]
                        if isinstance(val, list):
                            val = np.array(val)
                        yi_dict_to_send[k] = val

                comp_start = time.time()
                # 2. Compress & Pack
                compressed_yi = compress_model_update(yi_dict_to_send)
                yi_array = pack_compressed_data(compressed_yi)
                yi_blob = yi_array.tobytes() 
                comp_time = time.time() - comp_start
                yi_c = get_payload_size(yi_blob)
                yi_is_compressed = True

                # # 3. Append to parameters (Weights + Blob)
                # client_weights = parameters_to_ndarrays(fit_ins.parameters)
                # full_payload = client_weights + [yi_blob]

                # client_parameters = ndarrays_to_parameters(full_payload)
                # cfg["yi_compressed"] = True # Flag for Edge Server to decompress

            else:
                # Fallback (Existing logic)
                # yi_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in yi.items()}
                # # cfg["yi"] = json.dumps(yi_serializable)
                # cfg["yi"] = pickle.dumps(yi_serializable)
                # client_parameters = fit_ins.parameters
                yi_blob = pickle.dumps(yi)
                yi_is_compressed = False
            
            cfg["yi"] = yi_blob
            cfg["yi_compressed"] = yi_is_compressed
            target_id = f"Edge_Index_{i}"
            # --- Log Traffic Metrics ---
            metrics = get_traffic_metrics(
            round_num=server_round,
            direction=f"Downlink_to_{target_id}",
                model_tuple=(model_u, model_c),
                yi_tuple=(yi_u, yi_c),
                comp_time=comp_time 
            )
            self.traffic_logger.log(metrics)

            del model_payload                        

            # Re-wrap as FitIns and append with ClientProxy
            new_fit_instructions.append((client, FitIns(client_parameters, cfg)))

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
        # return super().evaluate(server_round, parameters) # This is returning None
        return float(loss), {"accuracy": float(accuracy)}

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
