import importlib
import json
import sys
import traceback
import flwr as fl
from flwr.server import ServerConfig
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns, FitRes, Parameters, Status, Code, GetPropertiesIns
import numpy as np
import multiprocessing
import argparse
from logger import Logger
from utils import ( load_datasets, set_parameters, test, log_to_dashboard, get_parameters, generate_mutated_models,
                    decompress_model_update, compress_model_update, get_payload_size, unpack_compressed_data,
                    pack_compressed_data, get_traffic_metrics, )
from config import COMPRESS_YI, COMPRESS_ZI, MODEL, MIN_CLIENTS_PER_EDGE, GRADIENT_CORRECTION_BETA, ENABLE_DASHBOARD, SEED, FEDMUT_EDGE, FEDMUT_ALPHA, COMPRESSION_METHOD
import gc
import pickle
import os
import time

# Optimize Threading to prevent freezing
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"

os.environ["GRPC_KEEPALIVE_TIME_MS"] = "10000"
os.environ["GRPC_KEEPALIVE_TIMEOUT_MS"] = "5000"

parser = argparse.ArgumentParser(description="Start a Flower Edge Server.")
parser.add_argument(
    "--server", required=True, help="Central server address (e.g., localhost:8081)"
)
parser.add_argument(
    "--client", required=True, help="Edge client address (e.g., localhost:8080)"
)

parser.add_argument(
    "--name",
    type=str,
    required=True,
    help="Edge Server name for logging",
)
parser.add_argument(
    "--exp_id",
    type=str,
    help="The experiment ID for the dashboard",
)
parser.add_argument(
    "--min_clients",
    type=int,
    help="Minimum number of clients per edge server",
    default=MIN_CLIENTS_PER_EDGE,
)

args = parser.parse_args()
min_clients = args.min_clients

logger = Logger(
    subfolder="edge",
    file_path=f"{args.name}.log",
    headers=["round", "loss", "accuracy"],
    init_file=False,
)

np.random.seed(seed=SEED)

class EdgeStrategy(fl.server.strategy.FedAvg):
    def __init__(self, shared_state, round, **kwargs):
        super().__init__(**kwargs)
        self.shared_state = shared_state
        self.round = round
        model_module = importlib.import_module(f"models.{MODEL}")
        ref_net = model_module.Net()
        self.num_model_layers = len(get_parameters(ref_net))
        self.grad_names = [n for n, p in ref_net.named_parameters()]
        self.traffic_logger = Logger(
            subfolder="edge",
            file_path=f"{args.name}_traffic.csv",
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
        print(f"[Edge Server {args.name}] Aggregating fit results at round {rnd}.")
        current_global_round = self.round
        
        if GRADIENT_CORRECTION_BETA == 0:
            print(f"[Edge Server {args.name}] Aggregating (Standard FedAvg).")
            
            # for non-gradient correction the clients sent standard weights, so we can pass them directly to FedAvg
            # No unpacking, no slicing, no Zi calculation.
            aggregated_parameters = super().aggregate_fit(rnd, results, failures)
            
            if aggregated_parameters is not None:
                self.shared_state["aggregated_model"] = aggregated_parameters[0]
                examples = [r.num_examples for _, r in results]
                self.shared_state["num_examples"] = sum(examples)
            
            return aggregated_parameters

        # --- Custom Aggregation Logic ---
        valid_results = []
        client_grads_by_name = {}
        client_grads = []
        clients_list = []
        client_names = []

        for client, fit_res in results:
            # 1. Convert packed parameters to NumPy
            packed_params = parameters_to_ndarrays(fit_res.parameters)
            
            # 2. SLICE the list [ Weights (0 to N) | Gradients (N to end) ]
            weights = packed_params[:self.num_model_layers]
            packed_tail = packed_params[self.num_model_layers:]
            
            c_grad_dict = {}
            is_compressed = fit_res.metrics.get("is_compressed", False)

            # 3. Decompress Gradients
            if is_compressed and len(packed_tail) > 0:
                # print(f"[Edge Server] Decompressing gradients for client {getattr(client, 'cid', 'N/A')}")
                # The tail contains a single blob [blob]
                blob = packed_tail[0]
                compressed_dict = unpack_compressed_data(blob)
                c_grad_dict = decompress_model_update(compressed_dict)
            else:
                # Raw List
                if len(packed_tail) == len(self.grad_names):
                        c_grad_dict = dict(zip(self.grad_names, packed_tail))
                else:
                    print(f"[Error] Uncompressed list length mismatch! Expected {len(self.grad_names)}, got {len(packed_tail)}")

            # 4. Padding safety
            for name in self.grad_names:
                if name not in c_grad_dict:
                    idx = self.grad_names.index(name)
                    c_grad_dict[name] = np.zeros_like(weights[idx])

            client_grads.append(c_grad_dict)
            clients_list.append(client)

            c_name = fit_res.metrics.get("client_name", "Unknown")
            client_grads_by_name[c_name] = c_grad_dict
            client_names.append(c_name)
            
            # 5. Clean FitRes
            new_fit_res = FitRes(
                status=fit_res.status,
                parameters=ndarrays_to_parameters(weights), 
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
            )
            valid_results.append((client, new_fit_res))
        
        # 6. Call Super (Weights Aggregation)
        aggregated_parameters = super().aggregate_fit(rnd, valid_results, failures)
        
        if aggregated_parameters is not None and client_grads_by_name:
            self.shared_state["aggregated_model"] = aggregated_parameters[0]
            examples = [r.num_examples for _, r in valid_results]
            self.shared_state["num_examples"] = sum(examples)

            if not client_grads:
                print(f"[Edge Server] ⚠️ Warning: No gradients received from clients. Skipping Zi.")
                return aggregated_parameters

            # 7. Compute Average Gradient
            avg_grad = {}
            all_grads = list(client_grads_by_name.values())
            for name in self.grad_names:
                avg_grad[name] = np.mean([cg[name] for cg in all_grads], axis=0)
            # for name in client_grads[0]:
            #     avg_grad[name] = np.mean([cg[name] for cg in client_grads], axis=0)

            # Compute zi_per_client as differences
            zi_per_client = {}
            # Store Zi by INDEX (0, 1, 2...)
            # We use index because CIDs change every round.
            # for i, grads in enumerate(client_grads):
            #     # Calculate Zi
            #     zi_value = {name: (avg_grad[name] - grads[name]).tolist() for name in grads}
            #     zi_per_client[i] = zi_value
                # Map Index to Name for logging later
                # self.shared_state[f"name_map_{i}"] = client_names[i]
            for c_name, grad in client_grads_by_name.items():
                zi_per_client[c_name] = {k: avg_grad[k] - grad[k] for k in grad}

            # for client, grads in zip(clients_list, client_grads):
            #     client_id_str = getattr(client, "cid", str(client))
            #     zi_per_client[client_id_str] = {name: (avg_grad[name] - grads[name]).tolist()
            #                         for name in grads}

            # Store safely in shared_state
            # print(zi_per_client, avg_grad_serializable)
            self.shared_state["zi_per_client"] = zi_per_client
            self.shared_state["group_avg_grad"] = pickle.dumps(avg_grad)
            
            print(f"[Edge Server] Computed zi for {len(zi_per_client)} clients.")
            print(f"[Edge Server] Aggregated model at round {rnd}, global round {current_global_round}.")

            del client_grads, valid_results, avg_grad, zi_per_client
            gc.collect()
        else:
             print(f"[Edge Server] ⚠️ Aggregation yielded None (Not enough clients?).")

        return aggregated_parameters


    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        self.shared_state["aggregated_loss"] = aggregated_loss
        print(
            f"[Edge Server] Aggregated evaluation loss at round {server_round}: {aggregated_loss}"
        )

        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        # print(list(zip(accuracies, examples)))
        aggregated_accuracy = sum(accuracies) / sum(examples)

        # print(f"[Edge Server] Number of examples: {self.shared_state['num_examples']}")
        self.shared_state["aggregated_accuracy"] = aggregated_accuracy
        print(
            f"[Edge Server] Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}"
        )

        return float(aggregated_loss), {"accuracy": float(aggregated_accuracy)}

    def evaluate(self, server_round, parameters):
        # print(f"Server round: {server_round}", "But real round:", self.round)

        if server_round == 0:
            # Skip evaluation for round 0
            return super().evaluate(server_round, parameters)

        server_round = self.round
        print(f"[Edge Server] Evaluate round {server_round}")

        model_module = importlib.import_module(f"models.{MODEL}")
        net = model_module.Net()
        set_parameters(net, parameters_to_ndarrays(parameters))
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
        if ENABLE_DASHBOARD:
            log_to_dashboard(
                args.exp_id,
                "edge",
                {
                    "device": args.name,
                    "round": server_round,
                    "loss": loss,
                    "accuracy": accuracy,
                },
            )
        print(f"[Edge Server] Evaluate Round {server_round}: Loss = {loss}, Accuracy = {accuracy}")
        return float(loss), {"accuracy": float(accuracy)}


    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        """Send per-client zi and global yi to clients."""
        print(f"[Edge Server] Configuring fit for round {server_round}...")
        current_global_round = self.round

        fit_instructions = super().configure_fit(
            server_round, parameters, client_manager, **kwargs
        )
        if not fit_instructions: return []

        current_weights = parameters_to_ndarrays(parameters)
        prev_weights = self.shared_state.get("fedmut_history_for_round", None)
        
        mutated_weights_list = []
        use_mutation = False

        if FEDMUT_EDGE and prev_weights is not None:
            print(f"[Edge Server] 🧬 FedMut: Mutating Model for {len(fit_instructions)} Clients.")
            mutated_weights_list = generate_mutated_models(
                current_weights,
                prev_weights,
                len(fit_instructions),
                FEDMUT_ALPHA
            )
            use_mutation = True
        
        # yi is received from central server, stored in shared_state
        yi_blob = self.shared_state.get("yi", b"")
        yi_is_compressed = self.shared_state.get("yi_is_compressed", False)

        # Smart Forwarding Logic:
        # If Central sent compressed Yi, and we want compressed, use it directly (No double compression).
        # If Central sent raw Yi (pickle), and we want compressed, compress it now.
        final_yi_blob = yi_blob
        final_yi_compressed = yi_is_compressed
        
        yi_comp_time = 0.0
        if GRADIENT_CORRECTION_BETA!=0 and yi_blob:
            if COMPRESSION_METHOD != "none" and COMPRESS_YI:
                if not yi_is_compressed:
                    yi_comp_start = time.time()
                    # Central sent Pickle, but Edge needs Compression -> Compress now
                    print(f"[Edge Server] Compressing Yi locally...")
                    yi_dict = pickle.loads(yi_blob)
                    c_yi = compress_model_update(yi_dict)
                    final_yi_blob = pack_compressed_data(c_yi).tobytes()
                    yi_comp_time = time.time() - yi_comp_start
                    final_yi_compressed = True

        zi_per_client = self.shared_state.get("zi_per_client", {})
        beta = GRADIENT_CORRECTION_BETA

        new_fit_instructions = []

        for i, (client, fit_ins) in enumerate(fit_instructions):
            # cid = getattr(client, "cid", None)
            c_name = "unknown"
            try:
                # This blocks until client replies (or timeout)
                res = client.get_properties(
                    GetPropertiesIns(config={}), 
                    timeout=10.0,
                    group_id=0 # Required by GrpcClientProxy
                )
                c_name = res.properties["client_name"]
            except Exception as e:
                print(f"[Edge Server] Failed to query properties from client {client}: {e}")

            client_parameters = fit_ins.parameters
            if use_mutation:
                # client_weights_arrays = mutated_weights_list[i]
                current_weights = mutated_weights_list[i]
                client_parameters = ndarrays_to_parameters(mutated_weights_list[i])

            # client_zi = zi_per_client.get(cid, None)
            client_zi = zi_per_client.get(c_name, {})
            # target_name = f"Client_Index_{i}"

            if client_zi is None and zi_per_client:
                 # Fallback logic
                 first = next(iter(zi_per_client.values()))
                 client_zi = {k: np.zeros_like(v) for k,v in first.items()}
            elif client_zi is None:
                 client_zi = {}

            zi_blob = b""
            zi_is_compressed = False

            zi_u = get_payload_size(client_zi)
            zi_c = zi_u
            zi_comp_time = 0.0
            if GRADIENT_CORRECTION_BETA != 0 and client_zi:
                if COMPRESSION_METHOD != "none" and COMPRESS_ZI:
                    # print(f"[Edge Server] Compressing Zi for client {cid}...")
                    comp_start = time.time()
                    compressed_zi = compress_model_update(client_zi)
                    # Convert array to bytes
                    zi_blob = pack_compressed_data(compressed_zi).tobytes()
                    zi_c = get_payload_size(zi_blob)
                    zi_comp_time = time.time() - comp_start
                    zi_is_compressed = True
                else:
                    # No compression
                    zi_blob = pickle.dumps(client_zi)
                    zi_is_compressed = False
            
            cfg = fit_ins.config.copy()
            cfg.update({
                "round": server_round,
                "zi_compressed": zi_is_compressed, 
                "zi": zi_blob,
                "yi": final_yi_blob,
                "yi_compressed": final_yi_compressed,
                "beta": beta,
                # "cid": cid,
                "compression_method": COMPRESSION_METHOD 
            })

            model_u = get_payload_size(current_weights) 
            model_c = model_u
            yi_u = get_payload_size(yi_blob)
            yi_c = get_payload_size(final_yi_blob)
            
            metrics = get_traffic_metrics(
                round_num=current_global_round,
                # direction=f"Downlink_to_{getattr(client, 'cid', 'unknown')}",
                direction=f"Downlink_to_{c_name}",
                model_tuple=(model_u, model_c),
                yi_tuple=(yi_u, yi_c),
                zi_tuple=(zi_u, zi_c),
                comp_time= zi_comp_time + yi_comp_time,
            )
            self.traffic_logger.log(metrics)

            new_fit_instructions.append((client, FitIns(client_parameters, cfg)))
            
        del zi_per_client, yi_blob, final_yi_blob
        gc.collect()

        print(f"Prepared {len(new_fit_instructions)} fit instructions")
        return new_fit_instructions

def run_edge_server(shared_state, params, round):
    strategy = EdgeStrategy(
        shared_state,
        round,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=ndarrays_to_parameters(params),
        # on_evaluate_config_fn=lambda rnd: {"round": rnd},
    )
    config = ServerConfig(num_rounds=1)

    print(f"[Edge Server {args.name}] Starting on {args.client}")
    fl.server.start_server(server_address=args.client, strategy=strategy, config=config)


def run_edge_as_client(shared_state):
    class EdgeClient(fl.client.NumPyClient):
        def __init__(self, shared_state):
            self.shared_state = shared_state
            self.traffic_logger = Logger(
                subfolder="edge",
                file_path=f"{args.name}_traffic.csv",
                headers=[
                    "Round", "Direction", 
                    "model_wts_MB", "compressed_model_wts_MB",
                    "Y_i_MB", "compressed_Y_i_MB", 
                    "Z_i_MB", "compressed_Z_i_MB", 
                    "Total_MB", "Compressed_Total_MB",
                    "compression_time_s", "decompression_time_s"
                ]
            )

        def get_properties(self, config):
            return {"client_name": args.name}
        
        def get_parameters(self, config):
            if self.shared_state.get("aggregated_model") is not None:
                return parameters_to_ndarrays(self.shared_state["aggregated_model"])
            print(f"[Edge Client {args.name}] No aggregated model available yet. Returning 0s.")
            return [np.array([0.0])]

        def fit(self, parameters, config):
            try:
                print(f"[Edge Client {args.name}] Received model from central server.")
                
                current_weights = parameters
                prev_weights = self.shared_state.get("fedmut_prev_weights", None)
                self.shared_state["fedmut_prev_weights"] = current_weights
                self.shared_state["fedmut_history_for_round"] = prev_weights
                
                if GRADIENT_CORRECTION_BETA == 0:
                    server_process = multiprocessing.Process(
                        target=run_edge_server,
                        args=(self.shared_state, parameters, config["round"]),
                        daemon=True
                    )
                    server_process.start()
                    server_process.join()
                    
                    agg_model = self.shared_state.get("aggregated_model")
                    if agg_model is not None:
                        num_examples = self.shared_state.get("num_examples")
                        edge_weights = parameters_to_ndarrays(agg_model)
                        self.shared_state["aggregated_model"] = None
                        gc.collect()
                        return edge_weights, num_examples, {}
                    else:
                         # No-Op Fallback
                        print(f"[Edge Client {args.name}] No-GC Aggregation Failed. Returning No-Op.")
                        return parameters, 0, {}

                yi_blob = config.get("yi", b"")
                yi_compressed = config.get("yi_compressed", False)
                self.shared_state["yi"] = yi_blob
                self.shared_state["yi_is_compressed"] = yi_compressed

                server_process = multiprocessing.Process(
                    target=run_edge_server,
                    args=(self.shared_state, parameters, config["round"]),
                    daemon=True, # Added Daemon for safety
                )
                server_process.start()
                server_process.join()

                agg_model = self.shared_state.get("aggregated_model")

                # --- FAILURE HANDLING LOGIC ---
                if agg_model is not None:
                    # SUCCESS PATH
                    num_examples = self.shared_state.get("num_examples")
                    edge_weights = parameters_to_ndarrays(agg_model)
                    
                    grad_blob = self.shared_state.get("group_avg_grad")
                    if grad_blob:
                        group_avg_grad_dict = pickle.loads(grad_blob)
                    else:
                        group_avg_grad_dict = {} 
                    
                    model_module = importlib.import_module(f"models.{MODEL}")
                    ref_net = model_module.Net()
                    grad_dict_to_send = {}
                    
                    for name, p in ref_net.named_parameters():
                        if name in group_avg_grad_dict:
                            grad_dict_to_send[name] = np.array(group_avg_grad_dict[name])
                    
                    model_u = get_payload_size(edge_weights)
                    model_c = model_u
                    grad_u = get_payload_size(grad_dict_to_send)
                    grad_c = grad_u
                    comp_time = 0.0

                    metrics = {}
                    payload_tail = []

                    if COMPRESSION_METHOD != "none":
                        t_start = time.time()
                        compressed_grads = compress_model_update(grad_dict_to_send)
                        packed_blob = pack_compressed_data(compressed_grads)
                        t_end = time.time()

                        payload_tail = [packed_blob]
                        comp_time = t_end - t_start
                        grad_c = get_payload_size(packed_blob)
                        metrics = {"is_compressed": True, "model_length": len(edge_weights), "client_name": args.name}
                    else:
                        grad_list_padded = []
                        for name, p in ref_net.named_parameters():
                            if name in grad_dict_to_send:
                                grad_list_padded.append(grad_dict_to_send[name])
                            else:
                                grad_list_padded.append(np.zeros_like(p.detach().cpu().numpy()))
                                
                        payload_tail = grad_list_padded
                        metrics = {"is_compressed": False, "model_length": len(edge_weights), "client_name": args.name}

                    metrics_dict = get_traffic_metrics(
                        round_num=config["round"],
                        direction="Uplink",
                        model_tuple=(model_u, model_c),
                        grad_tuple=(grad_u, grad_c),
                        comp_time=comp_time
                    )
                    self.traffic_logger.log(metrics_dict)

                    packed_params = edge_weights + payload_tail

                    del edge_weights, group_avg_grad_dict
                    self.shared_state["aggregated_model"] = None
                    self.shared_state["group_avg_grad"] = None
                    gc.collect()
                    
                    return packed_params, num_examples, metrics
                else:
                    # FAILURE PATH: Inner aggregation failed (0 clients).
                    # We return the ORIGINAL weights with 0 examples.
                    # This tells Central Server: "I am alive, but I have no update."
                    print(f"[Edge Client {args.name}] ⚠️ Aggregation Failed (All inner clients failed). Returning No-Op to keep connection alive.")
                    return parameters, 0, {"is_compressed": False}
                    
            except Exception as e:
                print(f"[Edge Client {args.name}] ❌ CRASH IN FIT: {e}")
                traceback.print_exc()
                # If we crash here, we must raise so Central knows.
                # But since we want to avoid exiting, we can try returning No-Op too.
                # But a crash usually means something worse. Let's raise.
                raise e

    print(f"[Edge Client {args.name}] Connecting to central server {args.server}")
    while True:
        try:
            fl.client.start_client(
                server_address=args.server, 
                client=EdgeClient(shared_state).to_client()
            )
            print(f"[Edge Client {args.name}] Disconnected nicely.")
            break
        except Exception as e:
            print(f"[Edge Client {args.name}] Connection lost: {e}. Retrying in 5s...")
            time.sleep(5)

if __name__ == "__main__":
    logger._init_file()
    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    shared_state["aggregated_model"] = None
    shared_state["aggregated_eval"] = None
    shared_state["num_examples"] = 1

    client_process = multiprocessing.Process(
        target=run_edge_as_client, args=(shared_state,)
    )
    client_process.start()
    client_process.join()
    print(f"[Edge Server {args.name}] Edge client process has ended.")
