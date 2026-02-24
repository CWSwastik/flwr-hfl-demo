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
from config import COMPRESS_YI, MODEL, MIN_CLIENTS_PER_EDGE, GRADIENT_CORRECTION_BETA, ENABLE_DASHBOARD, SEED, FEDMUT_EDGE, FEDMUT_ALPHA, COMPRESSION_METHOD
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
    """
        Initializes the Inner Edge Strategy.
        
        Functionality:
        1. Sets up the standard FedAvg strategy to manage the leaf clients.
        2. Links to a `shared_state` dictionary (multiprocessing manager). This allows 
           the inner Flower Server to pass data (like the final group model and metrics) 
           back out to the EdgeClient process that spawned it.
        """
    def __init__(self, shared_state, round, **kwargs):
        super().__init__(**kwargs)
        self.shared_state = shared_state
        self.round = round
        model_module = importlib.import_module(f"models.{MODEL}")
        ref_net = model_module.Net()
        self.num_model_layers = len(get_parameters(ref_net))
        self.grad_names = [n for n, p in ref_net.named_parameters()]
        self.num_model_layers = len(self.grad_names)
        self.state_keys = list(ref_net.state_dict().keys())
        self.param_index = {name: self.state_keys.index(name) for name in self.grad_names}

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
        """
        Aggregates local models from inner clients to form the Group Model.
        
        Functionality:
        1. Extracts the weights and H/lr metrics from the clients.
        2. Calculates the weighted average for H_edge and lr_edge.
        3. Uses standard FedAvg to average the weights.
        4. Caches the newly created Group Model into `shared_state["last_group_model"]`
           so the EdgeClient can calculate y_j drift in the next round.
        """
        print(f"[Edge Server {args.name}] Aggregating fit results for round {rnd}.")
        current_global_round = self.round
        
        # --- 1. Pure FedAvg Path (No Correction) ---
        if GRADIENT_CORRECTION_BETA == 0:
            aggregated_parameters = super().aggregate_fit(rnd, results, failures)
            if aggregated_parameters is not None:
                self.shared_state["aggregated_model"] = aggregated_parameters[0]
                self.shared_state["num_examples"] = sum(r.num_examples for _, r in results)
            return aggregated_parameters

        # --- 2. MTGC Path (Weights Only) ---
        valid_results = []
        client_H_by_name = {}
        client_lr_by_name = {}

        for client, fit_res in results:
            # 1. Extract parameters (These are JUST weights now, no tail!)
            packed_params = parameters_to_ndarrays(fit_res.parameters)
            c_name = fit_res.metrics.get("client_name", "Unknown")
            
            # 2. Extract H and LR
            client_H_by_name[c_name] = fit_res.metrics.get("H", None)
            client_lr_by_name[c_name] = fit_res.metrics.get("lr", None)
            
            # 3. Pass clean FitRes to valid_results
            valid_results.append((client, fit_res))
        
        # --- Compute edge-level H_edge and lr_edge ---
        H_weighted_sum = 0.0
        lr_weighted_sum = 0.0
        weight_sum = 0.0

        for (client, fit_res) in valid_results:
            c_name = fit_res.metrics.get("client_name", "Unknown")
            H_i = client_H_by_name.get(c_name, None)
            lr_i = client_lr_by_name.get(c_name, None)

            if H_i is not None and lr_i is not None and H_i > 0 and lr_i > 0:
                w = float(fit_res.num_examples)
                H_weighted_sum += w * float(H_i)
                lr_weighted_sum += w * float(lr_i)
                weight_sum += w

        self.shared_state["H_edge"] = (H_weighted_sum / weight_sum) if weight_sum > 0 else None
        self.shared_state["lr_edge"] = (lr_weighted_sum / weight_sum) if weight_sum > 0 else None

        # --- Call Super (Weights Aggregation) ---
        aggregated_parameters = super().aggregate_fit(rnd, valid_results, failures)
        
        if aggregated_parameters is not None:
            self.shared_state["aggregated_model"] = aggregated_parameters[0]
            self.shared_state["num_examples"] = sum(r.num_examples for _, r in valid_results)
            print(f"[Edge Server] Aggregated model at global round {current_global_round}.")

        return aggregated_parameters


    def aggregate_evaluate(self, server_round, results, failures):
        """
        Manages Group-Level Evaluation.
        
        Functionality:
        1. evaluate(): Runs centralized evaluation of the Group Model on the Edge Server's 
           local test dataset (if available).
        2. aggregate_evaluate(): Averages the distributed evaluation metrics reported 
           by the inner leaf clients.
        3. Caches the final evaluation metrics into `shared_state["aggregated_eval"]` 
           so the EdgeClient can report them up to the Cloud.
        """
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
        """
        Manages Group-Level Evaluation.
        
        Functionality:
        1. evaluate(): Runs centralized evaluation of the Group Model on the Edge Server's 
           local test dataset (if available).
        2. aggregate_evaluate(): Averages the distributed evaluation metrics reported 
           by the inner leaf clients.
        3. Caches the final evaluation metrics into `shared_state["aggregated_eval"]` 
           so the EdgeClient can report them up to the Cloud.
        """
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
        """
        Configures and dispatches training instructions to the inner leaf clients.
        
        Functionality:
        1. Takes the current group model (or initial global model) and prepares to send it down.
        2. Fetches the pre-calculated group-global correction term (y_j) from `shared_state`.
        3. Compresses y_j (if compression is enabled) to save downlink bandwidth.
        4. Injects y_j, beta, and compression configs into the configuration dictionary.
        5. Logs the Downlink (Edge -> Client) traffic metrics.
        """
        print(f"[Edge Server] Configuring fit for round {server_round}...")
        current_global_round = self.round

        fit_instructions = super().configure_fit(
            server_round, parameters, client_manager, **kwargs
        )
        if not fit_instructions: return []

        current_weights = parameters_to_ndarrays(parameters)
        prev_weights = self.shared_state.get("fedmut_history_for_round", None)
        # base_weights = parameters_to_ndarrays(parameters)[: self.num_model_layers]
        
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

        # zi_per_client = self.shared_state.get("zi_per_client", {})
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
                c_name = res.properties.get("client_name", "unknown")
            except Exception as e:
                print(f"[Edge Server] Failed to query properties from client {client}: {e}")

            client_parameters = fit_ins.parameters
            if use_mutation:
                # client_weights_arrays = mutated_weights_list[i]
                current_weights = mutated_weights_list[i]
                client_parameters = ndarrays_to_parameters(mutated_weights_list[i])
            
            cfg = fit_ins.config.copy()
            cfg.update({
                "round": server_round,
                "yi": final_yi_blob,
                "yi_compressed": final_yi_compressed,
                "beta": beta,
                # "cid": cid,
                "compression_method": COMPRESSION_METHOD,
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
                zi_tuple=(0, 0),
                comp_time= yi_comp_time,
            )
            self.traffic_logger.log(metrics)

            new_fit_instructions.append((client, FitIns(client_parameters, cfg)))
            
        # del zi_per_client, yi_blob, final_yi_blob
        del yi_blob, final_yi_blob
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
            self.yj = {} 
            
            self.prev_group_model = None # To store \bar{x}_j^{t,E}
            self.prev_H_edge = None
            self.prev_lr_edge = None

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
            """
            The crucial MTGC bridge between the Cloud Server and the Inner FL loop.
            
            Functionality:
            1. Receives the brand-new Global Model from the Cloud.
            2. MTGC Group Drift (y_j) Update: Compares the `last_group_model` (from the end of 
            the previous round) to the `new_global_model`. Updates y_j based on this drift.
            3. Saves the global model and y_j into `shared_state`.
            4. Spawns the inner Edge Server as a separate multiprocessing Process to run E rounds.
            5. Waits for the inner server to finish, retrieves the final Group Model, and 
            sends it up to the Cloud Server.
            6. Implements a safe fallback (returns 0 examples) if the inner aggregation fails.
            """
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

                if GRADIENT_CORRECTION_BETA == 1.0:
                    if self.prev_group_model is not None:
                        print(f"[Edge Client] 🔄 MTGC: Updating Local Y_j...")
                        
                        # Use History
                        H = self.prev_H_edge if self.prev_H_edge else 1.0
                        lr = self.prev_lr_edge if self.prev_lr_edge else 0.01
                        scale = 1.0 / (H * lr) # E is 1
                        
                        # Helper to map list -> dict
                        model_module = importlib.import_module(f"models.{MODEL}")
                        ref_net = model_module.Net()
                        param_keys = [n for n, p in ref_net.named_parameters()]
                        
                        global_weights_dict = dict(zip(param_keys, parameters))

                        # Init Yj if first update
                        if not self.yj:
                            self.yj = {k: np.zeros_like(v) for k,v in self.prev_group_model.items()}

                        # Update Loop
                        for name in self.yj:
                            if name in self.prev_group_model and name in global_weights_dict:
                                # Drift = Previous Group Model - New Global Model
                                drift = self.prev_group_model[name] - global_weights_dict[name]
                                self.yj[name] = self.yj[name] + (scale * drift)
                        
                        print(f"[Edge Client] ✅ Y_j updated. H={H}, lr={lr}")
                    
                    # Store Yi in shared_state for EdgeStrategy to broadcast
                    self.shared_state["yi"] = pickle.dumps(self.yj) 
                    self.shared_state["yi_is_compressed"] = False

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

                    if GRADIENT_CORRECTION_BETA == 1.0:
                        # 1. Save History for NEXT round's drift calculation
                        model_module = importlib.import_module(f"models.{MODEL}")
                        ref_net = model_module.Net()
                        param_keys = [n for n, p in ref_net.named_parameters()]
                        
                        self.prev_group_model = dict(zip(param_keys, edge_weights))
                        self.prev_H_edge = self.shared_state.get("H_edge", 1.0)
                        self.prev_lr_edge = self.shared_state.get("lr_edge", 0.01)
                        
                        # 2. Return Weights (No Gradients)
                        metrics = {
                            "is_compressed": False,
                            "client_name": args.name,
                            "model_length": len(edge_weights),
                            "H_edge": float(self.shared_state.get("H_edge", 1.0)),
                            "lr": float(self.shared_state.get("lr_edge", 0.01)),
                            "has_grads": False,
                        }
                        
                        self.shared_state["aggregated_model"] = None
                        gc.collect()
                        return edge_weights, num_examples, metrics

                    # === OTHER BETA RETURN PATH (Weights + Gradients) ===
                    else:
                        H_edge = self.shared_state.get("H_edge", None)
                        lr_edge = self.shared_state.get("lr_edge", None)
                    
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
                        else:
                            grad_list_padded = []
                            if grad_dict_to_send:
                                for name, p in ref_net.named_parameters():
                                    if name in grad_dict_to_send:
                                        grad_list_padded.append(grad_dict_to_send[name])
                                    else:
                                        grad_list_padded.append(np.zeros_like(p.detach().cpu().numpy()))
                                    
                            payload_tail = grad_list_padded
                        metrics = {
                                "is_compressed": COMPRESSION_METHOD != "none",
                                "model_length": len(edge_weights),
                                "client_name": args.name,
                                "H_edge": H_edge,
                                "lr": lr_edge,
                                "E": 1,   # optional but nice for debugging
                            }
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
                    # This tells Central Server: "I am alive, but I have no update."
                    print(f"[Edge Client {args.name}] ⚠️ Aggregation Failed (All inner clients failed). Returning No-Op to keep connection alive.")
                    return parameters, 0, {"is_compressed": False, "status": "FAILED_NO_CLIENTS"}
                    
            except Exception as e:
                print(f"[Edge Client {args.name}] ❌ CRASH IN FIT: {e}")
                traceback.print_exc()
                # If we crash here, we must raise so Central knows.
                # But since we want to avoid exiting, we can try returning No-Op too.
                # But a crash usually means something worse. Let's raise.
                raise e
        
        def evaluate(self, parameters, config):
            """
            Reports evaluation metrics back to the Cloud.
            
            Functionality:
            1. Retrieves the group-level evaluation metrics that were calculated by the 
            inner FL server and stored in `shared_state`.
            2. Returns these metrics to the Cloud so the Central Server can track the 
            Edge group's performance.
            3. Contains a fallback to run evaluation locally if the inner server failed to yield metrics.
            """
            round_num = config.get("round", 0)

            # Fast path: if EdgeStrategy stored aggregated eval metrics in shared_state, return them.
            loss = self.shared_state.get("aggregated_loss", None)
            acc  = self.shared_state.get("aggregated_accuracy", None)
            n    = int(self.shared_state.get("num_examples", 0) or 0)
            if loss is not None and acc is not None and n > 0:
                return float(loss), n, {"accuracy": float(acc)}

            # Fallback: run evaluation locally (full test set)
            model_module = importlib.import_module(f"models.{MODEL}")
            net = model_module.Net()
            set_parameters(net, parameters)
            _, _, testloader = load_datasets()
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

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
