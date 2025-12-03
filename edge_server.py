import importlib
import json
import sys
import traceback
import flwr as fl
from flwr.server import ServerConfig
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns, FitRes, Parameters, Status, Code
import numpy as np
import multiprocessing
import argparse
from logger import Logger
from utils import load_datasets, set_parameters, test, log_to_dashboard, get_parameters
from config import MODEL, MIN_CLIENTS_PER_EDGE, GRADIENT_CORRECTION_BETA, ENABLE_DASHBOARD, SEED
import gc
import pickle

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
        self.grad_names = [n for n, p in ref_net.named_parameters() if p.requires_grad]

    def aggregate_fit(self, rnd, results, failures):
        print(f"[Edge Server {args.name}] Aggregating fit results at round {rnd}.")
        
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

        # intercepting results to extract gradients
        valid_results = []
        client_grads = []
        clients_list = []

        for client, fit_res in results:
            # 1. Convert packed parameters to NumPy
            packed_params = parameters_to_ndarrays(fit_res.parameters)
            
            # 2. SLICE the list [ Weights (0 to N) | Gradients (N to end) ]
            weights = packed_params[:self.num_model_layers]
            raw_grads = packed_params[self.num_model_layers:]
            
            # 3. Reconstruct Gradient Dictionary map the raw arrays back to their names: {'conv1.weight': array, ...}
            c_grad_dict = dict(zip(self.grad_names, raw_grads))
            client_grads.append(c_grad_dict)
            clients_list.append(client)
            
            # 4. Create a clean FitRes with ONLY weights for the standard FedAvg aggregator
            # This ensures the standard Flower logic doesn't get confused by the extra gradient data
            new_fit_res = FitRes(
                status=fit_res.status,
                parameters=ndarrays_to_parameters(weights), # Send only weights to super
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
            )
            valid_results.append((client, new_fit_res))
        
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_parameters is not None:
            self.shared_state["aggregated_model"] = aggregated_parameters[0]
            examples = [r.num_examples for _, r in results]
            self.shared_state["num_examples"] = sum(examples)

            # Compute average gradient (NumPy)
            avg_grad = {}
            for name in client_grads[0]:
                avg_grad[name] = np.mean([cg[name] for cg in client_grads], axis=0)

            # Compute zi_per_client as differences
            zi_per_client = {}
            for client, grads in zip(clients_list, client_grads):
                client_id_str = getattr(client, "cid", str(client))
                zi_per_client[client_id_str] = {name: (avg_grad[name] - grads[name]).tolist()
                                    for name in grads}

            # Store safely in shared_state
            # print(zi_per_client, avg_grad_serializable)
            self.shared_state["zi_per_client"] = pickle.dumps(zi_per_client)
            self.shared_state["group_avg_grad"] = pickle.dumps(avg_grad)
            
            print(f"[Edge Server] Computed zi for {len(zi_per_client)} clients.")
            print(f"[Edge Server] Aggregated model at round {rnd}.")

            del client_grads
            del valid_results
            del avg_grad
            del zi_per_client
            gc.collect()

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

        # print(parameters_to_ndarrays(parameters)[0][0][0][0])
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

        print(
            f"[Edge Server] Evaluate Round {server_round}: Loss = {loss}, Accuracy = {accuracy}"
        )
        # return super().evaluate(server_round, parameters) # This is returning None
        return float(loss), {"accuracy": float(accuracy)}


    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        """Send per-client zi and global yi to clients."""
        print(f"[Edge Server] Configuring fit for round {server_round}...")

        # Let the base FedAvg select the clients
        fit_instructions = super().configure_fit(
            server_round, parameters, client_manager, **kwargs
        )

        # yi is received from central server, stored in shared_state
        yi_blob = self.shared_state.get("yi", None)
        yi = {}
        if isinstance(yi_blob, bytes):
            yi = pickle.loads(yi_blob)
        else:
            yi = yi_blob if yi_blob is not None else {}

        zi_blob = self.shared_state.get("zi_per_client", None)
        zi_per_client = {}
        if zi_blob:
            zi_per_client = pickle.loads(zi_blob)
        
            

        beta = GRADIENT_CORRECTION_BETA

        new_fit_instructions = []

        for client, fit_ins in fit_instructions:
            cid = getattr(client, "cid", None)

            client_zi = zi_per_client.get(cid, None)
            if client_zi is None and zi_per_client:
                 # Fallback logic
                 first = next(iter(zi_per_client.values()))
                 client_zi = {k: np.zeros_like(v) for k,v in first.items()}
            elif client_zi is None:
                 client_zi = {}

            cfg = fit_ins.config.copy()

            def to_list_dict(d):
                return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()}

            cfg.update({
                "round": server_round,
                "zi": json.dumps(to_list_dict(client_zi)),
                "yi": json.dumps(to_list_dict(yi)),
                "beta": beta,
                "cid": cid,
            })

            new_fit_instructions.append((client, FitIns(fit_ins.parameters, cfg)))
            
        del zi_per_client
        del yi
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

        def get_parameters(self, config):
            if self.shared_state.get("aggregated_model") is not None:
                return parameters_to_ndarrays(self.shared_state["aggregated_model"])

            print(
                f"[Edge Client {args.name}] No aggregated model available yet. Returning 0s."
            )
            return [np.array([0.0])]
            # return [np.array([0.0, 0.0, 0.0])]


        def fit(self, parameters, config):
            print(f"[Edge Client {args.name}] Received model from central server.")
            
            # --- 1. OPTIMIZATION FOR NO-GC (beta=0) ---
            if GRADIENT_CORRECTION_BETA == 0:
                # No Yi needed. No Gradients needed.
                # Just pass standard config to Edge Server.
                server_process = multiprocessing.Process(
                    target=run_edge_server,
                    args=(self.shared_state, parameters, config["round"]),
                    daemon=True # Added Daemon for safety
                )
                server_process.start()
                server_process.join()
                
                # Retrieve Standard Aggregated Model
                agg_model = self.shared_state.get("aggregated_model")
                if agg_model is not None:
                    num_examples = self.shared_state.get("num_examples")
                    edge_weights = parameters_to_ndarrays(agg_model)
                    
                    # Cleanup
                    self.shared_state["aggregated_model"] = None
                    gc.collect()
                    
                    print(f"[Edge Client {args.name}] Sending STANDARD model (No GC).")
                    return edge_weights, num_examples, {}
                else:
                    return [], 0, {}

            # self.shared_state["yi"] = json.loads(config["yi"])

            yi_dict = json.loads(config.get("yi", "{}")) # Received as JSON from Central Server
            yi_numpy = {k: np.array(v) for k,v in yi_dict.items()}
            self.shared_state["yi"] = pickle.dumps(yi_numpy)

            # Start the edge server process for local aggregation
            server_process = multiprocessing.Process(
                target=run_edge_server,
                args=(self.shared_state, parameters, config["round"]),
                daemon=True,
            )
            server_process.start()
            server_process.join()

            agg_model = self.shared_state.get("aggregated_model")

            if agg_model is not None:
                num_examples = self.shared_state.get("num_examples")
                edge_weights = parameters_to_ndarrays(agg_model)
                
                # Load Gradients from Binary Blob
                grad_blob = self.shared_state.get("group_avg_grad")
                group_avg_grad_dict = pickle.loads(grad_blob)
                
                # Pack Weights + Gradients
                model_module = importlib.import_module(f"models.{MODEL}")
                ref_net = model_module.Net()
                grad_list = []
                for name, p in ref_net.named_parameters():
                    if p.requires_grad and name in group_avg_grad_dict:
                        # Convert list back to numpy if needed, or keep as list
                        grad_list.append(np.array(group_avg_grad_dict[name]))
                
                # 4. Pack Weights + Gradients
                packed_params = edge_weights + grad_list

                del edge_weights
                del group_avg_grad_dict
                self.shared_state["aggregated_model"] = None
                self.shared_state["group_avg_grad"] = None
                gc.collect()
                
                print(f"[Edge Client {args.name}] Sending PACKED model to central server.")
                
                # Return packed parameters. Metrics is empty!
                return packed_params, num_examples, {}
            else:
                # something broke
                return [], 0, {}
                # default = [np.array([0.0, 0.0, 0.0, 0.0])]
                # return default, 1, {}

        # def evaluate(self, parameters, config):
        #     num_examples = self.shared_state.get("num_examples")

        #     if config["round"] == 0:
        #         print("Skipping evaluation for round 0")
        #         return super().evaluate(parameters, config)

        #     print(f"[Edge Client] Evaluate round {config['round']}")
        #     net = Net()
        #     # print(parameters_to_ndarrays(parameters)[0][0][0][0])
        #     set_parameters(net, parameters)
        #     _, _, testloader = load_datasets()  # full dataset for evaluation
        #     loss, accuracy = test(net, testloader)
        #     logger.log(
        #         {
        #             "round": config["round"],
        #             "loss": loss,
        #             "accuracy": accuracy,
        #         }
        #     )

        #     print(
        #         f"[Edge Server] Evaluate Round {config['round']}: Loss = {loss}, Accuracy = {accuracy}"
        #     )

        #     return (
        #         float(loss),
        #         num_examples,
        #         {"accuracy": accuracy},
        #     )

    print(f"[Edge Client {args.name}] Connecting to central server {args.server}")
    fl.client.start_client(
        server_address=args.server, client=EdgeClient(shared_state).to_client()
    )


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
