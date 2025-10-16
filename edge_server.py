import importlib
import json
import sys
import traceback
import flwr as fl
from flwr.server import ServerConfig
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns
import numpy as np
import multiprocessing
import argparse
from logger import Logger
from utils import load_datasets, set_parameters, test, log_to_dashboard
from config import MODEL, MIN_CLIENTS_PER_EDGE, GRADIENT_CORRECTION_BETA

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

args = parser.parse_args()

logger = Logger(
    subfolder="edge",
    file_path=f"{args.name}.log",
    headers=["round", "loss", "accuracy"],
    init_file=False,
)


class EdgeStrategy(fl.server.strategy.FedAvg):
    def __init__(self, shared_state, round, **kwargs):
        super().__init__(**kwargs)
        self.shared_state = shared_state
        self.round = round

    def aggregate_fit(self, rnd, results, failures):
        print(f"[Edge Server {args.name}] Aggregating fit results at round {rnd}.")
        
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_parameters is not None:
            self.shared_state["aggregated_model"] = aggregated_parameters
            examples = [r.num_examples for _, r in results]
            self.shared_state["num_examples"] = sum(examples)

            clients = [client for client, _ in results]
            # Already lists from client JSON
            client_grads = [json.loads(r.metrics["gradients"]) for _, r in results]

            # Compute average gradient (still NumPy temporarily)
            avg_grad = {}
            for name in client_grads[0]:
                avg_grad[name] = np.mean([cg[name] for cg in client_grads], axis=0)

            # Convert avg_grad to lists for shared_state
            avg_grad_serializable = {name: grad.tolist() for name, grad in avg_grad.items()}

            # Compute zi_per_client as differences
            zi_per_client = {}
            for client, grads in zip(clients, client_grads):
                client_id_str = getattr(client, "cid", str(client))
                zi_per_client[client_id_str] = {name: (avg_grad[name] - grads[name]).tolist()
                                    for name in grads}

            # Store safely in shared_state
            # print(zi_per_client, avg_grad_serializable)
            self.shared_state["zi_per_client"] = json.dumps(zi_per_client)
            self.shared_state["group_avg_grad"] = json.dumps(avg_grad_serializable)

            print(f"[Edge Server] Computed zi for {len(zi_per_client)} clients.")
            print(f"[Edge Server] Aggregated model at round {rnd}.")

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
        return super().evaluate(server_round, parameters)


    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        """Send per-client zi and global yi to clients."""
        print(f"[Edge Server] Configuring fit for round {server_round}...")

        # Let the base FedAvg select the clients
        fit_instructions = super().configure_fit(
            server_round, parameters, client_manager, **kwargs
        )

        # yi is received from central server, stored in shared_state
        yi = self.shared_state.get(
            "yi", {name: 0.0 for name in self.shared_state.get("group_avg_grad", {})}
        )

        if isinstance(yi, str):
            yi = json.loads(yi)


        # zi values computed in aggregate_fit
        zi_per_client = self.shared_state.get("zi_per_client", {})
        if isinstance(zi_per_client, str):
            zi_per_client = json.loads(zi_per_client)

        beta = GRADIENT_CORRECTION_BETA

        new_fit_instructions = []

        for client, fit_ins in fit_instructions:
            cid = getattr(client, "cid", None)

            # Default zi if missing
            if zi_per_client:
                client_zi = zi_per_client.get(
                    cid, {name: 0.0 for name in zi_per_client[next(iter(zi_per_client))]}
                )
            else:
                client_zi = {name: 0.0 for name in yi}

            cfg = fit_ins.config.copy()
            cfg.update({
                "round": server_round,
                "zi": json.dumps(client_zi),
                "yi": json.dumps(yi),
                "beta": beta,
                "cid": cid,
            })

            new_fit_instructions.append((client, FitIns(fit_ins.parameters, cfg)))

        print(f"Prepared {len(new_fit_instructions)} fit instructions")
        return new_fit_instructions

def run_edge_server(shared_state, params, round):
    strategy = EdgeStrategy(
        shared_state,
        round,
        min_fit_clients=MIN_CLIENTS_PER_EDGE,
        min_available_clients=MIN_CLIENTS_PER_EDGE,
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
            return [np.array([0.0, 0.0, 0.0])]

        def fit(self, parameters, config):
            print(f"[Edge Client {args.name}] Received model from central server.")
            # print(config)

            self.shared_state["yi"] = config["yi"] # do not json.loads this right now, it will be loaded later

            # Start the edge server process for local aggregation
            server_process = multiprocessing.Process(
                target=run_edge_server,
                args=(self.shared_state, parameters, config["round"]),
            )
            server_process.start()
            server_process.join()

            agg_model = self.shared_state.get("aggregated_model")

            if agg_model is not None:
                num_examples = self.shared_state.get("num_examples")
                res = parameters_to_ndarrays(agg_model[0])
                print(f"[Edge Client {args.name}] Sending model to central server.")
                group_avg_grad = self.shared_state.get("group_avg_grad")
                return res, num_examples, {"gradients": group_avg_grad}
            else:
                # something broke
                default = [np.array([0.0, 0.0, 0.0, 0.0])]
                return default, 1, {}

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
