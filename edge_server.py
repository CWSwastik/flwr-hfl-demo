import flwr as fl
from flwr.server import ServerConfig
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import numpy as np
import multiprocessing
import argparse


parser = argparse.ArgumentParser(description="Start a Flower Edge Server.")
parser.add_argument("--server", required=True, help="Central server address (e.g., localhost:8081)")
parser.add_argument("--client", required=True, help="Edge client address (e.g., localhost:8080)")
args = parser.parse_args()

class EdgeStrategy(fl.server.strategy.FedAvg):
    def __init__(self, shared_state, **kwargs):
        super().__init__(**kwargs)
        self.shared_state = shared_state

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            self.shared_state["aggregated_model"] = aggregated_parameters
            print(f"[Edge Server] Aggregated model at round {rnd}.")
        return aggregated_parameters
    
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        
        self.shared_state["aggregated_loss"] = aggregated_loss
        print(f"[Edge Server] Aggregated evaluation loss at round {server_round}: {aggregated_loss}")
    
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        aggregated_accuracy = sum(accuracies) / sum(examples)
        self.shared_state["aggregated_accuracy"] = aggregated_accuracy
        print(
            f"[Edge Server] Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}"
        )

        return float(aggregated_loss), {"accuracy": float(aggregated_accuracy)}
    
def run_edge_server(shared_state, params):
    strategy = EdgeStrategy(
        shared_state,
        min_fit_clients=2,
        min_available_clients=2,
        initial_parameters=ndarrays_to_parameters(
            params
        )
    )
    config = ServerConfig(num_rounds=1)
    
    print(f"[Edge Server] Starting on {args.client}")
    fl.server.start_server(
        server_address=args.client,
        strategy=strategy,
        config=config
    )

def run_edge_as_client(shared_state):
    class EdgeClient(fl.client.NumPyClient):
        def __init__(self, shared_state):
            self.shared_state = shared_state

        def get_parameters(self, config):
            if self.shared_state.get("aggregated_model") is not None:
                return parameters_to_ndarrays(self.shared_state["aggregated_model"[0]])
            
            print("[Edge Client] No aggregated model available yet. Returning 0s.")
            return [np.array([0.0, 0.0, 0.0])]



        def fit(self, parameters, config):
            print(f"[Edge Client] Received model from central server.")

            # Start the edge server process for local aggregation
            server_process = multiprocessing.Process(
                target=run_edge_server,
                args=(self.shared_state, parameters)
            )
            server_process.start()
            server_process.join()

            agg_model = self.shared_state.get("aggregated_model")
            
            if agg_model is not None:
                num_examples = 1
                res = parameters_to_ndarrays(agg_model[0])
                print(f"[Edge Client] Sending model to central server.")
                return res, num_examples, {}
            else:
                default = [np.array([0.0, 0.0, 0.0])]
                return default, len(default[0]), {}

        def evaluate(self, parameters, config):
            agg_loss = self.shared_state.get("aggregated_loss")
            agg_accuracy = self.shared_state.get("aggregated_accuracy")
            return agg_loss, 1, {"accuracy": agg_accuracy} # TODO: figure out num examples

    print(f"[Edge Client] Connecting to central server {args.server}")
    fl.client.start_client(
        server_address=args.server,
        client=EdgeClient(shared_state).to_client()
    )

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    shared_state["aggregated_model"] = None
    shared_state["aggregated_eval"] = None

    client_process = multiprocessing.Process(
        target=run_edge_as_client,
        args=(shared_state,)
    )
    client_process.start()
    client_process.join()
