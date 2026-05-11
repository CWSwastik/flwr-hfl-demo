import os
# Must be set before importing numpy/torch
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"

import json
import flwr as fl
from flwr.common import GetPropertiesIns, GetPropertiesRes, Status, Code
import numpy as np
import time
import argparse
import requests
import torch
import traceback
import sys
import random

import config
import pickle
from utils import (
    decompress_model_update,
    get_fisher_importance,
    load_datasets,
    get_parameters,
    set_parameters,
    train,
    train_with_zi_yi,
    train_fedprox,
    train_fedprox_with_zi_yi,
    test,
    DEVICE,
    get_dataloader_summary,
    post_to_dashboard,
    log_to_dashboard,
    compress_model_update,
    get_payload_size,
    unpack_compressed_data,
    pack_compressed_data,
    get_traffic_metrics, 
    get_gradient_shap_importance,
)
from config import (
    NUM_ROUNDS,
    MODEL,
    GRADIENT_CORRECTION_BETA,
    TRAINING_LEARNING_RATE,
    TRAINING_WEIGHT_DECAY,
    TRAINING_SCHEDULER_GAMMA,
    TRAINING_SCHEDULER_STEP_SIZE,
    TRAINING_STRATEGY,
    FedProx_MU,
    DASHBOARD_SERVER_URL,
    ENABLE_DASHBOARD,
    EXPERIMENT_NAME,
    LOCAL_EPOCHS,
    SEED,
    COMPRESSION_METHOD, 
)

import importlib
from logger import Logger


from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser(description="Start a Flower client.")
parser.add_argument(
    "server_address",
    help="Server address in the format host:port (e.g., localhost:8081)",
)
parser.add_argument("--partition_id", type=int, default=0, help="Partition ID")
parser.add_argument(
    "--name", type=str, default="client", help="Client name (default: client)"
)
parser.add_argument(
    "--exp_id",
    type=str,
    help="The experiment ID for the dashboard",
)

args = parser.parse_args()

test_logger = Logger(
    subfolder="clients",
    file_path=f"{args.name}_{MODEL}_test.log",
    headers=["round", "loss", "accuracy", "data_samples"],
)

train_logger = Logger(
    subfolder="clients",
    file_path=f"{args.name}_{MODEL}_train.log",
    headers=["round", "loss", "accuracy", "data_samples"],
)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.round = 1

        self.optimizer = Adam(
            self.net.parameters(),
            lr=TRAINING_LEARNING_RATE,
            weight_decay=TRAINING_WEIGHT_DECAY,
        )

        self.scheduler = StepLR(
            self.optimizer,
            step_size=TRAINING_SCHEDULER_STEP_SIZE,
            gamma=TRAINING_SCHEDULER_GAMMA,
        )
        self.traffic_logger = Logger(
            subfolder="clients",
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
        """Allows the server to query the client's name."""
        return {
            "client_name": args.name  # Return the static name (e.g., 'client_1')
        }
    
    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        try:
            compression_method = config.get("compression_method", COMPRESSION_METHOD)

            if not np.all(parameters[0] == 0):
                set_parameters(self.net, parameters)
            else:
                print("Received initial model from server, starting training...")

            yi_blob = config.get("yi", b"")
            zi_blob = config.get("zi", b"")
            
            # --- FIX: Robust Decoding ---
            def decode_cv(blob):
                if not blob: return {}
                try:
                    # 1. Unpack bytes/array -> dictionary
                    data = unpack_compressed_data(blob)
                    
                    # 2. Smart Decompression
                    # decompress_model_update in utils.py automatically checks 
                    # if the dict has "method" and "layers". If so, it decompresses.
                    # If not, it returns the data as-is (uncompressed).
                    return decompress_model_update(data)
                except Exception as e:
                    print(f"Error decoding CV blob: {e}")
                    # Fallback
                    try:
                        return pickle.loads(blob)
                    except:
                        return {}

            yi = decode_cv(yi_blob)
            zi = decode_cv(zi_blob)

            beta = config.get("beta", GRADIENT_CORRECTION_BETA)
            local_epochs = config.get("local_epochs", LOCAL_EPOCHS)

            # --- No Gradient Correction ---
            if beta == 0:
                if TRAINING_STRATEGY == "fedavg" or TRAINING_STRATEGY == "fedmut":
                    losses, accuracies = train(
                        self.net, self.trainloader, self.optimizer, epochs=local_epochs
                    )
                elif TRAINING_STRATEGY == "fedprox":
                    losses, accuracies = train_fedprox(
                        self.net, self.trainloader, self.optimizer, epochs=local_epochs, mu=FedProx_MU,
                    )
                return get_parameters(self.net), len(self.trainloader.dataset), {}

            # --- Gradient Correction Setup ---
            device = next(self.net.parameters()).device
            zi = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in zi.items()}
            yi = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in yi.items()}

            losses, accuracies, gradients = None, None, None
            if TRAINING_STRATEGY == "fedprox":
                losses, accuracies, gradients = train_fedprox_with_zi_yi(
                    self.net, self.trainloader, self.optimizer,
                    epochs=local_epochs, beta=beta, zi=zi, yi=yi
                )
            else:
                losses, accuracies, gradients = train_with_zi_yi(
                    self.net, self.trainloader, self.optimizer,
                    epochs=local_epochs, beta=beta, zi=zi, yi=yi
                )

            # --- Logging ---
            train_logger.log({
                "round": self.round,
                "loss": losses[0],
                "accuracy": accuracies[0],
                "data_samples": len(self.trainloader.dataset),
            })

            # --- Prepare Response ---
            model_params_list = get_parameters(self.net)
            grads_dict = {}
            for name, p in self.net.named_parameters():
                if gradients is not None and name in gradients:
                    grads_dict[name] = gradients[name].cpu().numpy()
            
            model_u = get_payload_size(model_params_list)
            model_c = model_u
            before_quantization = get_payload_size(grads_dict)

            grads_u = get_payload_size(gradients)
            grads_c = grads_u
            comp_time = 0.0
            payload_tail = []

            importance_scores = None
            if COMPRESSION_METHOD == "shap":
                importance_scores = get_gradient_shap_importance(self.net, self.trainloader, DEVICE)
            elif COMPRESSION_METHOD == "fisher":
                importance_scores = get_fisher_importance(self.net, self.trainloader, DEVICE)

            # --- Compress ---
            if COMPRESSION_METHOD != "none":
                t_start = time.time()
                compressed_grads_dict = compress_model_update(grads_dict, importance_scores=importance_scores)
                packed_grads_blob = pack_compressed_data(compressed_grads_dict)
                
                payload_tail = [packed_grads_blob]
                
                # after_quantization = get_payload_size(compressed_grads_dict)
                grads_c = get_payload_size(packed_grads_blob)
                comp_time = time.time() - t_start
                
                print(f"Compressed: {before_quantization/1024:.1f}KB -> {grads_c/1024:.1f}KB")
            else:
                grads_list = []
                for name, p in self.net.named_parameters():
                    if name in grads_dict:
                        grads_list.append(grads_dict[name])
                    else:
                        grads_list.append(np.zeros_like(p.data.cpu().numpy()))
                payload_tail = grads_list

            packed_params = model_params_list + payload_tail

            metrics = get_traffic_metrics(
                round_num=config["round"],
                direction="Uplink",
                model_tuple=(model_u, model_c),
                grad_tuple=(grads_u, grads_c),
                comp_time=comp_time
            )
            self.traffic_logger.log(metrics)
            metrics_dict = {
            "model_length": len(model_params_list), 
            "is_compressed": compression_method != "none",
            "client_name": args.name  # <--- CRITICAL: Send Identity
            }

            return packed_params, len(self.trainloader.dataset), metrics_dict
        
        except Exception as e:
            print(f"❌ CLIENT EXCEPTION IN FIT: {e}")
            traceback.print_exc()
            raise e

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        test_logger.log(
            {
                "round": self.round,
                "loss": loss,
                "accuracy": accuracy,
                "data_samples": len(self.valloader.dataset),
            }
        )
        if ENABLE_DASHBOARD:
            log_to_dashboard(
                args.exp_id,
                "client",
                {
                    "device": args.name,
                    "round": self.round,
                    "loss": loss,
                    "accuracy": accuracy,
                    "data_samples": len(self.valloader.dataset),
                },
            )
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


def create_client(partition_id, model) -> fl.client.Client:

    model_module = importlib.import_module(f"models.{model}")
    net = model_module.Net().to(DEVICE)

    trainloader, valloader, testloader = load_datasets(partition_id=partition_id)

    return FlowerClient(net, trainloader, valloader)


if __name__ == "__main__":
    print(
        f"Starting client {args.name} with partition_id {args.partition_id} and connecting to {args.server_address}"
    )
    client = create_client(args.partition_id, model=MODEL)
    while client.round <= NUM_ROUNDS:
        try:
            print(f"Starting client {args.name} for Round {client.round}")
            fl.client.start_client(
                server_address=args.server_address, client=client.to_client()
            )
            client.round += 1
        except Exception as e:
            print(f"Error: {type(e)}, Couldn't run client. Retrying in 5 seconds...")

        time.sleep(5)