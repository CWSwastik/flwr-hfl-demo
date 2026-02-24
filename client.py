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


from torch.optim import Adam, SGD
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
        """
        Initializes a stateful FL Client for MTGC.
        
        Functionality:
        1. Inherits from flwr.client.NumPyClient to integrate with the Flower framework.
        2. Initializes an empty tensor dictionary for $z_i$ (client-group drift) that matches 
           the exact architecture of the neural network.
        3. Sets up PyTorch Optimizers and a Learning Rate Scheduler to decay the learning rate 
           across global rounds.
        4. Initializes `prev_local_weights`, `prev_H`, and `prev_lr` to securely hold the 
           client's state between rounds, which is mathematically required to calculate $z_i$.
        """
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.round = 1
        self.zi = {
            name: torch.zeros_like(p, requires_grad=False, device=DEVICE) 
            for name, p in self.net.named_parameters()
        }
        
        # To store the model we sent in the previous round
        self.prev_local_weights = None 
        self.prev_H = None
        self.prev_lr = None
        self.experiment_ended = False

        # self.optimizer = Adam(
        #     self.net.parameters(),
        #     lr=TRAINING_LEARNING_RATE,
        #     weight_decay=TRAINING_WEIGHT_DECAY,
        # )
        self.optimizer = SGD(
            self.net.parameters(),
            lr=TRAINING_LEARNING_RATE,
            weight_decay=TRAINING_WEIGHT_DECAY,)

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
        """
        Executes Local Training with Multi-Timescale Gradient Correction (MTGC).
        
        Functionality:
        1. Fast Path (Beta=0): Bypasses all correction math for standard algorithms.
        2. MTGC Drift ($z_i$): Calculates the difference between the client's previous 
           local model and the newly received group model to update $z_i$.
        3. Local SGD: Injects both $z_i$ (local drift) and $y_j$ (global drift) into 
           the local gradient steps.
        4. State Caching: Saves the new local model, H, and lr for the next round's math.
        5. Payload Construction: Smartly packs only weights (for MTGC) or compresses 
           gradients (for baselines) to upload to the Edge Server.
        """
        try:
            compression_method = config.get("compression_method", COMPRESSION_METHOD)

            if not np.all(parameters[0] == 0):
                set_parameters(self.net, parameters)
            else:
                print("Received initial model from server, starting training...")

            yi_blob = config.get("yi", b"")
            
            def decode_cv(blob):
                if not blob: return {}
                try:
                    data = unpack_compressed_data(blob)
                    return decompress_model_update(data)
                except Exception as e:
                    print(f"Error decoding CV blob: {e}")
                    try:
                        return pickle.loads(blob)
                    except:
                        return {}

            yi = decode_cv(yi_blob)
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
                
                # FIX: Add basic metrics so Edge Server logging doesn't break
                metrics_dict = {
                    "client_name": args.name,
                    "model_length": len(get_parameters(self.net)),
                    "is_compressed": False,
                    "has_grads": False
                }
                return get_parameters(self.net), len(self.trainloader.dataset), metrics_dict

            # --- Gradient Correction Setup ---
            device = next(self.net.parameters()).device
            zi = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in self.zi.items()}
            yi = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in yi.items()}

            H_steps = 1
            # update zi by the below formula $$z_i^{t,e+1} = z_i^{t,e} + \frac{1}{H\gamma}(x_{i,H}^{t,e} - \overline{x}_j^{t,e+1})$$
            if self.prev_local_weights is not None:
                scale = 1.0 / (self.prev_H * self.prev_lr) if (self.prev_H and self.prev_lr) else 1.0
                
                with torch.no_grad():
                    for name, p in self.net.named_parameters():
                        if name in self.zi and name in self.prev_local_weights:
                            drift = self.prev_local_weights[name].to(DEVICE) - p.data
                            self.zi[name].add_(scale * drift)
                            zi = self.zi

            losses, accuracies, gradients = None, None, None
            if TRAINING_STRATEGY == "fedprox":
                losses, accuracies, gradients, H_steps = train_fedprox_with_zi_yi(
                    self.net, self.trainloader, self.optimizer,
                    epochs=local_epochs, beta=beta, zi=zi, yi=yi
                )
            else:
                losses, accuracies, gradients, H_steps = train_with_zi_yi(
                    self.net, self.trainloader, self.optimizer,
                    epochs=local_epochs, beta=beta, zi=zi, yi=yi
                )
            
            # Step the scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Cache state for next round's math
            if beta == 1.0:
                self.prev_local_weights = {
                    name: p.detach().clone().cpu() 
                    for name, p in self.net.named_parameters()
                }
                self.prev_H = H_steps
                self.prev_lr = current_lr

            # --- Logging ---
            train_logger.log({
                "round": self.round,
                "loss": losses[0],
                "accuracy": accuracies[0],
                "data_samples": len(self.trainloader.dataset),
            })

            # --- Payload Construction ---
            model_params_list = get_parameters(self.net)
            grads_dict = {}
            for name, p in self.net.named_parameters():
                if gradients is not None and name in gradients:
                    grads_dict[name] = gradients[name].cpu().numpy()
            
            model_u = get_payload_size(model_params_list)
            model_c = model_u

            grads_u = get_payload_size(grads_dict)
            grads_c = grads_u
            comp_time = 0.0
            payload_tail = []

            # Only compress if the method is not 'none' AND we actually have gradients!
            if compression_method != "none" and grads_dict:
                importance_scores = None
                if compression_method == "shap":
                    importance_scores = get_gradient_shap_importance(self.net, self.trainloader, DEVICE)
                elif compression_method == "fisher":
                    importance_scores = get_fisher_importance(self.net, self.trainloader, DEVICE)

                t_start = time.time()
                compressed_grads_dict = compress_model_update(grads_dict, importance_scores=importance_scores)
                packed_grads_blob = pack_compressed_data(compressed_grads_dict)
                
                payload_tail = [packed_grads_blob]
                
                grads_c = get_payload_size(compressed_grads_dict)
                comp_time = time.time() - t_start
                
                print(f"Compressed: {grads_u/1024:.1f}KB -> {grads_c/1024:.1f}KB")
            else:
                grads_list = []
                if grads_dict:
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
                "is_compressed": compression_method != "none" and bool(grads_dict),
                "client_name": args.name,
                "H": H_steps,
                "E": 1,
                "lr": current_lr,
            }

            return packed_params, len(self.trainloader.dataset), metrics_dict
        
        except Exception as e:
            print(f"❌ CLIENT EXCEPTION IN FIT: {e}")
            traceback.print_exc()
            raise e

    def evaluate(self, parameters, config):
        """
        Executes local evaluation of the provided model on the client's validation set.
        
        Functionality:
        1. Injects the received parameters into the local neural network.
        2. Runs a standard PyTorch inference loop on the local validation dataset.
        3. Logs the loss and accuracy to the client's local `.log` file and pushes the 
           metrics to the central dashboard.
        4. Returns the metrics to the Edge Server so they can be aggregated.
        """
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
            print(f"Warning: {type(e)}, Couldn't run client. Retrying in 5 seconds...")
        time.sleep(5)