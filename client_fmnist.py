# %%
import os
import time
import torch
import psutil
import argparse
import flwr as fl
import numpy as np
import fashion_mnist
from collections import OrderedDict
from typing import Dict, List, Tuple
from utils import memory_usage, get_gpu_power_consume
from flops_profiler.profiler import get_model_profile
from torch.utils.data import TensorDataset, DataLoader

# Defining the profiler
num_clients = 6
worker_id = 1
device = 0
server_IP = 'localhost:8080'

beta = 'iid'
lista_tempi = []
folder_path = os.getcwd()
_parser = argparse.ArgumentParser(
    prog="client",
    description="Run the client.",
)

_parser.add_argument('--worker_id', type=int, default=worker_id)
_parser.add_argument('--device', type=int, default=device)
_parser.add_argument('--server_IP', type=str, default=server_IP)
args = _parser.parse_known_args()[0]
worker_id = args.worker_id
server_IP = args.server_IP
CVD = args.device

DEVICE: str = torch.device(f"cuda:{CVD}" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


class FMnistClient(fl.client.NumPyClient):
    """Flower client implementing FashionMNIST image classification using
    PyTorch."""

    def __init__(
        self,
        model: fashion_mnist.Net,
        trainloader: DataLoader,
        testloader: DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
            self, parameters: List[np.ndarray], config: Dict[str, str]
            ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        print("Training local model")
        t0 = time.perf_counter()
        fashion_mnist.train(self.model, self.trainloader, epochs=5, device=DEVICE, worker_id=worker_id) #, prof=prof)
        t1 = time.perf_counter()
        lista_tempi.append(t1-t0)

        print("Training done")
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)    
        loss, accuracy = fashion_mnist.test(self.model, self.testloader, device=DEVICE)

        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def main(beta, num_clients) -> None:
    """Load data, start FMnistClient."""
    # Load model and data
    model = fashion_mnist.Net()
    model.to(DEVICE)
    _, testloader, num_examples = fashion_mnist.load_data()
    import pickle
    if beta == 'iid':
        with open(f'{folder_path}/fmnist_client/iid/num_client={num_clients}/X_train.pkl', 'rb') as f:
            X = pickle.load(f)[worker_id]
        with open(f'{folder_path}/fmnist_client/iid/num_client={num_clients}/y_train.pkl', 'rb') as f:
            y = pickle.load(f)[worker_id]
    else:
        with open(f'{folder_path}/fmnist_client/alpha={beta}/X_train.pkl', 'rb') as f:
            X = pickle.load(f)[worker_id]
        with open(f'{folder_path}/fmnist_client/alpha={beta}/y_train.pkl', 'rb') as f:
            y = pickle.load(f)[worker_id]

    # Create data loader
    batch_size = 32
    num_examples['trainset'] = len(X)
    trainset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Start client
    client = FMnistClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address=server_IP, client=client)
    with torch.cuda.device(CVD):
        flops, macs, params = get_model_profile(model=model,
                                                input_shape=(batch_size, 1, 28, 28),
                                                args=None,
                                                kwargs=None,
                                                print_profile=True,
                                                detailed=True,
                                                module_depth=-1,
                                                top_modules=1,
                                                warm_up=10,
                                                as_string=True,
                                                output_file=None,
                                                ignore_modules=None,
                                                func_name='forward')
        print(f'DEVICE: {DEVICE}')
        print(f"PARAMS: {params}")
        print(f"flops: {flops}")
        print(f"MACS: {macs}")
        tempo_medio = np.mean(lista_tempi[1:])
        print(f'Tempi per client {worker_id}: ', tempo_medio)

    return flops, macs, params, num_examples, tempo_medio


if __name__ == "__main__":
    pid_client = os.getpid()
    process = psutil.Process(os.getpid())
    print(process.pid, pid_client)
    t1 = time.perf_counter()
    flops, macs, params, num_examples, tempo_medio = main(beta, num_clients)
    flops = float(flops[:-2])
    macs = float(macs[:-5])
    params = float(params[:-1])
    t2 = time.perf_counter()
    execution_time = t2 - t1
    print("time: ", execution_time)
    rss, vms, data, shared, text, lib, dirty, uss, pss, swap, gpu_memory = memory_usage(process, device=CVD)
    power_draw = get_gpu_power_consume(device=CVD)
    power_draw = float(power_draw)
    
    os.makedirs(f'Results_{server_IP}_{num_clients}', exist_ok=True)
    with open(f'Results_{server_IP}_{num_clients}/Client{worker_id}_{beta}_performance.txt', 'w') as f:
        f.write(f'Server IP: {server_IP.split(":")[0]}\n')
        f.write(f'Port: {server_IP.split(":")[1]}\n')
        f.write(f'Client: {worker_id}\n')
        f.write(f'PID: {pid_client}\n')
        f.write(f'GPU: {CVD}\n')
        f.write(f'params: {params} k\n')
        f.write(f'flops: {flops} M\n')
        f.write(f'MACS: {macs} MMACs\n')
        f.write(f'Execution time: {tempo_medio} seconds\n')
        f.write(f'Memory Usage RSS: {rss} MB\n')
        f.write(f'Memory Usage VMS: {vms} MB\n') 
        f.write(f'Memory Usage DATA: {data} MB\n')
        f.write(f'Memory Usage SHARED: {shared} MB\n')
        f.write(f'Memory Usage TEXT: {text} MB\n') 
        f.write(f'Memory Usage LIB: {lib} MB\n') 
        f.write(f'Memory Usage DIRTY: {dirty} MB\n') 
        f.write(f'Memory Usage USS: {uss} MB\n') 
        f.write(f'Memory Usage PSS: {pss} MB\n') 
        f.write(f'Memory Usage SWAP: {swap} MB\n') 
        f.write(f'GPU Memory Usage: {gpu_memory} MB\n')
        f.write(f'GPU Power Consumption: {power_draw} W\n')
        f.write('\n')
        f.write(f'Non iid : {beta}\n')
        f.write(f'Number of examples: {num_examples}\n')
        f.write(f'THROUGHPUT: {flops/execution_time} FLOPS\n')
        f.write(f'ENERGY EFFICIENCY: {flops/execution_time/power_draw} FLOPS/W\n')