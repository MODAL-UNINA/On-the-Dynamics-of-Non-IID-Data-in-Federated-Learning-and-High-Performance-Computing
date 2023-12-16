
# %%
import sys
import os
import torch
import psutil
import GPUtil
import subprocess
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

_interactive_mode = 'ipykernel_launcher' in sys.argv[0] or \
                   (len(sys.argv) == 1 and sys.argv[0] == '')


def is_interactive():
    return _interactive_mode


def myshow(plot_show=True):
    if _interactive_mode and plot_show:
        plt.show()
    else:
        plt.close()


def get_cpu_usage():
    return psutil.cpu_percent()


def get_cpu_times():
    return psutil.cpu_times_percent()


def get_memory_usage():
    return psutil.virtual_memory().percent


def get_swap_usage():
    return psutil.swap_memory().percent


def get_max_memory_allocated(device):
    max_memory_allocated = torch.cuda.max_memory_allocated(device=device)
    max_memory_allocated_MB = max_memory_allocated / (1024 * 1024)

    return max_memory_allocated_MB


def get_metrics(pid):
    cpu_usage = get_cpu_usage()
    cpu_times = get_cpu_times()
    memory_usage = get_memory_usage()
    swap_usage = get_swap_usage() 
    with open('metrics.txt', 'a') as f:
        f.write(f'cpu_usage: {cpu_usage}\n')
        f.write(f'cpu_times: {cpu_times}\n')
        f.write(f'memory_usage: {memory_usage}\n')
        f.write(f'swap_usage: {swap_usage}\n')
        f.write('\n')


def memory_usage(process=None, device=0):
    if process is None:
        process = psutil.Process(os.getpid())
    print('Process ID:', process.pid)
    memory_info = process.memory_full_info()
    memory_usage_bytes = memory_info.rss
    vmemory_usage_bytes = memory_info.vms
    processed_data = memory_info.data
    shared_memory = memory_info.shared
    text_memory = memory_info.text
    lib_memory = memory_info.lib
    dirty_memory = memory_info.dirty
    uss = memory_info.uss
    pss = memory_info.pss
    swap = memory_info.swap

    gpus = GPUtil.getGPUs()
    gpus = gpus[device]
    gpus_memory = gpus.memoryUsed
    print(f"GPU memory used: {gpus_memory} MB")

    return memory_usage_bytes / (1024 ** 2), vmemory_usage_bytes / (1024 ** 2), processed_data / (1024 ** 2), shared_memory / (1024 ** 2), text_memory / (1024 ** 2), lib_memory / (1024 ** 2), dirty_memory / (1024 ** 2), uss / (1024 ** 2), pss / (1024 ** 2), swap / (1024 ** 2), gpus_memory


def get_gpu_power_consume(device=0):
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    power_draw = result.stdout.decode('utf-8').strip().split('\n')[device]
    print(f"Consumo energetico: {power_draw} W")
    return power_draw

# %%


def separate_data(data, num_clients, num_classes, niid=False, 
                  least_samples=None, partition=None, alpha=0.1, balance=False, 
                  class_per_client=2, save_fig=None):
    X = {}
    y = {}
    statistic = {}

    dataset_content, dataset_label = data

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(num_clients/num_classes*class_per_client)]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            print(f"Minimum number of samples per client: {min_size}")
        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        statistic.setdefault(client, [])
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs].copy()
        y[client] = dataset_label[idxs].copy()

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))

    del data

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print("\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)
    print("Separate data finished!\n")

    ylabels = np.arange(num_clients)
    xlabels = np.arange(num_classes)

    x_mesh, y_mesh = np.meshgrid(np.arange(num_classes), np.arange(num_clients))
    s = np.zeros((num_clients, num_classes), dtype=np.uint16)
    for k_stat, v_stat in statistic.items():
        for elem in v_stat:
            s[k_stat, elem[0]] = elem[1]
    if not niid:
        c = np.ones((num_clients, num_classes), dtype=np.uint16)
        R = s/s.max()/3
        viridis_cmap = cm.get_cmap('viridis')
        new_start = 0.5
        new_end = 1.0
        cm_color = cm.colors.LinearSegmentedColormap.from_list(
            'viridis_cut', viridis_cmap(np.linspace(new_start, new_end, 256))
        )
    else: 
        c = s
        R = s/s.max()/1.5
        cm_color = 'viridis'

    fig, ax = plt.subplots(figsize=(0.4*num_classes, 0.4*num_clients))
    circles = [plt.Circle((j, i), radius=r) for r, j, i in zip(R.flat, x_mesh.flat, y_mesh.flat)]
    col = PatchCollection(circles, array=c.flatten(), cmap=cm_color, zorder=10)
    ax.add_collection(col)

    ax.set(title='Number of samples per class per client', xlabel='Classes', ylabel='Client ID')
    ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_clients),
        xticklabels=xlabels, yticklabels=ylabels)
    ax.set_xticks(np.arange(num_classes+1)-0.5, minor=True)
    ax.set_yticks(np.arange(num_clients+1)-0.5, minor=True)
    ax.grid(which='minor', zorder=0, alpha=0.5, color='white')
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_facecolor("#E8EAED")
    if save_fig is not None:
        plt.savefig(save_fig+'/samples_per_class_per_client.png', dpi=300)
    myshow()

    fig, ax = plt.subplots(figsize=(0.4*num_clients, 0.4*num_classes))
    circles = [plt.Circle((i,j), radius=r) for r, j, i in zip(R.flat, x_mesh.flat, y_mesh.flat)]
    col = PatchCollection(circles, array=c.flatten(), cmap=cm_color, zorder=10)
    ax.add_collection(col)

    ax.set(xlabel='Client ID', ylabel='Classes')
    ax.set(xticks=np.arange(num_clients), yticks=np.arange(num_classes),
        xticklabels=ylabels, yticklabels=xlabels)
    ax.set_xticks(np.arange(num_clients+1)-0.5, minor=True)
    ax.set_yticks(np.arange(num_classes+1)-0.5, minor=True)
    ax.grid(which='minor', zorder=0, alpha=0.5, color='white')
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_facecolor("#E8EAED")
    if save_fig is not None:
        plt.savefig(save_fig+'samples_per_class_per_client_inverted.png', dpi=300)
    myshow()

    return X, y, statistic

# %%


def plot_loss_accuracy_server(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    round, measures = zip(*history.losses_distributed)
    plt.plot(round, measures, label="loss distributed")
    plt.xlabel("round")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    round, measures = zip(*history.metrics_distributed['accuracy'])
    plt.plot(round, measures, label="accuracy distributed", color='orange')
    plt.xlabel("round")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
# %%
