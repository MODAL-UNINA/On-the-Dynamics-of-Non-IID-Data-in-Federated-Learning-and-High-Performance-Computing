# %%
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_dir = ''

beta = 0.1
server_IP = 'localhost:8080'
num_clients = 6

betas = [0.1, 0.3, 0.5, 1., 'iid']

save_path = '...'
os.makedirs(save_path, exist_ok=True)
path = f'../{output_dir}/beta={beta}/Results_{server_IP}_{num_clients}'

list_file = os.listdir(f'{path}')
server_file = [file for file in list_file if file.startswith('Server')][0]
# %%


def plot_global_accuracy_betas(betas, path, save_path=None):
    accuracies_beta = []
    for beta in betas:
        list_file = os.listdir(f'{path}')
        server_file = [file for file in list_file if file.startswith('Server')][0]
        with open(f'{path}/{server_file}', 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            line_acc = [line for line in lines if line.startswith('Global model accuracy')]
            line_acc = [line.split(':')[-1] for line in line_acc]
            line_acc = line_acc[0].replace(" ", "").replace("\n", "")
            values_list = ast.literal_eval(line_acc)
        rounds = [round_value[0] for round_value in values_list]
        accuracies = [accuracy_value[1] for accuracy_value in values_list]
        accuracies_beta.append(accuracies)
    plt.figure(figsize=(10, 6))
    for i, beta in enumerate(betas):
        if beta == 'iid':
            plt.plot(rounds, accuracies_beta[i], '-', label='IID')
        else:
            plt.plot(rounds, accuracies_beta[i], '-', label=f'$\\beta$={beta}')
    
    plt.xlabel('Rounds', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.title('Global model accuracy', fontsize=18)
    plt.legend(fontsize=16)
    if save_path is not None:
        plt.savefig(save_path + f'/global_model_accuracy_betas_num_clients={num_clients}.png')


def plot_communication_efficiency_betas(betas, save_path=None):
    communication_efficiency_beta = []
    for beta in betas:
        path = f'{output_dir}/beta={beta}/Results_{server_IP}_{num_clients}'
        list_file = os.listdir(f'{path}')
        server_file = [file for file in list_file if file.startswith('Server')][0]
        with open(f'{path}/{server_file}', 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            line_c_e = [line for line in lines if line.startswith('Communication efficiency')]
            line_c_e = [line.split(':')[-1] for line in line_c_e]
            line_c_e = line_c_e[0].replace(" ", "").replace("\n", "")
            values_list = ast.literal_eval(line_c_e)
        rounds = [i for i in range(len(values_list))]
        communication_efficiency = [value for value in values_list]
        communication_efficiency_beta.append(communication_efficiency)
    bar_width = 0.2

    round_spacing = 0.5

    positions = np.arange(len(rounds)) * (len(betas) * bar_width + round_spacing)

    plt.figure(figsize=(12, 6))

    for i, beta in enumerate(betas):
        if beta == 'iid':
            plt.bar(positions + i * bar_width, communication_efficiency_beta[i], width=bar_width, label=f'IID')
        else:
            plt.bar(positions + i * bar_width, communication_efficiency_beta[i], width=bar_width, label=f'$\\beta$={beta}')

    plt.xlabel('Rounds', fontsize=18)
    plt.ylabel('Communication efficiency', fontsize=18)
    plt.yscale('log')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)

    plt.xticks(positions + (len(betas) * bar_width) / 2, [int(round+1) for round in rounds])

    if save_path is not None:
        plt.savefig(save_path + f'/communication_efficiency_betas_num_clients={num_clients}.png')

    plt.show()


def plot_efficiency_score(path, server_file, numclient, server_IP, save_path=None):

    path = f'{output_dir}/beta={beta}/Results_{server_IP}_{num_clients}'
    
    list_files = os.listdir(f'{path}')
    server_file = [file for file in list_files if file.startswith('Server')][0]
    with open(f'{path}/{server_file}', 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        line_acc = [line for line in lines if line.startswith('Global model accuracy')]
        line_acc = [line.split(':')[-1] for line in line_acc]
        line_acc = line_acc[0].replace(" ", "").replace("\n", "")
        values_list = ast.literal_eval(line_acc)
        rounds = [round_value[0] for round_value in values_list]
        accuracies = [accuracy_value[1] for accuracy_value in values_list]
        fle_cifar = [line for line in lines if line.startswith('Federated Learning Efficiency')]
        fle_cifar = [line.split(':')[-1] for line in fle_cifar]
        fle_cifar = ast.literal_eval(fle_cifar[0])
    
    path = f'{output_dir}/beta={beta}/Results_{server_IP}_{num_clients}'

    list_files = os.listdir(f'{path}')
    server_file = [file for file in list_files if file.startswith('Server')][0]
    with open(f'{path}/{server_file}', 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        line_acc = [line for line in lines if line.startswith('Global model accuracy')]
        line_acc = [line.split(':')[-1] for line in line_acc]
        line_acc = line_acc[0].replace(" ", "").replace("\n", "")
        values_list = ast.literal_eval(line_acc)
        rounds = [round_value[0] for round_value in values_list]
        accuracies = [accuracy_value[1] for accuracy_value in values_list]
        fle_fmnist = [line for line in lines if line.startswith('Federated Learning Efficiency')]
        fle_fmnist = [line.split(':')[-1] for line in fle_fmnist]
        fle_fmnist = ast.literal_eval(fle_fmnist[0])

    rounds = [i for i in range(len(values_list))]
    # efficiency_score = [round(accuracies[i] / np.mean(power_client), 2) for i in range(len(accuracies))]
    efficiency_score = [accuracies[i] / np.mean(power_client) for i in range(len(accuracies))]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, fle_cifar, '-o', label='FLE CIFAR-10', color='tab:red')
    plt.plot(rounds, fle_fmnist, '-o', label='FLE Fashion MNIST', color='tab:blue')
    plt.xlabel('Rounds', fontsize=18)
    plt.ylabel('Federated Learning Efficiency (FLE)', fontsize=18)
    plt.yscale('log')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)
    if save_path is not None:
        plt.savefig(save_path + f'/efficiency_score_num_clients={num_clients}.png')
    plt.show()




paths_cifar=['list of paths']
paths_fmnist=['list of paths']

def plot_execution_time(paths):
    execution_time = pd.DataFrame(columns=['path','execution_time'])
    for path in paths:
        list_files = os.listdir(f'{path}')
        list_files = [file for file in list_files if file.startswith('Server')]
        for f in list_files:
            with open(f'{path}/{f}', 'r') as file:
                lines = file.readlines()
                line_time = [line for line in lines if line.startswith('Total training time')]
                execution_time = pd.concat([execution_time, pd.DataFrame([[path, float(line_time[-1].split(' ')[-2])]], columns=['path','execution_time'])], ignore_index=True)
    
    speedup_tot = []
    for i in execution_time.index:
        num = execution_time.iloc[0,1]
        speedup = num/execution_time.iloc[i,1]
        speedup_tot.append(speedup)

    return speedup_tot

# %% -----------------------------------------------------------------------------------------------

plot_global_accuracy_betas(betas, num_clients, save_path=save_path)

plot_communication_efficiency_betas(betas, num_clients, save_path=save_path)

plot_efficiency_score(path, server_file, num_clients, server_IP, save_path=save_path)


speedup_cifar = plot_execution_time(paths_cifar)
speedup_fmnist = plot_execution_time(paths_fmnist)

num_clients = np.array(range(2, 7))
plt.figure(figsize=(10, 6))
xticks = [2, 3, 4, 5, 6]
plt.xticks(xticks, xticks, fontsize=18)
plt.plot(num_clients, speedup_cifar, marker='s', label='Speedup CIFAR-10')
plt.plot(num_clients, speedup_fmnist, marker='s', label='Speedup Fashion MNIST')
plt.xlabel('Number of clients', fontsize=18)
plt.ylabel('Speedup', fontsize=18)
plt.legend(fontsize=16)
plt.savefig(save_path + f'/speedup_num_clients={num_clients}.png')
plt.show()