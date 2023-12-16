# %%
import os
import cifar
import torch
import pickle
from tqdm.auto import tqdm
from utils import separate_data


def create_data_client(num_clients, num_classes, least_samples=1024, niid=False, partition='pat',
                       alpha=0.1, balance=True, class_per_client=2, save_fig=None):
    trainloader, testloader, num_examples = cifar.load_data()
    x_train_list = []
    y_train_list = []
    for train in tqdm(trainloader): 
        x_train_list.append(train[0])
        y_train_list.append(train[1])

    x_train = torch.cat(x_train_list, dim=0).numpy()
    y_train = torch.cat(y_train_list, dim=0).numpy()

    X, y, statistics = separate_data((x_train, y_train), num_clients, num_classes, niid=niid,
                                     least_samples=least_samples, partition=partition, alpha=alpha, balance=balance,
                                     class_per_client=class_per_client, save_fig=save_fig)

    return X, y, statistics
# %%


if __name__ == '__main__':
    num_clients = 5
    num_classes = 10
    least_samples = 1024
    partition = 'dir'
    alpha = .1
    niid = False
    if niid:
        balance = False
        fig_path = f'./cifar_client/alpha={alpha}/'
    else:
        balance = True
        fig_path = f'./cifar_client/iid/num_client={num_clients}/'
    os.makedirs(fig_path, exist_ok=True)

    X, y, statistics = create_data_client(num_clients, num_classes, least_samples=least_samples,
                                         niid=niid, partition=partition, alpha=alpha, balance=balance,
                                         class_per_client=2, save_fig=fig_path)

    with open(f'{fig_path}/X_train.pkl', 'wb') as f:
        pickle.dump(X, f)
    with open(f'{fig_path}/y_train.pkl', 'wb') as f:
        pickle.dump(y, f)

    print("Done!")
