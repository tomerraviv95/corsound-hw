import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from pickle5 import pickle
from torch import nn
from torch.utils.data import DataLoader

from datasets.identity_triplet_dataset import IdentityTripletDataset
from datasets.voice_faces_dataset import VoiceFacesDataset
from deep_learning.evaluation import evaluate
from deep_learning.network import Network
from deep_learning.training import train
from dir_paths import TRAIN_VAL_AUDIO_EMBEDDINGS, TRAIN_VAL_IMAGE_EMBEDDINGS
from main import BATCH_SIZE, VAL_TRIPLETS, LEARNING_RATE, MAX_EPOCHS

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


def plot_by_train_triplet(train_triplets: int, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # Load embeddings
    audio_embeddings = pickle.load(open(TRAIN_VAL_AUDIO_EMBEDDINGS, 'rb'))
    image_embeddings = pickle.load(open(TRAIN_VAL_IMAGE_EMBEDDINGS, 'rb'))

    # Create the train and validation datasets of triplets
    voice_faces_dataset = VoiceFacesDataset(audio_embeddings, image_embeddings)
    train_identity_triplet_dataset = IdentityTripletDataset(persons_list=voice_faces_dataset.train_persons_list,
                                                            triplets_n=train_triplets)
    train_dataloader = DataLoader(train_identity_triplet_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_identity_triplet_dataset = IdentityTripletDataset(persons_list=voice_faces_dataset.val_persons_list,
                                                          triplets_n=VAL_TRIPLETS)
    val_dataloader = DataLoader(val_identity_triplet_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create the network and hyperparameters
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()
    triplet_loss = nn.TripletMarginLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Train and validate the Network
    train(train_dataloader, optimizer, triplet_loss, net, MAX_EPOCHS)
    identification_acc = evaluate(val_dataloader, net)
    return identification_acc


if __name__ == "__main__":
    identification_accs = []
    train_triplets = [500, 1000, 2000, 3000, 5000, 7500, 10000, 12500, 15000]
    seeds = [100, 101, 102]
    for train_triplet in train_triplets:
        identification_acc = 0
        for seed in seeds:
            identification_acc += plot_by_train_triplet(train_triplet, seed)
        identification_acc /= len(seeds)
        identification_accs.append(identification_acc)
    plt.figure()
    plt.plot(train_triplets, identification_accs, markersize=11, linewidth=2.2, color='green', marker='o')
    xlabel = 'Train Triplets'
    ylabel = 'Identification Accuracy'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='both', ls='--')
    plt.savefig(f'{ylabel}_versus_{xlabel}.png', bbox_inches='tight')
