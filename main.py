import random

import numpy as np
import torch
import torch.nn as nn
from pickle5 import pickle
from torch.utils.data import DataLoader

from datasets.identity_triplet_dataset import IdentityTripletDataset
from datasets.voice_faces_dataset import VoiceFacesDataset
from deep_learning.evaluation import evaluate
from deep_learning.network import Network
from deep_learning.training import train

SEED = 100
BATCH_SIZE = 64
TRAIN_TRIPLETS = 10000
VAL_TRIPLETS = 5000
LEARNING_RATE = 1e-3
MAX_EPOCHS = 10

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

if __name__ == "__main__":
    # Load embeddings
    audio_embeddings = pickle.load(open('data/audio_embeddings.pickle', 'rb'))
    image_embeddings = pickle.load(open('data/image_embeddings.pickle', 'rb'))

    # Create the train and validation datasets of triplets
    voice_faces_dataset = VoiceFacesDataset(audio_embeddings, image_embeddings)
    train_identity_triplet_dataset = IdentityTripletDataset(persons_list=voice_faces_dataset.train_persons_list,
                                                            triplets_n=TRAIN_TRIPLETS)
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
    evaluate(val_dataloader, net)
