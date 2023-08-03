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

random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)

if __name__ == "__main__":
    audio_embeddings = pickle.load(open('data/audio_embeddings.pickle', 'rb'))
    image_embeddings = pickle.load(open('data/image_embeddings.pickle', 'rb'))

    voice_faces_dataset = VoiceFacesDataset(audio_embeddings, image_embeddings)
    train_identity_triplet_dataset = IdentityTripletDataset(persons_list=voice_faces_dataset.train_persons_list,
                                                            triplets_n=10000)
    train_dataloader = DataLoader(train_identity_triplet_dataset, batch_size=64, shuffle=True)

    val_identity_triplet_dataset = IdentityTripletDataset(persons_list=voice_faces_dataset.val_persons_list,
                                                          triplets_n=5000)
    val_dataloader = DataLoader(val_identity_triplet_dataset, batch_size=64, shuffle=True)

    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    triplet_loss = nn.TripletMarginLoss()
    lr = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    max_epoch = 10
    train(train_dataloader, optimizer, triplet_loss, net, max_epoch)

    evaluate(val_dataloader, net)
