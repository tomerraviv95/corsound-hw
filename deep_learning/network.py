from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    """
    The triplet network, having one pipeline for images and one pipeline for audios
    """

    def __init__(self):
        super(Network, self).__init__()
        self.audio_fc = nn.Sequential(nn.LayerNorm(192),
                                      nn.Linear(192, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 32))
        self.image_fc = nn.Sequential(nn.LayerNorm(512),
                                      nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 32))

    def forward(self, pos_audios: torch.Tensor, pos_images: torch.Tensor, neg_images: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The forward pass in the triplet network.
        :param pos_audios: the anchors, the batch of person1 audios in our case.
        :param pos_images: the positives, the batch of person1 images in our case.
        :param neg_images: the negatives, the batch of person2 images in our case.
        :return: tuple of outputs, running each input through the network in an iid manner of the rest
        """
        # normalize the inputs to the network before processing
        norm_pos_audios = F.normalize(pos_audios, dim=0, p=2)
        norm_pos_images = F.normalize(pos_images, dim=0, p=2)
        norm_neg_images = F.normalize(neg_images, dim=0, p=2)
        return self.audio_fc(norm_pos_audios), self.image_fc(norm_pos_images), self.image_fc(norm_neg_images)
