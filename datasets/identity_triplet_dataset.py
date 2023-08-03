from torch.utils.data import Dataset, DataLoader
import random
import numpy as np


class IdentityTripletDataset(Dataset):
    def __init__(self, persons_list, triplets_n):
        self._persons_list = persons_list
        self._triplets_n = triplets_n
        self._triplets = self._generate_triplets()

    def _generate_triplets(self):
        triplets = []
        for _ in range(self._triplets_n):
            pos_person_idx, neg_person_idx = random.sample(range(len(self._persons_list)), 2)
            pos_person = self._persons_list[pos_person_idx]
            pos_audio_idx = random.choice(range(pos_person._audio_embeddings_n))
            pos_image_idx = random.choice(range(pos_person._image_embeddings_n))
            neg_person = self._persons_list[neg_person_idx]
            neg_image_idx = random.choice(range(neg_person._image_embeddings_n))
            triplets.append([pos_person_idx, pos_audio_idx, pos_image_idx, neg_person_idx, neg_image_idx])
        return np.array(triplets)

    def __len__(self):
        return self._triplets_n

    def __getitem__(self, idx):
        pos_person_idx, pos_audio_idx, pos_image_idx, neg_person_idx, neg_image_idx = self._triplets[idx]
        pos_audio = self._persons_list[pos_person_idx]._audio_embeddings[pos_audio_idx]
        pos_image = self._persons_list[pos_person_idx]._image_embeddings[pos_image_idx]
        neg_image = self._persons_list[neg_person_idx]._image_embeddings[neg_image_idx]
        return pos_audio, pos_image, neg_image  # anchor, positive, negative
