import random
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset

from datasets.person import Person


class IdentityTripletDataset(Dataset):
    def __init__(self, persons_list: List[Person], triplets_n: int):
        self._persons_list = persons_list
        self._triplets_n = triplets_n
        self._triplets = self._generate_triplets()

    def _generate_triplets(self) -> np.ndarray:
        """
        Generates the triplets - three arrays: (1) audio anchors (2) image positives (3) image negatives.
        Due to memory consumption the triplets are saved as indices rather than full vectors.
        :return: In our case, the first column is positive person id, second column is its anchor audio index, third column
        is its positive image index, fourth column is negative person id, fifth column is negative image index.
        """
        triplets = []
        # generate the number of require triplets
        for _ in range(self._triplets_n):
            # sample two persons - first one for the anchor and positive samples, second for the negative sample
            pos_person_idx, neg_person_idx = random.sample(range(len(self._persons_list)), 2)
            pos_person = self._persons_list[pos_person_idx]
            # save anchor index
            pos_audio_idx = random.choice(range(pos_person._audio_embeddings_n))
            # save positive index
            pos_image_idx = random.choice(range(pos_person._image_embeddings_n))
            neg_person = self._persons_list[neg_person_idx]
            # save negative index
            neg_image_idx = random.choice(range(neg_person._image_embeddings_n))
            triplets.append([pos_person_idx, pos_audio_idx, pos_image_idx, neg_person_idx, neg_image_idx])
        return np.array(triplets)

    def __len__(self):
        return self._triplets_n

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_person_idx, pos_audio_idx, pos_image_idx, neg_person_idx, neg_image_idx = self._triplets[idx]
        # retrieves the vectors from the indices
        pos_audio = self._persons_list[pos_person_idx]._audio_embeddings[pos_audio_idx]
        pos_image = self._persons_list[pos_person_idx]._image_embeddings[pos_image_idx]
        neg_image = self._persons_list[neg_person_idx]._image_embeddings[neg_image_idx]
        return pos_audio, pos_image, neg_image  # anchor, positive, negative
