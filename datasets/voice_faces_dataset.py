from typing import Dict, List

import numpy as np

from datasets.person import Person
from utils import get_person_name_from_key, list_splitter


class VoiceFacesDataset:
    def __init__(self, audio_embeddings: Dict[str, np.ndarray], image_embeddings: Dict[str, np.ndarray],
                 train_ratio: float = 0.8):
        # get only the persons having at least one image and one audio
        names_from_image_embeddings = list(set(map(get_person_name_from_key, list(image_embeddings.keys()))))
        names_from_audio_embeddings = list(set(map(get_person_name_from_key, list(audio_embeddings.keys()))))
        unique_names = [name for name in names_from_image_embeddings if name in names_from_audio_embeddings]

        # create empty persons dict
        persons_dict = {name: Person(name) for name in unique_names}

        # populate the persons for the images
        self.populate_image_embeddings(image_embeddings, persons_dict, unique_names)

        # populate the persons for the audios
        self.populate_audio_embeddings(audio_embeddings, persons_dict, unique_names)

        # create full, train, and validation persons list
        self.persons_list = list(persons_dict.values())
        self.train_persons_list, self.val_persons_list = list_splitter(self.persons_list, train_ratio)

    def populate_audio_embeddings(self, audio_embeddings: Dict[str, np.ndarray], persons_dict: Dict[str, Person],
                                  unique_names: List[str]):
        for full_key, audio_embedding in audio_embeddings.items():
            name_from_embedding = get_person_name_from_key(full_key)
            if name_from_embedding in unique_names:
                persons_dict[name_from_embedding].add_audio_embedding(audio_embedding)

    def populate_image_embeddings(self, image_embeddings: Dict[str, np.ndarray], persons_dict: Dict[str, Person],
                                  unique_names: List[str]):
        for full_key, image_embedding in image_embeddings.items():
            name_from_embedding = get_person_name_from_key(full_key)
            if name_from_embedding in unique_names:
                persons_dict[name_from_embedding].add_image_embedding(image_embedding)

    def __len__(self):
        return len(self.persons_list)
