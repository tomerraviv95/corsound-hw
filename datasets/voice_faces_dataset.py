from datasets.person import Person
from utils import get_person_name_from_key, list_splitter


class VoiceFacesDataset:
    def __init__(self, audio_embeddings, image_embeddings, train_ratio=0.8):
        names_from_image_embeddings = list(set(map(get_person_name_from_key, list(image_embeddings.keys()))))
        names_from_audio_embeddings = list(set(map(get_person_name_from_key, list(audio_embeddings.keys()))))
        self.names_from_image_embeddings = names_from_image_embeddings
        self.unique_names = [name for name in names_from_image_embeddings if name in names_from_audio_embeddings]
        self.persons_dict = {name: Person(name) for name in self.unique_names}

        ## populate the persons for the images
        for full_key, image_embedding in image_embeddings.items():
            name_from_embedding = get_person_name_from_key(full_key)
            if name_from_embedding in self.unique_names:
                self.persons_dict[name_from_embedding].add_image_embedding(image_embedding)

        ## populate the persons for the audios
        for full_key, audio_embedding in audio_embeddings.items():
            name_from_embedding = get_person_name_from_key(full_key)
            if name_from_embedding in self.unique_names:
                self.persons_dict[name_from_embedding].add_audio_embedding(audio_embedding)

        self.persons_list = list(self.persons_dict.values())
        self.train_persons_list, self.val_persons_list = list_splitter(self.persons_list, train_ratio)

    def __len__(self):
        return len(self.persons_list)
