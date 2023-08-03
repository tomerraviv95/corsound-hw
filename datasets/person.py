import numpy as np


class Person:
    """
    Class for holding the embedding of a certain individual
    """

    def __init__(self, name: str):
        self.name = name
        self._image_embeddings = []
        self._audio_embeddings = []
        self._image_embeddings_n = 0
        self._audio_embeddings_n = 0

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def add_image_embedding(self, image_embedding: np.ndarray):
        self._image_embeddings.append(image_embedding)
        self._image_embeddings_n += 1

    def add_audio_embedding(self, audio_embedding: np.ndarray):
        self._audio_embeddings.append(audio_embedding)
        self._audio_embeddings_n += 1
