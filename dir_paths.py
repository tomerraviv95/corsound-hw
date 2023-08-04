import os

# main folders
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# given embeddings
TRAIN_VAL_AUDIO_EMBEDDINGS = os.path.join(DATA_DIR, 'audio_embeddings.pickle')
TRAIN_VAL_IMAGE_EMBEDDINGS = os.path.join(DATA_DIR, 'image_embeddings.pickle')

# hidden test embeddings
TEST_AUDIO_EMBEDDINGS = os.path.join(DATA_DIR, 'test_audio_embeddings.pickle')
TEST_IMAGE_EMBEDDINGS = os.path.join(DATA_DIR, 'test_image_embeddings.pickle')
