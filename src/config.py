
MAX_NUM_WORDS = 15000
EMBEDDING_DIM = 300
MAX_SEQ_LENGTH = 140
USE_GLOVE = True

# MODEL
FILTER_SIZES = [3, 4, 5]
FEATURE_MAPS = [10, 10, 10]
DROPOUT_RATE = 0.5

# LEARNING
BATCH_SIZE = 200
NB_EPOCHS = 10
RUNS = 1


text_data = "/content/drive/MyDrive/medical_multimodal_with_transfer_learning/text_processed.pkl"
img_data = "/content/drive/MyDrive/medical_multimodal_with_transfer_learning/x_ray_processed.pkl"
vocab = "/content/drive/MyDrive/medical_multimodal_with_transfer_learning/vocab.json"
raw_text_labels = "/content/drive/MyDrive/medical_multimodal_with_transfer_learning/ids_raw_texts_labels.csv"

# EMBEDDING
MAX_NUM_WORDS  = 15000
EMBEDDING_DIM  = 300
MAX_SEQ_LENGTH = 140
USE_GLOVE      = True

# MODEL
FILTER_SIZES   = [3,4,5]
FEATURE_MAPS   = [10,10,10]
DROPOUT_RATE   = 0.5

# LEARNING
BATCH_SIZE     = 200
NB_EPOCHS      = 10
RUNS           = 1
