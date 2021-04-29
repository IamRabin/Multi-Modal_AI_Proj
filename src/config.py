
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


text_data = "/source/Data_preprocessed/text_processed.pkl"
img_data = "/source/Data_preprocessed/x_ray_processed.pkl"
vocab = "/source/Data_preprocessed/vocab.json"
raw_text_labels = "/source/Data_preprocessed/ids_raw_texts_labels.csv"

# EMBEDDING
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
