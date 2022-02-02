import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

TRAIN = True

DATASET_NAME = "./Datasets/clean_hatespeech_text_label_vote.csv"
RANDOM_STATE = RANDOM_SEED = 42

PRE_TRAINED_MODEL_NAME = 'nreimers/BERT-Tiny_L-2_H-128_A-2'
SAVED_MODEL_NAME = './Model/best_model_state.bin'
MODEL_TRAINING_NAME = './Model/training_best_model_state.bin'

BATCH_SIZE = 32
MAX_LEN = "max_length"
CLASS_NAMES = ["Normal", "Abusive"]

EPOCHS = 10
LEARNING_RATE = 0.001