
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


from config import *

np.random.seed(RANDOM_SEED)


# Prepare Dataset
df = pd.read_csv(DATASET_NAME)
abusive = df[df.label == "abusive"]
normal = df[df.label == "normal"].sample(abusive.shape[0])
print("Abusive vs. Normal Shape:", (abusive.shape, normal.shape))

X = pd.concat([abusive.text, normal.text], axis=0)
y = pd.concat([abusive.label, normal.label], axis=0)
y = y.apply(lambda x: 1 if x == "abusive"  else 0)
print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.10, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.20, random_state=RANDOM_STATE)

print('Train split:', (X_train.shape, y_train.shape))
print('Validation split:', (X_val.shape, y_val.shape))
print('Test split:', (X_test.shape, y_test.shape))

train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

class CyberbullyingDataset(Dataset):

  def __init__(self, text, targets, tokenizer, max_len):
    self.text = text
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.text)
  
  def __getitem__(self, item):
    text = str(self.text[item])
    target = self.targets[item]

    encoding = tokenizer(text, padding=self.max_len, truncation=True, return_tensors="pt")

    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = CyberbullyingDataset(
    text=df.text.to_numpy(),
    targets=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=2 # Max allowed by Colab
  )

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

## Just testing
data = next(iter(train_data_loader))
print(data.keys())
print(data["input_ids"].shape)


