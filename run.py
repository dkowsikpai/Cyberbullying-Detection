import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer, BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from models import CyberbullyingClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

PRE_TRAINED_MODEL_NAME = 'nreimers/BERT-Tiny_L-2_H-128_A-2'
SAVED_MODEL_NAME = './Model/best_model_state.bin'
BATCH_SIZE = 32
MAX_LEN = "max_length"
class_names = ["Normal", "Abusive"]

# DATASET_NAME = "clean_hatespeech_text_label_vote.csv"
# RANDOM_STATE = 42

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)



def get_predictions(model, data_loader):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values


if __name__ == "__main__":

    best_model = CyberbullyingClassifier(PRE_TRAINED_MODEL_NAME, n_classes=len(class_names))
    best_model.load_state_dict(torch.load(SAVED_MODEL_NAME, map_location=torch.device(device)))
    best_model = best_model.to(device)
    
    print("\n==========================================================\n")

    
    # """userid Why are there a lot 3rd gen kpop fans so.. fucking stupid? Saying tvxq, bigbang, t-ara, wonder girls"""
    review_text = input("Enter a text: ")

    encoded_review = tokenizer(review_text, padding=MAX_LEN, truncation=True, return_tensors="pt")
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    output = best_model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)


    print(f'Review text: {review_text}')
    print(f'Sentiment  : {class_names[prediction]}')

