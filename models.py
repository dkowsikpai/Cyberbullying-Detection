from torch import nn
from transformers import BertModel

class CyberbullyingClassifier(nn.Module):

  def __init__(self, PRE_TRAINED_MODEL_NAME, n_classes):
    super(CyberbullyingClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    bert_out = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    pooled_output = bert_out[1]
    # output = self.drop(pooled_output)
    return self.out(pooled_output)