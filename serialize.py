 
import torch
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification
def get_model(model):
  """Loads model from Hugginface model hub"""
  try:
    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)
    return model
  except Exception as e:
    raise(e)

def get_tokenizer(tokenizer):
  """Loads tokenizer from Hugginface model hub"""
  try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer,padding_side='left')
    return tokenizer
  except Exception as e:
    raise(e)

model=get_model('bert_model_reddit_tsla_tracked_actions')
tokenizer=get_tokenizer('bert_model_reddit_tsla_tracked_actions')



inputs=tokenizer.encode('This is the best',return_tensors='pt',max_length=256,truncation=True,padding='max_length')
class PyToScript(torch.nn.
                 Module):
    def __init__(self,model):
        super(PyToScript,self).__init__()
        self.model=model
    def forward(self,data):
        return self.model(data).logits
     
pt_model=PyToScript(model).eval()
traced=torch.jit.trace(pt_model,inputs)
traced.save('model.pt')

import boto3
s3 = boto3.resource('s3')
s3.meta.client.upload_file('model.pt', 'triton-repository/models', 'model.pt')