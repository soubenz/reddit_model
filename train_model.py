from datasets import load_dataset
from transformers import DataCollatorWithPadding
import numpy as np
from datasets import load_metric
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers.integrations import MLflowCallback
import os
from getpass import getpass
import mlflow
import numpy as np
from datasets import load_metric
 

train_dataset = load_dataset('fourthbrain-demo/reddit-comments-demo', split='train[:1%]')
test_dataset = load_dataset('fourthbrain-demo/reddit-comments-demo')["test"]
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(data):
   return tokenizer(data["comment"], truncation=True, padding=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)



# os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ['DAGSHUB_USERNAME']
# os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ['DAGSHUB_TOKEN']
# os.environ['MLFLOW_TRACKING_PROJECTNAME'] = os.environ['DAGSHUB_PROJECT']
os.environ['MLFLOW_TRACKING_URI'] = f'https://dagshub.com/' + os.environ['MLFLOW_TRACKING_USERNAME'] + '/' + os.environ['MLFLOW_TRACKING_PROJECTNAME'] + '.mlflow'


def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)
repo_name = "bert_model_reddit_tsla_tracked"
 
training_args = TrainingArguments(
   output_dir=repo_name,
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=False,
)
 
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
   callbacks=[MLflowCallback()]
)


trainer.train()
