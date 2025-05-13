# Copyright 2025 Chen Yi Low
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# d = {0: 62103, 1: 7673, 3: 2274, 101: 169, 5: 219, 2: 1117, 100: 530, 7: 627, 6: 346, -1: 282, 102: 329, 104: 38, 4: 114}
# print(d[0], d[1] + d[-1] + d[100] + d[101] + d[102] + d[104], d[2] + d[4] + d[6], d[3] + d[5] + d[7] )
# 62103 9021 1577 3120

import json

with open("vectorised.json", "r") as file:
  data = json.load(file)

converted_data = []
for paragraph in data:
  vectors = []
  for i in paragraph[1]:
    vector = []
    for j in i:
      if j in [-1, 1, 100, 101, 102, 103, 104]:
        vector.append(1)
      elif j == 0:
        vector.append(0)
      elif j == 2 or j == 4 or j == 6:
        vector.append(2)
      elif j == 3 or j == 5 or j == 7:
        vector.append(3)
    vectors.append(vector)
  converted_data.append([paragraph[0], vectors])
print(converted_data)
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import numpy as np

tokenised_sentences = []
labels = []
for i in converted_data:
  tokenised_sentences += i[0]
  labels += i[1]
print(tokenised_sentences)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=5) 

def preprocess_data(sentences, labels):
    input_ids, attention_masks, label_sequences = [], [], []

    for sentence, label in zip(sentences, labels):

        sentence_str = " ".join(sentence)

        encoding = tokenizer(sentence_str, padding="max_length", truncation=True, max_length=64, return_tensors="pt")

        label = label[:64]
        label += [0] * (64 - len(label))

        input_ids.append(encoding["input_ids"].squeeze(0).tolist())
        attention_masks.append(encoding["attention_mask"].squeeze(0).tolist())
        label_sequences.append(label)

    return input_ids, attention_masks, label_sequences

input_ids, attention_masks, label_sequences = preprocess_data(tokenised_sentences, labels)

dataset = Dataset.from_dict({
    'input_ids': input_ids,
    'attention_mask': attention_masks,
    'labels': label_sequences
})

train_dataset = dataset.train_test_split(test_size=0.2)['train']
eval_dataset = dataset.train_test_split(test_size=0.2)['test']

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(p):
    predictions, labels = p

    predictions = predictions.argmax(axis=-1).flatten()
    labels = labels.flatten()

    mask = labels != 0
    predictions = predictions[mask]
    labels = labels[mask]

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results", 
    eval_strategy="epoch",  
    per_device_eval_batch_size=8, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)


eval_results = trainer.evaluate()

print(eval_results)
