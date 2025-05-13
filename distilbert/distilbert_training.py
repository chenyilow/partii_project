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

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold

col = "def" # Other uses: "ref", "def", "Hawkins"

if col == "ref":
    label_map = {"CLS": 0, "SEP": 0, "PAD": 0, '-': 1, 'pred/prop': 2, 'ref': 3, 'nonref': 4, 'id_pred': 5}
elif col == "def":
    label_map = {"CLS": 0, "SEP": 0, "PAD": 0, '-': 1, 'def': 2, 'indef': 3}
elif col == "Hawkins":
    label_map = {"CLS": 0, "SEP": 0, "PAD": 0, '-': 1, 'indef': 2, 'situational': 3, 'explanatory': 4, 'anaphoric': 5, 'kind': 6}

df = pd.read_csv("conll.csv")
#df = df.iloc[:1000, :]

def vectorise(column):
    sequences = []
    current_sequence = []
    sentence = []
    for _, row in df.iterrows():
        if pd.notna(row["POS Tag"]):
            if row["POS Tag"] != "-":  # Continue collecting words
                sentence.append(row[column])
            else: # End of a sentence in a paragraph
                current_sequence.append(sentence)
                sentence = []
        else:  # End of a paragraph
            sequences.append(current_sequence)
            current_sequence = []
    return sequences

def flatten(xss):
    return [x for xs in xss for x in xs]

def preprocess_data(sentences, labels, tokenizer):
    input_ids, attention_masks, label_sequences = [], [], []
    for sentence, label in zip(sentences, labels):
        encoding = tokenizer(sentence, padding="max_length", truncation=True, max_length=256, return_tensors="pt", is_split_into_words=True)
        new_labels = [0] # initialise -1 for [CLS]

        for word, pos in zip(sentence, label):
            word_tokens = tokenizer.tokenize(word)
            for _ in word_tokens:
                new_labels.append(pos)

        new_labels.append(0) # add -2 for [SEP]
        new_labels += [0] * (256 - len(new_labels)) # for padding

        input_ids.append(encoding["input_ids"].squeeze(0).tolist())
        attention_masks.append(encoding["attention_mask"].squeeze(0).tolist())
        label_sequences.append(new_labels)

    return input_ids, attention_masks, label_sequences

def generate_dataset(col):
    word_vector = flatten(vectorise("Word"))
    label_vector = flatten(vectorise(col))
    converted_label_vector = [[label_map[label] for label in sentence] for sentence in label_vector]
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    input_ids, attention_masks, label_sequences = preprocess_data(word_vector, converted_label_vector, tokenizer)

    dataset = Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': label_sequences
    })
    return dataset

def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    labels = p.label_ids

    # Flatten predictions & labels
    preds = preds.flatten()
    labels = labels.flatten()

    # Mask: Ignore PAD (0) and label '-' (1), keep only labels >1
    mask = labels > 1
    filtered_preds = preds[mask]
    filtered_labels = labels[mask]

    if len(filtered_labels) == 0:  # Avoid division by zero
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Compute precision, recall, and F1-score (excluding ignored labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_labels, filtered_preds, labels=[i for i in range(len(label_map)-2)], average=None, zero_division=0
    )

    return {
        "accuracy": accuracy_score(filtered_labels, filtered_preds),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist()
    }

def generate_results(col, folds=10, random_state=42):
    # Placeholder for storing results
    all_logs = []
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    dataset = generate_dataset(col)
    for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}:")
        
        train_dataset = dataset.select(train_index)
        test_dataset = dataset.select(test_index)

        model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_map) - 2) 

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f'./results/{col}/fold_{fold+1}',
            eval_strategy="epoch",  
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        for log in trainer.state.log_history:
            if "eval_loss" in log:
                all_logs.append({
                    "Fold": fold + 1,
                    "Epoch": log.get("epoch"),
                    "Accuracy": log.get("eval_accuracy"),
                    "Precision": log.get("eval_precision"),
                    "Recall": log.get("eval_recall"),
                    "F1 Score": log.get("eval_f1")
                })
    results_df = pd.DataFrame(all_logs)

    # Save to CSV
    results_df.to_csv(f"cross_validation_results_{col.lower()}.csv", index=False)

    print(f"Results saved to cross_validation_results_{col.lower()}.csv")

generate_results(col)