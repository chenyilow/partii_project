# Copyright 2025 [Your Name]
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
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# Requires download
# python3 -m spacy download en_core_web_md
import spacy
from sklearn.linear_model import LogisticRegression

def classifier(path1, path2, l1=False, l2=False, mixed=False, C=False):

    with open(path1, 'r') as file:
        abstr_list = [line.strip() for line in file.readlines()]

    with open(path2, 'r') as file:
        concr_list = [line.strip() for line in file.readlines()]

    train_set = [concr_list, abstr_list]

    nlp = spacy.load("en_core_web_md")

    X = np.stack([nlp(w)[0].vector for part in train_set for w in part])
    y = [label for label, part in enumerate(train_set) for _ in part]

    if l1:
        clf = LogisticRegression(class_weight='balanced', penalty='l1', solver = 'saga', max_iter=1000).fit(X, y)
    elif l2:
        clf = LogisticRegression(class_weight='balanced', penalty='l2', solver = 'saga', max_iter=1000).fit(X, y)
    elif isinstance(mixed, (int, float)) and 0 <= mixed <= 1:
        clf = LogisticRegression(class_weight='balanced', penalty='elasticnet', solver = 'saga', max_iter=1000, l1_ratio=mixed).fit(X, y)
    elif C:
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, C=C).fit(X, y)
    else:
        clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X, y)

    return clf, nlp

def classifier_accuracy(clf, nlp, classes, target_column, annotated_data_path="annotated_data.csv", count=False, only_accuracy=False):
    df = pd.read_csv(annotated_data_path)
    if count:
        df['Ntype'] = df['Ntype'].replace(['sing', 'pl'], 'countable')
    pred = []

    for word in df["HeadN"]:
        token = nlp(word)
        pred.append(classes[clf.predict([token.vector])[0]])

    if only_accuracy:
        return accuracy_score(df[target_column], pred)
    return classification_report(df[target_column], pred, target_names=classes)

def predict_words(clf, nlp, classes, words):
    predictions = []
    for word in words:
        token = nlp(word)
        pred_label = classes[clf.predict([token.vector])[0]]
        predictions.append((word, pred_label))
    return predictions

# clf, nlp = classifier("abstract.txt", "concrete.txt")
# print(classifier_accuracy(clf, nlp, ["concr", "abstr"], "Abstract"))

# clf, nlp = classifier("countable.txt", "uncountable.txt")
# print(classifier_accuracy(clf, nlp, ["mass", "countable"], "Ntype", count=True))

def plot_graph(x_vals, accuracies, filename, axis, log=False):
    plt.figure(figsize=(8, 3))
    plt.plot(x_vals, accuracies, marker='o', linestyle='-', color='blue')
    if log:
        plt.xscale("log")
    plt.xlabel(axis)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("visualisations/" + filename + ".png")
    plt.show()

c_values = [0.001, 0.01, 0.1, 1, 10, 100]

# accuracies = []
# for C in c_values:
#     clf, nlp = classifier("abstract.txt", "concrete.txt", C=C)
#     accuracies.append(classifier_accuracy(clf, nlp, ["concr", "abstr"], "Abstract", only_accuracy=True))
# plot_graph(c_values, accuracies, "abstr_C", "C", log=True)

accuracies = []
for C in c_values:
    clf, nlp = classifier("abstract.txt", "concrete.txt", C=C)
    accuracies.append(classifier_accuracy(clf, nlp, ["concr", "abstr"], "Abstract", only_accuracy=True))
plot_graph(c_values, accuracies, "count_C", "C", log=True)

accuracies = []
for i in range(11):
    clf, nlp = classifier("abstract.txt", "concrete.txt", mixed=i/10)
    accuracies.append(classifier_accuracy(clf, nlp, ["concr", "abstr"], "Abstract", only_accuracy=True))
x_vals = [i/10 for i in range(11)]
plot_graph(x_vals, accuracies, "abstr_l1", "l1_ratio")

accuracies = []
for i in range(11):
    clf, nlp = classifier("abstract.txt", "concrete.txt", mixed=i/10)
    accuracies.append(classifier_accuracy(clf, nlp, ["concr", "abstr"], "Abstract", only_accuracy=True, count=True))
x_vals = [i/10 for i in range(11)]
plot_graph(x_vals, accuracies, "count_l1", "l1_ratio")