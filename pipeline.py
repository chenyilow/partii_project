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

import stanza
import pandas as pd
import json
from syntactic.syntactic import *
from distilbert.distilbert_prediction import *
from logistic_regression.nltk import *

# -------- Pipeline: Part 1 --------

df, test_sentences, positions = make_syntactic_information(10, correct="datasets/parallel_corpus/correct.txt",
                                orig="datasets/parallel_corpus/orig.txt",
                                mother_tongue="datasets/parallel_corpus/mother_tongue.txt")

# -------- Pipeline: Part 2 --------

tokenizer_def, model_def = load_model_and_tokenizer("def")
tokenizer_ref, model_ref = load_model_and_tokenizer("ref")
tokenizer_hawkins, model_hawkins = load_model_and_tokenizer("Hawkins")

predictions_def = predict_labels(test_sentences, tokenizer_def, model_def, "def")
predictions_hawkins = predict_labels(test_sentences, tokenizer_hawkins, model_hawkins, "Hawkins")
predictions_ref = predict_labels(test_sentences, tokenizer_ref, model_ref, "ref")

data_2 = []

for i, sentence in enumerate(test_sentences):
    pos = positions[i]
    data_2.append({
        "def": predictions_def[i][pos],
        "ref": predictions_ref[i][pos],
        "Hawkins": predictions_hawkins[i][pos]
    })

df_2 = pd.DataFrame(data_2)
df = pd.concat([df, df_2], axis=1)

# -------- Pipeline: Part 3 --------

nouns = df["HeadN"].to_list()

clf_abstr, nlp_abstr = classifier(
        path1="logistic_regression/abstract.txt",
        path2="logistic_regression/concrete.txt", 
    )

pred_abstr = predict_words(clf_abstr, nlp_abstr, ['concr', 'abstr'], nouns)
pred_abstr = [label for _, label in pred_abstr]

clf_count, nlp_count = classifier(
        path1="logistic_regression/countable.txt",
        path2="logistic_regression/uncountable.txt", 
    )

pred_count = predict_words(clf_count, nlp_count, ['mass', 'count'], nouns)
pred_count = [label for _, label in pred_count]

df["Abstract"] = pred_abstr
df["Rev_abstr"] = pred_abstr

for idx, value in enumerate(pred_count):
    if value == 'mass':
        df.at[idx, 'Ntype'] = 'mass'

df = df.rename(columns={"L2": "NP"})
df["item_id"] = range(1, 1 + len(df))
df = df[["item_id", "NL", "NP", "Oblig", "Error", "Target", "ErType", "ErType2", "def", "ref", "Hawkins", "HeadN", "Ntype", "Abstract", "Rev_abstr", "Synt", "modif"]]
#df.to_csv("results/synthetic_dataset.csv", index=False)
df.to_csv("test.csv", index=False)