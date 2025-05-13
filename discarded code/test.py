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

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import pandas as pd
import string

def freqList(l):
    output = {}
    for item in l:
        if item not in output:
            output[item] = 1
        else:
            output[item] += 1
    return output

count = 0

entireCorpus = pd.read_csv("annotated_data.csv")
print(entireCorpus.columns)
entireCorpus = entireCorpus.sort_values(by="item_id")
noWritings = entireCorpus.drop_duplicates("no_writing")["no_writing"]
for noWriting in noWritings:
    subCorpus = entireCorpus[entireCorpus["no_writing"] == noWriting]
    sentence = subCorpus.reset_index().loc[0, "Text1"]
    HeadN = subCorpus["HeadN"]
    problematicSentence = False
    for word in HeadN:
        if sentence.count(word) != freqList(list(HeadN))[word]:
            print(word, noWriting, sentence.count(word), freqList(list(HeadN))[word])
            problematicSentence = True
    if problematicSentence:
        count += 1

print(count)