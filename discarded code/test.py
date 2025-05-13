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