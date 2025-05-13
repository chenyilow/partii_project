from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import pandas as pd
import string

def corpus(fileName, colList):
    return pd.read_csv(fileName)[colList]

def remove_punctuation(someString):
    output = ""
    for char in someString:
        if char.lower() in string.ascii_lowercase:
            output += char
    return output

# Generate pandas dataframe for the sentences
entireCorpus = corpus("annotated_data.csv", ["no_writing", "Text1", "HeadN", "def"])

noWritingList = entireCorpus[["no_writing"]]
uniqueNoWritingList = list(noWritingList.drop_duplicates()["no_writing"])

sentence_list = []
headNList = []
defList = []
for noWriting in uniqueNoWritingList:
    headN = []
    definiteness = []
    subCorpus = entireCorpus[entireCorpus["no_writing"] == noWriting]
    for row in subCorpus[["HeadN", "def"]].values:
        headN.append(row[0])
        definiteness.append(row[1])
    sentence = subCorpus.reset_index().loc[0, "Text1"]
    sentence_list.append(sentence)
    headNList.append(headN)
    defList.append(definiteness)

def common_nouns(model_name, sentence_list):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    inputs = tokenizer(sentence_list, return_tensors="pt", padding=True, truncation=True)

    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

    tokens = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
    predicted_labels = predictions.numpy()
    
    headNModelList = []
    definitenessModelList = []

    for sentence, token_list, pred in zip(sentence_list, tokens, predicted_labels):
        headNModel = []
        definitenessModel = []
        pred = list(zip(token_list.split(), pred[1:-1]))
        for idx, (token, label) in enumerate(pred):
            token = remove_punctuation(token)
            if token != "" and label == 7: # ignores tokens that are cleansed to "" and finds common nouns (which have label 7)
                print(token)
                headNModel.append(token)
                if idx == 0: # checks if it is the first word of the sentence
                    definitenessModel.append("indef")
                elif pred[idx - 1][0] == "the":
                    definitenessModel.append("def")
                else:
                    definitenessModel.append("indef")
        headNModelList.append(headNModel)
        definitenessModelList.append(definitenessModel)

    return headNModelList, definitenessModelList
    # label_map = model.config.id2label
    # print(label_map)

wordScore = 0
defScore = 0


for i in range(0, len(sentence_list), 100):
    batch = sentence_list[i:i+100]
    predWords, predDef = common_nouns("vblagoje/bert-english-uncased-finetuned-pos", sentence_list[i:i+100])

    for sentenceNo in range(len(batch)):
        for actualPos in range(len(headNList[i + sentenceNo])):
            word = headNList[i + sentenceNo][actualPos]
            if word in predWords[sentenceNo]:
                wordScore += 1
                predPos = predWords[sentenceNo].index(word)
                if defList[i + sentenceNo][actualPos] == predDef[sentenceNo][predPos]:
                    defScore += 1
    print("Finished sentences", i+1, "to", i+100)

print("Number of HeadN detected:", wordScore)
print("Number of def/indef correctly predicted:", defScore)
print("Total Number of words:", len(entireCorpus))
print("Accuracy of HeadN detection:", wordScore / len(entireCorpus))
print("Accuracy of def/indef prediction:", defScore / wordScore)

# common_nouns("vblagoje/bert-english-uncased-finetuned-pos", 
#             ["This is just another question that determines the aptitude of your application in a subject."])

# https://github.com/richardpaulhudson/coreferee/blob/master/coreferee/lang/en/language_specific_rules.py