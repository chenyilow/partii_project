import stanza
import logging
import pandas as pd
import string
import json

logging.getLogger('stanza').setLevel(logging.WARNING)
nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'})

def token(paragraph, sentence=False, pos=False):
    doc = nlp(paragraph)
    sentence_list = []
    if sentence:
      for sentence in doc.sentences:
        sentence_list.append(sentence.text)
    elif pos:
      for sentence in doc.sentences:
        sentence_list.append([word.upos for word in sentence.words])
    else:
      for sentence in doc.sentences:
          sentence_list.append([word.text for word in sentence.words])
    return sentence_list
corpus = pd.read_csv("annotated_data.csv")
uniqueTexts = corpus.drop_duplicates(["Text1"])["Text1"]
sum = 0
text = 0
for paragraph in uniqueTexts:
      text += 1
      sentences = token(paragraph, sentence=True)
      tokenised_sentence = token(paragraph, pos=True)
      for i in tokenised_sentence:
         for j in i:
            if j == "NOUN":
               sum += 1
      print(tokenised_sentence, sum, text)
print(sum)

# def flatten(l):
#    if l == []:
#       return []
#    else:
#       return l[0] + flatten(l[1:])
   
# def cleanstring(text: str) -> str:
#     return ''.join(char for char in text if char not in string.whitespace + string.punctuation)

# def vectorise(fileName):

#   corpus = pd.read_csv(fileName)
#   uniqueTexts = corpus.drop_duplicates(["no_writing"])[["no_writing"]]
#   vectorList = []
#   counter = 0

#   for noWriting in uniqueTexts["no_writing"]:
#       trueset = corpus[corpus["no_writing"] == noWriting][["Text1", "NP", "HeadN", "def"]]
#       trueset = trueset.reset_index(drop=True)
#       paragraph = trueset.loc[0, "Text1"].strip()
#       sentences = token(paragraph, sentence=True)
#       tokenised_sentence = token(paragraph)
#       pos_sentence = token(paragraph, pos=True)
#       #print("sentences:", sentences, "\n", "tokenised_sentence", tokenised_sentence, "\n", "pos_sentence", pos_sentence)
#       vector = [[-1 if word == "NOUN" else 0 for word in sentence] for sentence in pos_sentence]
#       #print("vector:", vector)
#       headN = list(trueset["HeadN"])
#       defset = list(trueset["def"])
#       #print("headN:", headN)
#       #print()
#       all_tokens = flatten(tokenised_sentence)
#       for i, sentence in enumerate(tokenised_sentence):
#         for j, word in enumerate(sentence):
#             if vector[i][j] == -1:
#               if word not in headN: # word is not annotated, label ambiguous
#                   vector[i][j] = 1
#               elif headN.count(word) == 1:
#                   if all_tokens.count(word) == 1: # 1 in true and 1 in pred, must match
#                     vector[i][j] = 2 if defset[headN.index(word)] == "def" else 3
#                   elif sentence.count(word) == 1: # 1 in true but multiple of the same word in pred, disambiguate by NP
#                     pred_sentence = cleanstring(sentences[i])
#                     count = 0
#                     for _, row in trueset[["NP", "HeadN"]].iterrows():
#                       if pred_sentence == cleanstring(row["NP"]) and word == row["HeadN"]:
#                         count += 1
#                     if count == 1: # only match if the given word belongs in an NP that matches a single row's NP in annotated data
#                       vector[i][j] = 4 if defset[headN.index(word)] == "def" else 5
#                     else:
#                       vector[i][j] = 100
#                   else:
#                      vector[i][j] = 101
#               else: # multiple of the same word in true annotated dataset
#                   if all_tokens.count(word) != 1: # immediately assume multiple words pred, otherwise pred is ambiguous
#                     if sentence.count(word) == 1: # 1 in that NP of the tokenised sentence
#                       pred_sentence = cleanstring(sentences[i])
#                       count = 0
#                       for _, row in trueset[["NP", "HeadN"]].iterrows():
#                         if pred_sentence == cleanstring(row["NP"]) and word == row["HeadN"]:
#                           count += 1
#                       if count == 1: # only match if the given word belongs in an NP that matches a single row's NP in annotated data
#                         vector[i][j] = 6 if defset[headN.index(word)] == "def" else 7
#                       else:
#                         vector[i][j] = 102
#                     else:
#                        vector[i][j] = 103
#                   else:
#                      vector[i][j] = 104
#                         # There is no case for sentence.count(word) != 1 since it means the same NP has multiple words.
#                         # This cannot be disambiguated by the annotated dataset since it separates at the level of NP. 
#       counter += 1
#       print(counter, len(uniqueTexts))
#       vectorList.append([tokenised_sentence, vector])
#   return vectorList

# vectors = vectorise("annotated_data.csv")
# flat_list = []
# for item in vectors:
#    flat_list += flatten(item[1])

# # Count occurrences
# count_dict = {}
# for num in flat_list:
#     if num in count_dict:
#         count_dict[num] += 1
#     else:
#         count_dict[num] = 1

# print(count_dict)

# with open("vectorised.json", "w") as f:
#     json.dump(vectors, f, indent=4)

# print("Data saved to vectorised.json")