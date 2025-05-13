import stanza
import logging
import pandas as pd
from sklearn.metrics import classification_report

# Suppress stanza logs
logging.getLogger('stanza').setLevel(logging.WARNING)

# Initialize Stanza NLP pipeline
nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'})

# Occurrences of HeadN in NP: {0: 521, 1: 5099, 2: 226, 3: 16, 4: 1, 5: 0}
# Most HeadN are recorded twice, negligible thrice and above
df = pd.read_csv("annotated_data.csv")
df.sort_values(by='item_id', ascending=True, inplace=True)
df["Pred"] = "zero"
df["Occurrence"] = df.groupby(["NP", "HeadN"]).cumcount()

def get_word(sentence, target_word, occurrence):
    # Takes a sentence object, finds the dictionary representing the occurrence of that word
    for word in sentence.words:
        if word.text == target_word:
            if occurrence == 0:
                return word
            occurrence -= 1
    return None
count = 0
for paragraph in df["NP"].drop_duplicates():
    print(count)
    count += 1
    sentence = nlp(paragraph)
    for tokenised_sentence in sentence.sentences:
        rows = df[df["NP"] == paragraph][["HeadN", "Occurrence"]]
        for i, row in rows.iterrows():
            word = get_word(tokenised_sentence, row["HeadN"], row["Occurrence"])
            if word != None:
                for w in tokenised_sentence.words:
                    if w.deprel == "det" and w.head == word.id:
                        if w.text.lower() in {"a", "the"}:
                            df.loc[i, "Pred"] = w.text.lower()

print(classification_report(df['Target'], df['Pred'], target_names=None))

df.to_csv("dependency_pred.csv")

#               precision    recall  f1-score   support

#            a       0.91      0.63      0.74      1695
#          the       0.87      0.76      0.81      2108
#         zero       0.64      0.90      0.75      2060

#     accuracy                           0.77      5863
#    macro avg       0.81      0.76      0.77      5863
# weighted avg       0.80      0.77      0.77      5863