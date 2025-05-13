import stanza
import pandas as pd
import json
from distilbert.distilbert_prediction import *
from logistic_regression.nltk import *

l1 = []
l2 = []
mt = []

for line in open("datasets/parallel_corpus/correct.txt"):
    l1.append(line.strip())
for line in open("datasets/parallel_corpus/orig.txt"):
    l2.append(line.strip())
for line in open("datasets/parallel_corpus/mother_tongue.txt"):
    mt.append(line.strip())

with open("results/alignments.json", "r") as f:
    alignments = json.load(f)

corpus = pd.DataFrame([l1, l2, alignments, mt]).T
corpus.columns = ["L1", "L2", "Alignment", "NL"]

nlp = stanza.Pipeline(lang='en', processors={"tokenize": "spaCy"})

error = {'a': {'a': 'correct', 'the': 'error', 'zero': 'error'},
          'the': {'a': 'error', 'the': 'correct', 'zero': 'error'},
          'zero': {'a': 'error', 'the': 'error', 'zero': 'correct'}}
ertype = {'a': {'a': 'correct_a', 'the': 'sub_the_inst_a', 'zero': 'omit_a'},
          'the': {'a': 'sub_a_inst_the', 'the': 'correct_the', 'zero': 'omit_the'},
          'zero': {'a': 'over_a', 'the': 'over_the', 'zero': 'correct_zero'}}
ertype2 = {'a': {'a': 'correct_art', 'the': 'error_art', 'zero': 'omit'},
          'the': {'a': 'error_art', 'the': 'correct_art', 'zero': 'omit'},
          'zero': {'a': 'over', 'the': 'over', 'zero': 'correct_om'}}

def extract_article(word_idx, words):
    noun = words[word_idx]

    for w in words:
        if w.head == noun.id and w.deprel == "det":
            det = w.text.lower()
            if det in {"a", "the"}:
                return det
    
    return "zero"

def check_error(l1_idx, l1_words, l2_words, alignments, errordict):
    l1_article = extract_article(l1_idx, l1_words)
    # Find aligned word in L2
    for l2_i, l1_i in alignments:
        if l1_i - 1 == l1_idx:
            l2_article = extract_article(l2_i - 1, l2_words)
            return errordict[l1_article][l2_article]

def get_ntype(word):
    if 'Number=Plur' in word.feats:
        return 'plural'
    return 'sing'

subj_deprels = {'subj', 'csubj', 'csubjpass', 'nsubj', 'nsubjpass', 'xobj'}
obj_deprels = {'comp', 'acomp', 'ccomp', 'xcomp', 'obj', 'dobj', 'iobj', 'pobj', 'xobj', 'obl'}

def get_syntactic_role(word, sentence):
    if word.deprel in subj_deprels:
        return 'sub'
    elif word.deprel in obj_deprels:
        return 'obj'
    elif word.deprel == 'appos':
        return 'pred/prop'
    siblings = [w for w in sentence if w.head == word.head and w != word]
    for sib in siblings:
        if sib.deprel == 'expl':
            return 'ex'
    return None

modifiers = {"amod", "num", "perp", "poss", "possessive", "quantmod", "rcmod"}
def has_premodifier(word_idx, words):
    noun = words[word_idx]

    for w in words:
        if w.head == noun.id and w.deprel in modifiers and w.id < noun.id:
            return "mod"
    
    return "no_mod"


data_1 = []
excluded_dets = {
    "my", "your", "his", "her", "its", "our", "their", "whose",
    "this", "that", "these", "those"
}
test_sentences = []
positions = []

for j, row in corpus.iterrows():
    l1_sent = row.iloc[0]
    l2_sent = row.iloc[1]
    alignments = row.iloc[2]
    nl = row.iloc[3]
        
    l1_doc = nlp(l1_sent)
    l2_doc = nlp(l2_sent)
        
    l1_words = l1_doc.sentences[0].words
    l2_words = l2_doc.sentences[0].words

    if len(l1_doc.sentences) > 1:
        l1_words = [w for sent in l1_doc.sentences for w in sent.words]
    if len(l2_doc.sentences) > 1:
        l2_words = [w for sent in l2_doc.sentences for w in sent.words]
    
    if l1_words[0].text == '"':
        l1_words = l1_words[1:]
    if l2_words[0].text == '"':
        l2_words = l2_words[1:]

    for i, word in enumerate(l1_words):
        if word.upos == 'NOUN' and word.text in l2_sent:
            has_excluded_det = any(
                (w.deprel == "det" and w.head == word.id and w.text.lower() in excluded_dets) or
                (w.deprel == "nmod:poss" and w.head == word.id)  # possessive nominal modifiers like "teacher's"
                for w in l1_words
            )
            l2_idx = None
            for l2_i, l1_i in alignments:
                if l1_i - 1 == i:
                    l2_idx = l2_i - 1
                    break

            if l2_idx is not None:
                has_excluded_det_2 = any(
                    (w.deprel == "det" and w.head == l2_words[l2_idx].id and w.text.lower() in excluded_dets) or
                    (w.deprel == "nmod:poss" and w.head == l2_words[l2_idx].id)
                    for w in l2_words
                )
                article = extract_article(i, l1_words)
                synt = get_syntactic_role(word, l1_words)
                if not has_excluded_det and not has_excluded_det_2 and article in {"a", "the", "zero"} and synt is not None:
                    row = {
                        'L1': l1_sent,
                        'L2': l2_sent,
                        'NL': nl,
                        'HeadN': word.text,
                        'Oblig': "obl_om" if article == "zero" else "obl",
                        'Error': check_error(i, l1_words, l2_words, alignments, error),
                        'Target': article,
                        'ErType': check_error(i, l1_words, l2_words, alignments, ertype),
                        'ErType2': check_error(i, l1_words, l2_words, alignments, ertype2),
                        'Ntype': get_ntype(word),
                        'Synt': get_syntactic_role(word, l1_words),
                        'modif': has_premodifier(l2_idx, l2_words)
                    }
                    data_1.append(row)
                    print("Data Number:", j+1)
                    print("L1 Sentence:", " ".join(w.text for w in l1_words))
                    print("L2 Sentence:", " ".join(w.text for w in l2_words))
                    print("Target NOUN:", word.text)
                    print("Row:", row)
                    test_sentences.append([w.text for w in l2_words])
                    positions.append(l2_idx)
                    print("-" * 60)

    if j >= 49999:
        break

df = pd.DataFrame(data_1)

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
        annotated_data_path="datasets/annotated_data.csv"
    )

pred_abstr = predict_words(clf_abstr, nlp_abstr, ['concr', 'abstr'], nouns)
pred_abstr = [label for _, label in pred_abstr]

clf_count, nlp_count = classifier(
        path1="logistic_regression/countable.txt",
        path2="logistic_regression/uncountable.txt", 
        annotated_data_path="datasets/annotated_data.csv"
    )

pred_count = predict_words(clf_count, nlp_count, ['mass', 'count'], nouns)
pred_count = [label for _, label in pred_count]

synthetic_2 = pd.read_csv("synthetic_2.csv")

df["Abstract"] = pred_abstr
df["Rev_abstr"] = pred_abstr

for idx, value in enumerate(pred_count):
    if value == 'mass':
        df.at[idx, 'Ntype'] = 'mass'

df = df.rename(columns={"L2": "NP"})
df["item_id"] = range(1, 1 + len(df))
df = df[["item_id", "NL", "NP", "Oblig", "Error", "Target", "ErType", "ErType2", "def", "ref", "Hawkins", "HeadN", "Ntype", "Abstract", "Rev_abstr", "Synt", "modif"]]
df.to_csv("results/synthetic_dataset.csv", index=False)