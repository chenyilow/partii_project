import os
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import stanza
import json

col = "ref" # Uses: "def", "Hawkins"

def label_map(col):
    if col == "ref":
        return {0: "CLS/SEP/PAD", 1: "-", 2: "pred/prop", 3: "ref", 4: "nonref", 5: "id_pred"}
    elif col == "def":
        return {0: "CLS/SEP/PAD", 1: "-", 2: "def", 3: "indef"}
    elif col == "Hawkins":
        return {0: "CLS/SEP/PAD", 1: "-", 2: "indef", 3: "situational", 4: "explanatory", 5: "anaphoric", 6: "kind"}
    else:
        return None

def get_latest_checkpoint(model_dir: str) -> str:
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(model_dir, checkpoints[-1]) if checkpoints else model_dir

def load_model_and_tokenizer(label_col: str):
    # ref had the highest accuracy for fold_6, def and Hawkins for fold_10
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if label_col == "ref":
        model_base_dir = os.path.join(script_dir, "results", label_col, "fold_6")
    elif label_col == "def" or label_col == "Hawkins":
        model_base_dir = os.path.join(script_dir, "results", label_col, "fold_10")
    else:
        return None
    
    checkpoint = get_latest_checkpoint(model_base_dir)
    print(f"Using model checkpoint: {checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    return tokenizer, model

def predict_labels(sentences: List[List[str]], tokenizer, model, col):
    model.eval()
    all_predictions = []
    count = 0
    for sentence in sentences:
        count += 1
        print(count)
        inputs = tokenizer(sentence, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
        word_ids = inputs.word_ids(batch_index=0)

        word_predictions = []
        prev_word_idx = None
        for pred, word_idx in zip(predictions, word_ids):
            if word_idx is None or word_idx == prev_word_idx:
                continue
            word_predictions.append(label_map(col)[pred])
            prev_word_idx = word_idx

        all_predictions.append(word_predictions)

    return all_predictions

if __name__ == "__main__":
    tokenizer_def, model_def = load_model_and_tokenizer("def")
    tokenizer_ref, model_ref = load_model_and_tokenizer("ref")
    tokenizer_hawkins, model_hawkins = load_model_and_tokenizer("Hawkins")
    
    nlp = stanza.Pipeline(lang='en', processors={"tokenize": "spaCy"})

    l1 = []
    l2 = []
    for line in open("L1 Texts"):
        l1.append(line.strip())
    for line in open("L2 Texts"):
        l2.append(line.strip())
    with open("alignments.json", "r") as f:
        alignments = json.load(f)

    corpus = pd.DataFrame([l1, l2, alignments]).T
    corpus.columns = ["L1", "L2", "Alignment"]

    synthetic_1 = pd.read_csv("synthetic_1.csv")
    
    corpus = corpus.drop_duplicates(subset=["L1", "L2"], keep="first")
    merged_df = synthetic_1.merge(corpus, on=["L1", "L2"], how="left")

    test_sentences = []
    positions = []

    for j, row in merged_df.iterrows():
        l2_doc = nlp(row["L2"])
        l2_words = l2_doc.sentences[0].words
        if len(l2_doc.sentences) > 1:
            l2_words = [w for sent in l2_doc.sentences for w in sent.words]
        l2_words = [w.text for w in l2_words]
        test_sentences.append(l2_words)
        positions.append(row["Position"])
        print(j)

    predictions_def = predict_labels(test_sentences, tokenizer_def, model_def, "def")
    predictions_hawkins = predict_labels(test_sentences, tokenizer_hawkins, model_hawkins, "Hawkins")
    predictions_ref = predict_labels(test_sentences, tokenizer_ref, model_ref, "ref")
    data = []

    for i, sentence in enumerate(test_sentences):
        pos = positions[i]
        data.append({
            "NP": " ".join(sentence),
            "HeadN": sentence[pos],
            "def": predictions_def[i][pos],
            "ref": predictions_ref[i][pos],
            "Hawkins": predictions_hawkins[i][pos]
        })

    df = pd.DataFrame(data)
    df.to_csv("synthetic_2_new.csv", index=True)
    # print(test_sentences, predictions, positions)
    # for sent, preds, position in zip(test_sentences, predictions, positions):
    #     preds = list(zip(sent, preds, position))
    #     print(preds)

        # for i, word in enumerate(l2_words):
        #     if word.upos == 'NOUN' and word.text in l2_sent:
        #         print(word.text)
    #     test_sentences.append(tokens)
    #     if i == 50:
    #         break
    #     print(i, tokens, doc_1)


    # predictions = predict_labels(test_sentences, tokenizer, model, col)

    # for sent, preds in zip(test_sentences, predictions):
    #     preds = list(zip(sent, preds))
    #     print(preds)