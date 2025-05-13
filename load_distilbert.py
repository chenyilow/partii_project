import stanza
from definiteness.distilbert.distilbert_prediction import *

l1 = []
l2 = []
for line in open("dataloader/L1 Texts"):
    l1.append(line.strip())
for line in open("dataloader/L2 Texts"):
    l2.append(line.strip())
data = pd.DataFrame([l1, l2]).T
data.columns = ["L1", "L2"]

# Group sentences
df = pd.read_csv("dataloader/aligned.csv")
df = pd.merge(data, df, left_on="L2", right_on="NP")
sentences = df["L1"].unique()

# Initialize the Stanza pipeline
stanza.download("en")
nlp = stanza.Pipeline("en", processors={"tokenize": "spaCy"}, processors_to_skip=["mwt", "lemma"])

tokenizer_ref, model_ref = load_model_and_tokenizer("ref")
tokenizer_def, model_def = load_model_and_tokenizer("def")
tokenizer_hawkins, model_hawkins = load_model_and_tokenizer("Hawkins")

all_tokens = []
all_noun_tokens = []
sent_map = []

count = 0
for sent in sentences[:20]:
    doc = nlp(sent)

    for sentence in doc.sentences:
        tokens = [word.text for word in sentence.words]
        noun_tokens = [word.text for word in sentence.words if word.upos == "NOUN"]

        all_tokens.append(tokens)
        all_noun_tokens.append(noun_tokens)
        sent_map.append(" ".join(tokens))

        count += 1
        print(count)

predicted_all_ref = predict_labels(all_tokens, tokenizer_ref, model_ref, "ref")
predicted_all_def = predict_labels(all_tokens, tokenizer_def, model_def, "def")
predicted_all_hawkins = predict_labels(all_tokens, tokenizer_hawkins, model_hawkins, "Hawkins")

for sent, tokens, noun_tokens, predicted_labels_ref, predicted_labels_def, predicted_labels_hawkins in zip(sent_map, all_tokens, all_noun_tokens, predicted_all_ref, predicted_all_def, predicted_all_hawkins):
    # Get the predicted labels for each noun from all 3 prediction lists
    doc_noun_labels = [
        (pred_ref, pred_def, pred_hawkins)  # Group the predictions together as tuples
        for word, pred_ref, pred_def, pred_hawkins in zip(tokens, predicted_labels_ref, predicted_labels_def, predicted_labels_hawkins)
        if word in noun_tokens
    ]
    
    # Find the indices in the DataFrame corresponding to this sentence
    noun_indices = df[df["L1"] == sent].index

    # Update the DataFrame with all 3 predicted labels for each noun
    for i, idx in enumerate(noun_indices):
        if i < len(doc_noun_labels):
            # Assign the tuple of predictions for each noun
            df.at[idx, "ref"] = doc_noun_labels[i][0]
            df.at[idx, "def"] = doc_noun_labels[i][1]
            df.at[idx, "Hawkins"] = doc_noun_labels[i][2]

df.to_csv("predicted_output_test.csv", index=False)
print("Predictions added and saved to predicted_output_test.csv")
