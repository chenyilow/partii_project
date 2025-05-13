import stanza
import logging
import pandas as pd

# Suppress stanza logs
logging.getLogger('stanza').setLevel(logging.WARNING)

# Initialize Stanza NLP pipeline
nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'})

def tokenize_and_pos(paragraph):
    """ Tokenizes text and extracts POS tags in one pass """
    doc = nlp(paragraph)
    return [[word.text for word in sentence.words] for sentence in doc.sentences], \
           [[word.upos for word in sentence.words] for sentence in doc.sentences]

def conll(columns):
    # Read and process dataset
    corpus = pd.read_csv("annotated_data.csv")
    corpus.sort_values(by='item_id', ascending=True, inplace=True)
    uniqueTexts = corpus.drop_duplicates(["Text1"])["Text1"]
    feature_df = corpus[["Text1", "HeadN"] + columns]

    exclude_words = {"my", "his", "her", "their", "your", "this", "that"}
    output = []
    count = 0
    total = len(uniqueTexts)

    for paragraph in uniqueTexts[1]:
        noun_data = feature_df[feature_df["Text1"] == paragraph].drop(columns=["Text1"])
        
        # Precompute noun dictionary
        noun_dict = {key.lower(): group.drop(columns=["HeadN"]).values.tolist() for key, group in noun_data.groupby("HeadN")}
        
        paragraph = paragraph.strip()
        wordlist, poslist = tokenize_and_pos(paragraph)

        data = []
        for words, pos_tags in zip(wordlist, poslist):
            prev_word = None
            for word, pos in zip(words, pos_tags):
                row = [word, pos] + ["-"] * len(columns)
                word = word.lower()
                if pos == "NOUN" and (prev_word not in exclude_words if prev_word else True):
                    if word in noun_dict and noun_dict[word]:
                        row[2:] = noun_dict[word].pop(0)  # Assign and remove first available annotation
                data.append(row)
                prev_word = word
            data.append(["-"] * (2 + len(columns)))
        output += data
        output.append([None] * (2 * len(columns)))  # Separator row
        count += 1
        print(f"Completed sentence {count} of {total}")

    # Save results to CSV
    print(output)
    pd.DataFrame(output, columns=["Word", "POS Tag"] + columns).to_csv("conll.csv", index=False)

conll(["def", "ref", "Hawkins"])