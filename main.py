import os
from log_reg.nltk import *
#from dataloader.conll import *
import pandas as pd

def logreg(nouns, file_1, file_2, labels):
    # Define paths - adjust these as needed for your directory structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_reg_dir = os.path.join(base_dir, 'log_reg')
    
    # Paths to required files
    # abstract.txt retrieved from concrete score 100-300, NOUNs only from https://websites.psychology.uwa.edu.au/school/mrcdatabase/uwa_mrc.htm
    # concrete.txt retrieved from concrete score 500-700, NOUNs only from https://websites.psychology.uwa.edu.au/school/mrcdatabase/uwa_mrc.htm

    # uncountable.txt retrieved from https://www.aprendeinglesenleganes.com/resources/1147%20UNCOUNTABLE%20NOUNS%20-%20LIST.pdf
    # countable.txt retrieved from https://mrmrsenglish.com/countable-nouns-list-in-english/#A_words_list_of_Countable_nouns
    abstract_path = os.path.join(log_reg_dir, file_1)
    concrete_path = os.path.join(log_reg_dir, file_2)
    annotated_data_path = os.path.join(log_reg_dir, 'annotated_data.csv')
    
    # Call classifier with full paths
    clf, nlp = classifier(
        path1=abstract_path,
        path2=concrete_path, 
        annotated_data_path=annotated_data_path
    )

    #print(classifier_accuracy(clf, nlp, ["concr", "abstr"], "Rev_abstr", annotated_data_path))

    return predict_words(clf, nlp, labels, nouns)

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataloader_dir = os.path.join(base_dir, "data")
    conll_dir = os.path.join(dataloader_dir, "L2 Texts")

    for line in open(conll_dir):
        line = tokenize_and_pos(line)
        print(line)

def main():
    df = pd.read_csv("synthetic_1.csv")
    nouns = df["HeadN"].to_list()
    pred_abstr = logreg(nouns, 'abstract.txt', 'concrete.txt', ['concr', 'abstr'])
    pred_abstr = [label for _, label in pred_abstr]

    pred_count = logreg(nouns, 'countable.txt', 'uncountable.txt', ['mass', 'count'])
    pred_count = [label for _, label in pred_count]

    synthetic_2 = pd.read_csv("synthetic_2.csv")

    df["def"] = synthetic_2["def"]
    df["ref"] = synthetic_2["ref"]
    df["Hawkins"] = synthetic_2["Hawkins"]

    df["Abstract"] = pred_abstr
    df["Rev_abstr"] = pred_abstr

    for idx, value in enumerate(pred_count):
        if value == 'mass':
            df.at[idx, 'Ntype'] = 'mass'
    print(df)

    df = df.rename(columns={"L2": "NP"})
    df = df[["item_id", "NP", "Oblig", "Error", "Target", "ErType", "ErType2", "def", "ref", "Hawkins", "HeadN", "Ntype", "Abstract", "Rev_abstr", "Synt", "modif"]]
    df.to_csv("synthetic_dataset.csv", index=False)

if __name__ == "__main__":
    main()