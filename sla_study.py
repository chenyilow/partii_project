import pandas as pd
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
plt.rcParams.update({'font.size': 12})

def load_df(general=False):
    # Returns a dataframe of all rows that are fully filled and only has ref or nonref for specificity
    df = pd.read_csv("results/synthetic_dataset.csv")
    df = df[df["def"] != "-"]
    df = df[df["ref"] != "-"]
    df = df[df["Hawkins"] != "-"]
    df = df[df["ref"] != "pred/prop"]
    df["ref"] = df["ref"].replace({"ref": "spec", "nonref": "nonspec"})
    if general: # Modifies only for mosaic plots for general trends
        df.loc[df["Target"] == "the", "Target"] = "def"
        df.loc[df["Target"] == "a", "Target"] = "indef"
        df.loc[df["Target"] == "zero", "Target"] = "indef"
    return df

def load_mosaic(df, columns, name, category_orders=None):
    # Apply custom category ordering before grouping
    if category_orders:
        for col, order in category_orders.items():
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)

    grouped_counts = df.groupby(columns).size()
    print(grouped_counts)
    _, ax = plt.subplots(figsize=(8, 6))
    mosaic(grouped_counts, ax = ax, horizontal=False, labelizer=lambda key: grouped_counts[key])
    plt.savefig("visualisations/" + name + ".png", bbox_inches='tight')
    plt.show()

def load_general_ref():
    df = load_df(general=True)
    load_mosaic(
        df,
        columns=["Target", "Ntype", "ref"],
        name="mosaic_1",
        category_orders={"Target": ["indef", "def"], 
                        "Ntype": ["sing", "mass", "plural"],
                        "ref": ["nonspec", "spec"]}
    )

def load_general_modifier():
    df = load_df(general=True)
    load_mosaic(
        df,
        columns=["Target", "Ntype", "modif"],
        name="mosaic_2",
        category_orders={"Target": ["indef", "def"], 
                        "Ntype": ["sing", "mass", "plural"],
                        "modif": ["no_mod", "mod"]}
    )

def load_general_abstractness():
    df = load_df(general=True)
    load_mosaic(
        df,
        columns=["Target", "Ntype", "Rev_abstr"],
        category_orders={"Target": ["indef", "def"], 
                        "Ntype": ["sing", "mass", "plural"],
                        "Rev_abstr": ["concr", "abstr"]}
    )

def load_general_synt():
    df = load_df(general=True)
    load_mosaic(
        df,
        columns=["Target", "Ntype", "Synt"],
        category_orders={"Target": ["indef", "def"], 
                        "Ntype": ["sing", "mass", "plural"],
                        "Synt": ["pred/prop", "sub", "obj"]}
    )

def load_general_obl_error():
    df = load_df()
    df = df[df["Error"] == "error"]

    load_mosaic(
        df,
        columns=["Target", "Ntype", "ErType2"],
        category_orders={"Target": ["the", "a"],
                         "Ntype": ["sing", "mass", "plural"],
                         "ErType2": ["omit", "error_art"]}
    )

def load_general_zero_error():
    df = load_df()
    df = df[df["Error"] == "error"]

    load_mosaic(
        df,
        columns=["Target", "Ntype", "ErType"],
        category_orders={"Target": ["zero"],
                         "Ntype": ["sing", "mass", "plural"],
                         "ErType": ["over_a", "over_the"]}
    )

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats

def load_and_prepare_data():
    """Load and preprocess the dataset."""
    df = load_df()
    df['Score'] = (df['Error'] == 'correct').astype(int)

    keep_nls = ["Japanese", "Mandarin", "Korean", "Russian"]
    df = df[df["NL"].isin(keep_nls)]

    df_nl_encoded = pd.get_dummies(df['NL'], prefix='NL', drop_first=True)
    df_ntype_encoded = pd.get_dummies(df['Ntype'], prefix='Ntype', drop_first=True)
    df_synt_encoded = pd.get_dummies(df['Synt'], prefix='Synt', drop_first=True).rename(columns={'Synt_pred/prop': 'Synt_predprop'})

    df = df.drop(['NL', 'Ntype', 'Synt'], axis=1)
    df = pd.concat([df, df_nl_encoded, df_ntype_encoded, df_synt_encoded], axis=1)
    
    return df

def train_logistic_model(df):
    """Train a logistic regression model using the prepared data."""
    model = smf.logit(
        "Score ~ Target * (Ntype_sing + Ntype_plural) * (NL_Mandarin + NL_Korean + NL_Russian + ref + modif) + Abstract + Synt_sub + Synt_obj + Synt_predprop",
        data=df
    ).fit(method='bfgs', maxiter=1000)
    print(model.summary())
    return model

def impute_missing_values(df_japanese, categorical_features, numerical_features):
    """Impute missing values in the dataset."""
    # Impute categorical features with mode
    for feature in categorical_features:
        mode_value = df_japanese[feature].mode().iloc[0]
        df_japanese[feature].fillna(mode_value, inplace=True)
    
    # Impute numerical features with mean
    for feature in numerical_features:
        mean_value = df_japanese[feature].mean()
        df_japanese[feature].fillna(mean_value, inplace=True)
    
    return df_japanese

def compute_confidence_interval(predictions):
    """Compute the mean and 95% confidence interval of predictions."""
    mean_prob = np.mean(predictions)
    std_prob = np.std(predictions)
    n = len(predictions)
    z = stats.norm.ppf(0.975)  # z-score for 95% confidence

    # Calculate margin of error
    margin_of_error = z * (std_prob / np.sqrt(n))

    # Confidence Interval
    ci_lower = mean_prob - margin_of_error
    ci_upper = mean_prob + margin_of_error

    return mean_prob, ci_lower, ci_upper

def process_for_predictions(df, model, col, target):
    """Main function to prepare data, train model, and compute confidence intervals."""
    # Load and preprocess data
    df = load_and_prepare_data()

    # Train the logistic regression model
    model = train_logistic_model(df)

    # Filter data for Japanese language
    df_japanese = df[df['NL_Korean'] == 1]  # Assuming 'NL_Korean' represents Japanese

    # Define features
    categorical_features = ['Ntype_sing', 'Ntype_plural', 'NL_Mandarin', 'NL_Korean', 'NL_Russian', 'ref', 'modif', 'Abstract', 'Synt_sub', 'Synt_obj', 'Synt_predprop']
    numerical_features = []  # Add more numerical features if necessary

    # Impute missing values in the dataset
    df_japanese = impute_missing_values(df_japanese, categorical_features, numerical_features)

    # Drop the target column for prediction
    X_japanese = df_japanese.drop(columns=['Score'])

    # Reindex to match the model's expected input
    X_japanese = X_japanese.reindex(columns=model.model.exog_names, fill_value=0)

    # Make predictions
    predictions = model.predict(X_japanese)

    # Compute the mean and confidence interval of the predictions
    mean_prob, ci_lower, ci_upper = compute_confidence_interval(predictions)

    # Print the results
    print(f"Mean of predicted probabilities: {mean_prob}")
    print(f"95% Confidence Interval: [{ci_lower}, {ci_upper}]")

    ordered = ['sing', 'mass', 'plural']
    if col == "ref":
        col_name = "ref"
        col_list = ["spec", "nonspec"]
    elif col == "modif":
        col_name = "modif"
        col_list = ["mod", "no_mod"]
    else:
        raise ValueError("Invalid column")

    # Get predicted probabilities per row
    df["predicted_accuracy"] = model.predict(df)

    # Aggregate predictions and compute CIs per group
    grouped = df.groupby(["Ntype", col_name])
    bar_data = []
    for (ntype, label), group in grouped:
        if ntype not in ordered or label not in col_list:
            continue
        mean = group["predicted_accuracy"].mean()
        std = group["predicted_accuracy"].std()
        n = len(group)
        ci = 1.96 * std / np.sqrt(n)  # 95% CI
        bar_data.append((ntype, label, mean * 100, ci * 100))

    # Prepare data for plotting
    x = np.arange(len(ordered))
    width = 0.35
    blue_data = [d for d in bar_data if d[1] == col_list[0]]
    orange_data = [d for d in bar_data if d[1] == col_list[1]]

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, (ntype, _, acc, err) in enumerate(blue_data):
        xpos = x[i] - width / 2
        ax.bar(xpos, acc, width, label=col_list[0] if i == 0 else "", color="C0")
        ax.errorbar(xpos, acc, yerr=[[err], [err]], fmt='none', ecolor='black', capsize=5)
        ax.text(xpos, acc + 1, f'{acc:.0f}%', ha='center', va='bottom')

    for i, (ntype, _, acc, err) in enumerate(orange_data):
        xpos = x[i] + width / 2
        ax.bar(xpos, acc, width, label=col_list[1] if i == 0 else "", color="C1")
        ax.errorbar(xpos, acc, yerr=[[err], [err]], fmt='none', ecolor='black', capsize=5)
        ax.text(xpos, acc + 1, f'{acc:.0f}%', ha='center', va='bottom')

    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 110, 10))
    ax.set_yticklabels([f'{i}%' for i in range(0, 110, 10)])
    ax.set_xticks(x)
    ax.set_xticklabels(ordered)
    ax.set_ylabel('accuracy rate')
    ax.legend()
    plt.tight_layout()
    plt.show()


# load_error_bars("ref", "def")
# load_error_bars("ref", "indef")
# load_error_bars("modif", "def")
# load_error_bars("modif", "indef")

def load_error_synt():
    df = load_df()
    df = df[df["Synt"] != "appos"]
    df = df[df["Synt"] != "gen"]
    df = df[df["Synt"] != "other"]
    counts = df.groupby(["Synt", "Error"]).size()
    counts = counts.unstack(fill_value=0)
    counts["accuracy"] = counts["correct"] / (counts["correct"] + counts["error"]) * 100
    ordered = ["obj", "sub", "pred/prop"]
    x = np.arange(len(ordered))
    width = 0.6

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, counts["accuracy"], width, color='skyblue')

    # Annotate bars
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{counts["accuracy"][i]:.1f}%', ha='center', va='bottom')

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(ordered)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 110, 10))
    ax.set_yticklabels([f'{i}%' for i in range(0, 110, 10)])
    ax.set_ylabel('accuracy rate')
    ax.set_title('Accuracy by Syntactic Role')
    plt.tight_layout()
    plt.show()

# load_general_ref()
# load_general_modifier()
# load_general_abstractness()
# load_general_synt()
# load_general_obl_error()
# load_general_zero_error()
process("ref", "def")
# load_error_bars("ref", "indef")
# load_error_bars("modif", "def")
# load_error_bars("modif", "indef")
# load_error_synt()

# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.formula.api import mixedlm

# # Assuming you have a DataFrame `df` with columns for the fixed effects and random effect
# # df = pd.read_csv('your_data.csv')

# # Define the formula for the mixed model
# # Fixed effects: NL, level, specificity, modifier, abstractness, syntactic position
# # Random effects: (1|wr_ID)

# df = load_df()
# from sklearn.preprocessing import LabelEncoder

# # Label encode the 'ErType' column
# le = LabelEncoder()
# df["ErType"] = le.fit_transform(df["ErType"])
# formula = "ErType ~ ref * (modif + Abstract) + Synt"

# # Define the mixed effects model
# # Random effect: (1 | wr_ID), this indicates random intercept for 'wr_ID'
# model = mixedlm(formula, data=df, groups=df["item_id"])

# # Fit the model
# result = model.fit()

# # Print the summary of the model
# print(result.summary())