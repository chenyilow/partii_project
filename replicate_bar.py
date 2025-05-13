import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

plt.rcParams.update({'font.size': 20})

class MMELR:
    def __init__(self, csv_path="results/synthetic_dataset.csv"):
        self.csv_path = csv_path
        self.df = None
        self.model = None

    def load_dataframe(self, general=False):
        df = pd.read_csv(self.csv_path)
        
        df = df[(df["def"] != "-") & (df["ref"] != "-") & (df["Hawkins"] != "-")]
        df = df[df["ref"] != "pred/prop"]
        
        df["ref"] = df["ref"].replace({"ref": "spec", "nonref": "nonspec"})
        df = df.rename(columns={"def": "definiteness"})

        if general:
            df.loc[df["Target"].isin(["a", "zero"]), "Target"] = "indef"
            df.loc[df["Target"] == "the", "Target"] = "def"

        model.df = df
        return df

    def prepare_data(self, df):
        df = df.copy()
        df['Score'] = (df['Error'] == 'correct').astype(int)

        keep_nls = ["Japanese", "Mandarin", "Korean", "Russian"]
        df = df[df["NL"].isin(keep_nls)]

        df_target_encoded = pd.get_dummies(df['Target'], prefix='Target', drop_first=True)
        df_nl_encoded = pd.get_dummies(df['NL'], prefix='NL', drop_first=True)
        df_ntype_encoded = pd.get_dummies(df['Ntype'], prefix='Ntype', drop_first=True)
        df_synt_encoded = pd.get_dummies(df['Synt'], prefix='Synt', drop_first=True).rename(columns={'Synt_pred/prop': 'Synt_predprop'})

        df = df.drop(['Target', 'NL', 'Ntype', 'Synt'], axis=1)
        df = pd.concat([df, df_target_encoded, df_nl_encoded, df_ntype_encoded, df_synt_encoded], axis=1)
        
        df.to_csv("check.csv")
        return df

    def train_logistic_model(self, df, formula):
        model = smf.logit(
            formula,
            data=df
        ).fit(method='bfgs', maxiter=1000)
        print(model.summary())
        return model

    # def impute_missing_values(self, df, expected_features=None):
    #     df = df.copy()
    #     if expected_features is None:
    #         expected_features = ['Ntype_sing', 'Ntype_plural', 'NL_Mandarin', 'NL_Korean', 'NL_Russian',
    #                             'ref', 'modif', 'Abstract', 'Synt_sub', 'Synt_obj', 'Synt_predprop']

    #     for feature in expected_features:
    #         if feature in df.columns:
    #             mode_value = df[feature].mode().iloc[0]
    #             df.loc[:, feature] = df[feature].fillna(mode_value)
    #     return df


    def predict(self, model, df):
        """Make predictions using the trained model."""
        X = df.drop(columns=['Score'])
        predictions = model.predict(X)
        
        # Calculate mean and 95% confidence interval
        mean_prob = np.mean(predictions)
        std_prob = np.std(predictions)

        n = len(predictions)
        z = stats.norm.ppf(0.975)
        margin_of_error = z * (std_prob / np.sqrt(n))

        ci_lower = mean_prob - margin_of_error
        ci_upper = mean_prob + margin_of_error

        print(f"Mean predicted probability: {mean_prob}")
        print(f"95% Confidence Interval: [{ci_lower}, {ci_upper}]")

        return mean_prob, ci_lower, ci_upper

    def plot_error_bar(self, model, df, col, col_name, categories, filename, definiteness):
        """
        Plot accuracy with 95% CI error bars for Ntype_sing, Ntype_mass, Ntype_pl,
        further split by specificity (values in ref_col, e.g., 'spec' vs. 'nonspec').
        """
        df['Ntype_mass'] = (~df['Ntype_sing']) & (~df['Ntype_plural'])
        ntype_cols = ['Ntype_sing', 'Ntype_mass', 'Ntype_plural']
        ntype_labels = ['Singular', 'Mass', 'Plural']
        bar_data = []

        for ntype_col, ntype_label in zip(ntype_cols, ntype_labels):
            for val in categories:
                subset = df[(df[ntype_col] == True) & (df[col] == val) & (df["definiteness"] == definiteness)].copy()
                if subset.empty:
                    continue

                subset['Score'] = (subset['Error'] == 'correct').astype(int)
                print(ntype_label, val, definiteness)
                mean_prob, ci_lower, ci_upper = self.predict(model, subset)

                bar_data.append({
                    'Ntype': ntype_label,
                    'Col': col_name[val],
                    'mean_prob': mean_prob,
                    'error_margin': ci_upper - mean_prob
                })

        plot_df = pd.DataFrame(bar_data)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=plot_df,
            x='Ntype', y='mean_prob', hue='Col',
            errorbar=None, capsize=0.1, palette='Set1',
            hue_order=[col_name[categories[1]], col_name[categories[0]]],
            err_kws={'color': 'grey'}, edgecolor='black'
        )

        plt.text(
            x=len(plot_df['Ntype'].unique()) - 0.5,  # Position near the last x-tick
            y=1.25,  # Position just above the top of the bars (adjust as necessary)
            s=definiteness,  # The text to display
            ha='center',  # Horizontal alignment
            va='center',  # Vertical alignment
            color='black'  # Color of the text
        )
        
        for i, row in plot_df.iterrows():
            x = i // 2 + (-0.2 if row['Col'] == col_name[categories[1]] else 0.2)
            plt.errorbar(x=x, y=row['mean_prob'], yerr=row['error_margin'],
                        fmt='none', c='black', capsize=5)

        plt.ylabel("Accuracy")
        plt.xlabel("Countability")  
        plt.ylim(0, 1.2)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

# --- Generate MMELR Table ---

# model = MMELR(csv_path="results/synthetic_dataset.csv")
# df = model.load_dataframe()
# df = model.prepare_data(df)

# saved_model = model.train_logistic_model(df, 
#     "Score ~ (Target_the + Target_zero) * (Ntype_sing + Ntype_plural) * (NL_Mandarin + NL_Korean + NL_Russian + ref + modif) + Abstract + Synt_sub + Synt_obj + Synt_predprop"
# )

# --- Generate Error Bars ---

model = MMELR(csv_path="results/synthetic_dataset.csv")
df = model.load_dataframe()
df = model.prepare_data(df)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

saved_model = model.train_logistic_model(train_df, 
    "Score ~ (Target_the + Target_zero) * (Ntype_sing + Ntype_plural) * (NL_Mandarin + NL_Korean + NL_Russian + ref + modif) + Abstract + Synt_sub + Synt_obj + Synt_predprop"
)

model.plot_error_bar(saved_model, test_df, col='ref', col_name={"spec": "Specific", "nonspec": "Non-specific"}, categories=["spec", "nonspec"], filename="ref_def.png", definiteness="def")
model.plot_error_bar(saved_model, test_df, col='modif', col_name={"mod": "Modifier", "no_mod": "No Modifier"}, categories=["mod", "no_mod"], filename="modif_def.png", definiteness="def")
model.plot_error_bar(saved_model, test_df, col='ref', col_name={"spec": "Specific", "nonspec": "Non-specific"}, categories=["spec", "nonspec"], filename="ref_indef.png", definiteness="indef")
model.plot_error_bar(saved_model, test_df, col='modif', col_name={"mod": "Modifier", "no_mod": "No Modifier"}, categories=["mod", "no_mod"], filename="modif_indef.png", definiteness="indef")