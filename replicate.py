# Copyright 2025 [Your Name]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
plt.rcParams.update({'font.size': 18})

class MMELR():
    def __init__(self, csv_path="results/synthetic_dataset.csv"):
        self.csv_path = csv_path
        self.df = None

    def load_dataframe(self, general=False):
        df = pd.read_csv(self.csv_path)
        df = df[(df["def"] != "-") & (df["ref"] != "-") & (df["Hawkins"] != "-")]
        df = df[df["ref"] != "pred/prop"]
        df["ref"] = df["ref"].replace({"ref": "spec", "nonref": "nonspec"})
        df = df.rename(columns={"def": "definiteness"})

        if general:
            df.loc[df["Target"].isin(["a", "zero"]), "Target"] = "indef"
            df.loc[df["Target"] == "the", "Target"] = "def"

        self.df = df

    def generate_mosaic(self, columns, filename, category_orders=None):
        if self.df is None:
            raise ValueError("Dataframe not loaded. Call load_dataframe() first.")

        df = self.df.copy()

        if category_orders:
            for col, order in category_orders.items():
                df[col] = pd.Categorical(df[col], categories=order, ordered=True)

        grouped_counts = df.groupby(columns).size()
        _, ax = plt.subplots(figsize=(8, 6))
        mosaic(grouped_counts, ax=ax, horizontal=False, labelizer=lambda key: grouped_counts[key])
        plt.savefig(f"visualisations/{filename}.png", bbox_inches='tight')
        plt.show()

    def train_logistic_regression(self, formula, to_map=None):
        """
        Fits a logistic regression model on the DataFrame using the given formula.

        Parameters:
        - formula (str): statsmodels formula, e.g., "definiteness ~ ref + Ntype + modifier"
        - to_map (dict): optional dict specifying how to map categorical columns to integers.
                        e.g., {"ref": {"spec": 1, "nonspec": 0}, "Ntype": {"sing": 0, "mass": 1, "plural": 2}}
        """
        self.load_dataframe(general=True)

        if self.df is None:
            raise ValueError("Dataframe not loaded. Call load_dataframe() first.")

        df_copy = self.df.copy()

        if to_map:
            for col, mapping in to_map.items():
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].map(mapping)
                else:
                    raise ValueError(f"Column '{col}' not found in DataFrame.")

        print(f"Fitting logistic regression model with formula: {formula}")
        print(df_copy)
        model = smf.mnlogit(formula, data=df_copy).fit()
        print(model.summary())
        return model
    

# vis = MMELR()
# vis.load_dataframe(general=True)
# vis.generate_mosaic(["definiteness", "Ntype", "ref"], "mosaic_1", category_orders={"definiteness": ["indef", "def"], "Ntype": ["sing", "mass", "plural"]})

# model = vis.train_logistic_regression(
#     formula="definiteness ~ ref * Ntype",
#     to_map={
#         "definiteness": {"def": 1, "indef": 0},
#         "ref": {"spec": 1, "nonspec": 0},
#         "Ntype": {"sing": 0, "mass": 1, "plural": 2},
#         "modif": {"mod": 1, "no_mod": 0}
#     }
# )

# vis = MMELR()
# vis.load_dataframe(general=True)
# vis.generate_mosaic(["definiteness", "Ntype", "modif"], "mosaic_2", category_orders={"definiteness": ["indef", "def"],
#                                                                                  "Ntype": ["sing", "mass", "plural"],
#                                                                                  "modif": ["no_mod", "mod"]})

# model = vis.train_logistic_regression(
#     formula="definiteness ~ modif * Ntype",
#     to_map={
#         "definiteness": {"def": 1, "indef": 0},
#         "ref": {"spec": 1, "nonspec": 0},
#         "Ntype": {"sing": 0, "mass": 1, "plural": 2},
#         "modif": {"mod": 1, "no_mod": 0}
#     }
# )

# vis = MMELR()
# vis.load_dataframe(general=True)
# vis.generate_mosaic(["definiteness", "Ntype", "modif"], "mosaic_2", category_orders={"definiteness": ["indef", "def"],
#                                                                                  "Ntype": ["sing", "mass", "plural"],
#                                                                                  "modif": ["no_mod", "mod"]})

# model = vis.train_logistic_regression(
#     formula="Ntype ~ definiteness * modif",
#     to_map={
#         "definiteness": {"def": 1, "indef": 0},
#         "ref": {"spec": 1, "nonspec": 0},
#         "Ntype": {"sing": 0, "mass": 1, "plural": 0},
#         "modif": {"mod": 1, "no_mod": 0}
#     }
# )

# vis = MMELR()
# vis.load_dataframe(general=True)
# vis.generate_mosaic(["definiteness", "Ntype", "Abstract"], "mosaic_3", category_orders={"definiteness": ["indef", "def"],
#                                                                                  "Ntype": ["sing", "mass", "plural"],
#                                                                                  "Abstract": ["abstr", "concr"]})

# model = vis.train_logistic_regression(
#     formula="Abstract ~ definiteness * Ntype",
#     to_map={
#         "definiteness": {"def": 0, "indef": 1},
#         "ref": {"spec": 1, "nonspec": 0},
#         "Ntype": {"sing": 0, "mass": 1, "plural": 0},
#         "Abstract": {"abstr": 1, "concr": 0}
#     }
# )

# vis = MMELR()
# vis.load_dataframe(general=True)
# vis.generate_mosaic(["definiteness", "Ntype", "Synt"], "mosaic_4", category_orders={"definiteness": ["indef", "def"],
#                                                                                  "Ntype": ["sing", "mass", "plural"],
#                                                                                  "Synt": ["ex", "pred/prop", "obj", "sub"]})

# model = vis.train_logistic_regression(
#     formula="Synt ~ definiteness * Ntype",
#     to_map={
#         "definiteness": {"def": 1, "indef": 0},
#         "ref": {"spec": 1, "nonspec": 0},
#         "Ntype": {"sing": 0, "mass": 1, "plural": 0},
#         "Synt": {"sub": 1, "obj": 0, "pred/prop": 0, "ex": 0}
#     }
# )
