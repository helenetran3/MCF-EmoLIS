import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from aeon.visualisation import plot_critical_difference

########################################################################################################################
# The code of this file have been copied from the repository of the original author:
# https://gist.github.com/milenamonteiro/33ea5223c7cd43cabd71fc96375908c1
#
# The file has been slightly modified to input a csv file named f1_per_fold.csv (an example is attached to this repository).
########################################################################################################################


# === 1. Charger le CSV ===
df = pd.read_csv("f1_per_fold.csv", sep=";", decimal=",")

# Vérification
print("First lines")
print(df.head())

compare_model_variant = False
col_to_group = "model_id" if compare_model_variant else "model"
col_best = "model" if compare_model_variant else "model_id"

# === 2. Sélectionner la meilleure variante par modèle ===
def best_variant(df, score_col):
    """
    Pour chaque fold et chaque modèle,
    on garde la variante avec le meilleur score_col.
    """
    print("\nCalculating best variant for", score_col)
    idx = df.groupby(["fold_id", col_to_group])[score_col].idxmax()
    df_best = df.loc[idx, ["fold_id", col_to_group, score_col, col_best]]
    print(df_best)
    df_best.to_csv('best_{}.csv'.format(score_col), index=False)
    return df.loc[idx, ["fold_id", col_to_group, score_col]]

# Matrices F1_1 et F1_2 avec les meilleures variantes
df_best_f1_1 = best_variant(df, df.columns[3])  # F1_1
df_best_f1_2 = best_variant(df, df.columns[4])  # F1_2

# === 3. Construire matrices (folds × modèles) ===
def build_matrix(best_df, score_col):
    pivot = best_df.pivot(index="fold_id", columns=col_to_group, values=score_col)
    return pivot

matrix_f1_1 = build_matrix(df_best_f1_1, df.columns[3])
matrix_f1_2 = build_matrix(df_best_f1_2, df.columns[4])
print("\nFinal matrix_F1_macro:")
print(matrix_f1_1)
print("\nFinal matrix_F1_weighted:")
print(matrix_f1_2)

# === 4. Fonction pour générer CDD ===
def process_and_plot(matrix, label):
    rankings_matrix = matrix.rank(axis=1, method="min", ascending=False)

    # Friedman test
    friedman_stat, p_value = friedmanchisquare(*rankings_matrix.T.values)
    print(f"[{label}] Friedman test statistic: {friedman_stat}, p-value = {p_value}")

    # CDD
    scores = matrix.values
    classifiers = matrix.columns.tolist()

    plt.figure(figsize=(16, 12))
    plot_critical_difference(
        scores,
        classifiers,
        lower_better=False,
        test='wilcoxon',
        correction='holm',
    )
    ax = plt.gca()
    for l in ax.get_xticklabels():
        l.set_fontsize(14)
        l.set_rotation(45)
        l.set_horizontalalignment('right')
    ax.tick_params(axis='x', which='major', pad=20)
    ax.tick_params(axis='y', labelsize=12)
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(f'critical_difference_diagram_{label}.png', format="png", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Critical difference diagram saved as critical_difference_diagram_{label}.png")

# === 5. Générer les deux diagrammes ===
process_and_plot(matrix_f1_1, "F1_macro_{}".format(col_to_group))
process_and_plot(matrix_f1_2, "F1_weighted_{}".format(col_to_group))
