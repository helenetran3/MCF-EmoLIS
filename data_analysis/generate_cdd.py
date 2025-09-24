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


def best_model_per_variant(df, score_col):
    """
    Pour chaque variante (model_id), on prend le modèle ayant la meilleure moyenne
    sur tous les folds selon score_col.
    """
    print("\nSelecting best model per variant for", score_col)

    # Moyenne par (model_id, model)
    means = df.groupby(["model_id", "model"])[score_col].mean()

    # Pour chaque variante, on garde le modèle au meilleur score moyen
    best_pairs = means.groupby("model_id").idxmax()
    print("means.groupby('model_id'):", means.groupby("model_id").head())
    print("best_pairs:", best_pairs)
    best_set = set(best_pairs.values)
    print("Best model per variant:", best_pairs.to_dict())

    # Filtrer uniquement les (model_id, model) choisis
    mask = df.apply(lambda r: (r["model_id"], r["model"]) in best_set, axis=1)
    df_best = df[mask][["fold_id", "model_id", score_col]]

    # Agréger au cas où il resterait plusieurs lignes par (fold_id, model_id)
    df_best = df_best.groupby(["fold_id", "model_id"], as_index=False).mean()

    df_best.to_csv(f'best_{score_col}.csv', index=False)
    return df_best


# Matrices F1_1 et F1_2 avec les meilleures variantes
df_best_f1_1 = best_model_per_variant(df, df.columns[3])  # F1_1
df_best_f1_2 = best_model_per_variant(df, df.columns[4])  # F1_2

# === 3. Construire matrices (folds × modèles) ===
def build_matrix(best_df, score_col):
    pivot = best_df.pivot(index="fold_id", columns="model_id", values=score_col)
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
        test='nemenyi',
        alpha=0.05)
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
process_and_plot(matrix_f1_1, "F1_macro_{}_best_model".format(col_to_group))
process_and_plot(matrix_f1_2, "F1_weighted_{}_best_model".format(col_to_group))
