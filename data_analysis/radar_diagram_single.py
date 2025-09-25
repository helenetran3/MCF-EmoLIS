import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Lecture du fichier
csv_path = "classif_pres_radar.txt"
df = pd.read_csv(csv_path, sep=",")

# 2. Renommage des colonnes
def rename_columns(df):
    rename_map = {
        "wei_acc_macro": "WA (U)",
        "wei_acc_weighted": "WA (W)",
        "f1_macro": "F1 (U)",
        "f1_weighted": "F1 (W)",
        "rec_macro": "Recall (U)",
        "rec_weighted": "Recall (W)",
        "prec_macro": "Precision (U)",
        "prec_weighted": "Precision (W)",
        "jacc_macro": "Jaccard (U)",
        "jacc_weighted": "Jaccard (W)",
    }
    return df.rename(columns=rename_map)

df = rename_columns(df)
df.set_index("Task", inplace=True)

# 3. Choix des métriques
metrics = [
    "WA (U)", "WA (W)",
    "F1 (U)", "F1 (W)",
    "Recall (U)", "Recall (W)",
    "Precision (U)", "Precision (W)",
    "Jaccard (U)", "Jaccard (W)"
]

df = df[metrics]

# 4. Création du radar
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for model_name, row in df.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, label=model_name)
    ax.fill(angles, values, alpha=0.1)

# Mise en forme
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=10)

y_pos = [0, 0.03, 0.05, 0.05, 0.06, 0, 0.06, 0.13, 0.1, 0.06]
for lab_id, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):
    label.set_horizontalalignment('center')
    label.set_fontsize(12)
    x, y = label.get_position()
    label.set_position((x, y-y_pos[lab_id]))  # Décale verticalement vers l’extérieur

ax.set_ylim(0, 100)
ax.set_yticks(range(10, 101, 10))
ax.set_yticklabels([str(i) for i in range(10, 101, 10)], color="grey", size=7)

plt.legend(
    loc="upper center",           # place la légende en haut de la zone de légende
    bbox_to_anchor=(0.5, -0.05),  # décale sous la figure
    ncol=len(df.index),           # met tous les éléments sur une seule ligne
    fontsize=15
)

plt.savefig("radar_best.png", dpi=300, bbox_inches="tight")
plt.show()
