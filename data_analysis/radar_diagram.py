import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# 1. Lecture des blocs séparés par lignes vides
def split_csv_by_empty_lines(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    blocks = []
    current_block = []
    for line in lines:
        if line.strip() == "":
            if current_block:
                df = pd.read_csv(io.StringIO("".join(current_block)), sep=",")
                blocks.append(df)
                current_block = []
        else:
            current_block.append(line)
    if current_block:
        df = pd.read_csv(io.StringIO("".join(current_block)), sep=",")
        blocks.append(df)

    return blocks

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

# 3. Fonction principale de génération des radars
def plot_radar_charts(blocks, titles, metrics, output_file):
    assert len(blocks) == len(titles) == 6, "Il doit y avoir exactement 6 blocs et 6 titres."

    fig, axs = plt.subplots(2, 3, figsize=(20, 12), subplot_kw=dict(polar=True))
    axs = axs.flatten()

    # Réduction des tailles / espacement
    fig.subplots_adjust(wspace=0.2, hspace=0.5, top=0.9, bottom=0.1)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Fermer le radar

    all_handles = []
    all_labels = []

    for i, (df, ax) in enumerate(zip(blocks, axs)):
        df = rename_columns(df)
        df.set_index("task", inplace=True)
        df = df[metrics]

        for model_name, row in df.iterrows():
            values = row.tolist()
            values += values[:1]
            line, = ax.plot(angles, values, label=model_name)
            ax.fill(angles, values, alpha=0.1)

        # Mise en forme du radar
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=8)

        y_pos = [0.03, 0.03, 0.03, 0.03, 0.03, 0, 0.03, 0.12, 0.1, 0.03]
        for lab_id, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):
            label.set_horizontalalignment('center')
            label.set_fontsize(8)
            x, y = label.get_position()
            label.set_position((x, y-y_pos[lab_id]))  # Décale verticalement vers l’extérieur

        ax.set_rlabel_position(0)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(10, 101, 10))
        ax.set_yticklabels([str(i) for i in range(10, 101, 10)], color="grey", size=7)

        ax.set_title(titles[i], size=14, pad=20)

        # Récupérer handles/labels pour la légende globale
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

    # Légende globale (unique en bas)
    by_label = dict(zip(all_labels, all_handles))  # Éviter doublons
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=6, fontsize=11)

    plt.savefig(output_file, dpi=300)
    plt.show()


# Exemple d’utilisation
# Remplacer par le chemin réel vers ton CSV
csv_path = "csv_diagram_radar.csv"

titles = [
    "RNN-based early fusion",
    "Transformer-based early fusion",
    "RNN-based late fusion",
    "Transformer-based late fusion",
    "Memory Fusion Network",
    "Multimodal Transformer"
]

metrics = [
    "WA (U)", "WA (W)",
    "F1 (U)", "F1 (W)",
    "Recall (U)", "Recall (W)",
    "Precision (U)", "Precision (W)",
    "Jaccard (U)", "Jaccard (W)"
]

# metrics = [
#     "Weighted Accuracy (U)", "Weighted Accuracy (W)",
#     "F1 Score (U)", "F1 Score (W)",
#     "Recall (U)", "Recall (W)",
#     "Precision (U)", "Precision (W)",
#     "Jaccard Score (U)", "Jaccard Score (W)"
# ]

blocks = split_csv_by_empty_lines(csv_path)
plot_radar_charts(blocks, titles, metrics, "../diagram_radars.png")
