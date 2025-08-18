import pandas as pd
import numpy as np

input_file_p = "test_csv_latex.csv"
find_max_p = True

def process_csv_blocks(input_file, find_max, output_file="output.txt"):
    # Lire brut les lignes
    with open(input_file, encoding="utf-8") as f:
        raw_lines = f.readlines()

    # Remplacer virgules par points
    raw_lines = [line.replace(",", ".").strip() for line in raw_lines]

    # Identifier les blocs (séparés par lignes vides)
    blocks = []
    current_block = []

    for line in raw_lines:
        if line == "":
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line)
    if current_block:
        blocks.append(current_block)  # Ajouter le dernier bloc

    # En-têtes (supposés identiques pour chaque bloc)
    headers = blocks[0][0].split(";")
    processed_rows = []
    col_indices = list(range(1, len(headers), 2))

    # Stocker toutes les valeurs \textbf{v} pour chaque colonne A
    bold_values_per_col = {idx: [] for idx in col_indices}

    # Étape 1 : appliquer \textbf{v} dans chaque bloc
    processed_blocks = []
    for block in blocks:
        df = pd.DataFrame([row.split(";") for row in block])
        df = df.applymap(lambda x: float(x) if x and not x.upper().isupper() else x)

        df_copy = df.copy()
        print(df_copy)
        print("\n")

        for col_a in col_indices:
            col_data = df.iloc[:, col_a]
            if find_max:
                target_val = col_data.max()
            else:
                target_val = col_data.min()

            # Marquer la première occurrence uniquement
            for i in range(len(col_data)):
                if col_data.iat[i] == target_val:
                    df_copy.iat[i, col_a] = f"\\textbf{{{target_val}}}"
                    bold_values_per_col[col_a].append(target_val)
                    break

        processed_blocks.append((df_copy, df.columns.tolist()))

    # Étape 2 : marquer \textcolor{magenta}{\textbf{v}} si v est global max/min
    for col_a in col_indices:
        vals = bold_values_per_col[col_a]
        if not vals:
            continue
        global_extreme = max(vals) if find_max else min(vals)
        # Mise à jour dans tous les blocs
        for block_idx, (df_block, columns) in enumerate(processed_blocks):
            for i in range(len(df_block)):
                val = df_block.iat[i, col_a]
                if isinstance(val, str) and val.startswith("\\textbf"):
                    num = float(val.strip("\\textbf{}"))
                    if num == global_extreme:
                        df_block.iat[i, col_a] = f"\\textcolor{{magenta}}{{{val}}}"

    for df_block, _ in processed_blocks:
        print("\nBlock:")
        print(df_block)

    # Étape 3 : fusion A ± B et format LaTeX
    final_lines = []
    for idx, (df_block, columns) in enumerate(processed_blocks):
        for i in range(len(df_block)):
            row = [str(df_block.iat[i, 0])]  # première colonne
            for col_a in col_indices:
                col_b = col_a + 1
                val_a = str(df_block.iat[i, col_a])
                val_b = str(df_block.iat[i, col_b])
                combined = f"{val_a} $\\pm$ {val_b}"
                row.append(combined)
            final_lines.append(" & ".join(row) + r" \\ \hline")

        final_lines.append("")

            # Écriture dans le fichier texte
    with open(output_file, "w", encoding="utf-8") as f:
        for line in final_lines:
            f.write(line + "\n")

    print(f"✅ Fichier généré : {output_file}")


process_csv_blocks(input_file_p, find_max_p)