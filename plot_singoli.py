import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('risultati_medi_aggregati.csv', sep=';')
df_all = df[df['selection_mode'] == 'all'].copy()

output_dir = 'risultati'
os.makedirs(output_dir, exist_ok=True)

corruption_types = df_all['corruption_type'].unique()

for ct in corruption_types:
    subset = df_all[df_all['corruption_type'] == ct]
    models = subset['model'].unique()

    plt.figure(figsize=(10, 6))
    for model in models:
        model_data = subset[subset['model'] == model].sort_values('corruption_level')
        plt.plot(model_data['corruption_level'], model_data['f1_score'], marker='o', label=model)

    plt.xlabel('Corruption Level')
    plt.ylabel('F1 Score')
    plt.title(f'Tipologia corruzione: {ct}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, f'{ct}.png'), dpi=150)
    plt.close()

print(f"Plots saved in {output_dir}/")
