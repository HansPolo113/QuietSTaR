import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    setting = ['standard', 'padding', 'reverse', 'paddingreverse'][3]
    if setting == 'standard':
        csv_path = '/home/emzhang/code/nanoGPT/output/multiplication_8digit/accuracy_matrix.csv'
    else:
        csv_path = f'/home/emzhang/code/nanoGPT/output/multiplication_8digit_{setting}/accuracy_matrix.csv'
    df = pd.read_csv(csv_path)
    sns.set_theme()

    plt.figure(figsize=(8,8))
    heatmap = sns.heatmap(df, annot=True, fmt='.4f', cmap='viridis', cbar=True, square=True)

    plt.title(f'{setting}')
    plt.show()