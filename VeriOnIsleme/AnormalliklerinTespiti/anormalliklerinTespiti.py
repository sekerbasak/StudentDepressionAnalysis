import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Veriyi yükle
df = pd.read_csv(r'C:\Users\ABC\Desktop\MachineLearning\datasets\mobile-device-usage-and-user-behavior-dataset\Student Depression Dataset.csv')

# Z-Score hesaplaması
def calculate_z_scores(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    z_scores = pd.DataFrame(index=df.index)
    
    for column in numeric_columns:
        # İlgili sütundaki değerlerin z-score'larını hesapla
        z_scores[column] = np.abs(stats.zscore(df[column], nan_policy='omit'))
    
    return z_scores

# Anormallikleri tespit et
def detect_anomalies(df, z_scores, threshold=3):
    anomalies = (z_scores > threshold)
    return anomalies

# Boxplot görselleştirme
def plot_boxplots(df, anomalies_df, numeric_columns):
    columns_without_id = [col for col in numeric_columns if col.lower() != 'id']
    n_cols = 3
    n_rows = (len(columns_without_id) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Boxplots of Numeric Features with Anomalies', fontsize=16, y=1.02)

    for idx, (column, ax) in enumerate(zip(columns_without_id, axes.flat)):
        sns.boxplot(data=df, y=column, ax=ax, color='skyblue')

        # Anomaliler için overlay
        anomaly_indices = anomalies_df[column][anomalies_df[column]].index
        anomaly_values = df.loc[anomaly_indices, column]
        if not anomaly_values.empty:
            sns.stripplot(data=df.loc[anomaly_indices], y=column, ax=ax, color='red', label='Anomalies', jitter=True)

        ax.set_title(f'Boxplot of {column}')
        ax.legend()

    # Boş plotları gizle
    for idx in range(len(columns_without_id), len(axes.flat)):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(r'C:\Users\ABC\Desktop\MachineLearning\VeriOnIsleme\AnormalliklerinTespiti\boxplots.png')
    plt.close()

# Violinplot görselleştirme (pairplot yerine)
def plot_violinplots(df, anomalies_df, numeric_columns):
    df['Anomaly'] = anomalies_df.any(axis=1)
    n_cols = 3
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Violin Plots of Numeric Features with Anomalies', fontsize=16, y=1.02)

    for idx, (column, ax) in enumerate(zip(numeric_columns, axes.flat)):
        sns.violinplot(data=df, y=column, ax=ax, inner="point", scale="count", color='skyblue')

        # Anomaliler için overlay
        anomaly_indices = anomalies_df[column][anomalies_df[column]].index
        anomaly_values = df.loc[anomaly_indices, column]
        if not anomaly_values.empty:
            sns.scatterplot(data=df.loc[anomaly_indices], x=[0]*len(anomaly_values), y=anomaly_values, color='red', label='Anomalies', ax=ax, s=100)

        ax.set_title(f'Violin Plot of {column}')
        ax.legend()

    # Boş plotları gizle
    for idx in range(len(numeric_columns), len(axes.flat)):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(r'C:\Users\ABC\Desktop\MachineLearning\VeriOnIsleme\AnormalliklerinTespiti\violinplots.png')
    plt.close()

# Ana işlem akışı
numeric_columns = df.select_dtypes(include=[np.number]).columns
z_scores = calculate_z_scores(df)
anomalies = detect_anomalies(df, z_scores)

# Sonuçları yazdır
print(f"Toplam anormal veri sayısı: {anomalies.sum().sum()}")

# Anormal verilerin detaylarını yazdır
for column in numeric_columns:
    anomaly_indices = anomalies[column][anomalies[column]].index
    if len(anomaly_indices) > 0:
        print(f"\nAnormallikler - {column}:")
        for idx in anomaly_indices:
            print(f"Satır: {idx}, Değer: {df.loc[idx, column]}, Z-Score: {z_scores.loc[idx, column]:.2f}")

# Her sütunda anormal veri sayısını yazdır
print(f"\nHer sütunda anormal veri sayısı:\n{anomalies.sum()}")

# Görselleştirmeleri yap ve dosyaya kaydet
plot_boxplots(df, anomalies, numeric_columns)
plot_violinplots(df, anomalies, numeric_columns)
