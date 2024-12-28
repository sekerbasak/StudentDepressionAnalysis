import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Veriyi yükle
data = pd.read_csv(r'c:/Users/ABC/Desktop/MachineLearning/VeriOnIsleme/OzellikCikarma/features_with_extraction.csv')

# Sayısal sütunları seçme
numerical_columns = [
    'Age',
    'Academic Pressure',
    'CGPA',
    'Study Satisfaction',
    'Work/Study Hours',
    'Financial Stress',
    'pressure_thoughts_interaction',
    'Depression'
]

# Histogramlar
plt.figure(figsize=(15, 10))
for i, column in enumerate(numerical_columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(data[column], bins=10, kde=True)
    plt.title(f'{column} Histogram')
plt.tight_layout()
plt.show()

# Scatter Plotlar
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Age', y='pressure_thoughts_interaction', hue='Depression')
plt.title('Age vs pressure_thoughts_interaction Scatter Plot')
plt.xlabel('Age')
plt.ylabel('pressure_thoughts_interaction')
plt.legend(title='Depression')
plt.show()

# Korelasyon Isı Haritası
correlation_matrix = data[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Korelasyon Isı Haritası')
plt.show()