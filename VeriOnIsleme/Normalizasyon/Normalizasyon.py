import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Veri setini yükleme
dataset_path = r'VeriOnIsleme\EksikVeriDoldurma\cleaned_dataset.csv'
df = pd.read_csv(dataset_path)

# Sayısal sütunları seçme
numeric_columns = [
     "Academic Pressure", "Work Pressure", "CGPA",
    "Study Satisfaction", "Job Satisfaction",
    "Work/Study Hours", "Financial Stress"
]

# Normalizasyon öncesi örnek veriler
print("Normalizasyon Öncesi Veri:")
print(df[numeric_columns].head())

# Normalizasyon işlemi
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Normalizasyon sonrası örnek veriler
print("\nNormalizasyon Sonrası Veri:")
print(df[numeric_columns].head())

# Normalizasyon yapılmış veri setini kaydetme
output_path = r"C:\Users\ABC\Desktop\MachineLearning\VeriOnIsleme\Normalizasyon\Normalizasyon.csv"
df.to_csv(output_path, index=False)

print(f"Normalizasyon tamamlandı! Normalleştirilmiş veri seti kaydedildi: {output_path}")
# Eksik değerleri kontrol edin
print("\nEksik değerlerin sayısı (doldurduktan sonra):")
print(df.isnull().sum())
