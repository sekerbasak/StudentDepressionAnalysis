import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veriyi yükle
df = pd.read_csv('datasets/mobile-device-usage-and-user-behavior-dataset/Student Depression Dataset.csv')

# Veri setinin boyutunu ve özelliklerini yazdır
print(f"\n\nVeri setinin boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
print("\nVeri setindeki sütunlar ve türleri:")
print(df.info())

# Eksik veri bulunan sütunları bul
missing_counts = df.isnull().sum()
missing_columns = missing_counts[missing_counts > 0].index

print("\nEksik Veri Bulunan Sütunlar:")
print(missing_columns)

# Eksik veri bulunan her sütunda hangi satırlarda eksik veri olduğunu yazdır
for column in missing_columns:
    print(f"\n{column} sütununda eksik veri bulunan satırlar:")
    print(df[df[column].isnull()])  # Belirli bir sütundaki eksik veriye sahip satırları yazdır

# Hedef değişkeni belirleme (örnek olarak 'Depression' kolonu)
target_variable = 'Depression'  # Burada hedef değişkenin adı örnek olarak verilmiştir.
print(f"\nHedef değişken: {target_variable}")

# Eksik verilerin görselleştirilmesi
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Eksik Verilerin Görselleştirilmesi", fontsize=15)
plt.show()

# Her sütundaki eksik veri sayısını yazdır
print("\nHer sütundaki eksik veri sayısı:")
print(missing_counts)

# Toplam eksik veri sayısını yazdır
total_missing = missing_counts.sum()
print("Toplam eksik veri:", total_missing)

# Değişikliklerden sonra dosyayı üzerine yaz (eğer gerekli ise)
df.to_csv('datasets/mobile-device-usage-and-user-behavior-dataset/Student Depression Dataset.csv', index=False)