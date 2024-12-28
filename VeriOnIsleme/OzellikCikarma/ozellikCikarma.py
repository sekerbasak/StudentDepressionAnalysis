from matplotlib import pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Veri setini yükleme
df = pd.read_csv(r'VeriOnIsleme\OzellikSecme\updated_dataset.csv')

def create_depression_features(df):
    """
    Depression tahmini için feature engineering yapan ve sonuçları kaydeden fonksiyon
    """
    # Veri setinin bir kopyasını oluştur
    df_new = df.copy()
    
    # Suicidal thoughts'u numerik hale getir (Yes=1, No=0)
    df_new['suicidal_thoughts_numeric'] = (df_new['Have you ever had suicidal thoughts ?'] == 'Yes').astype(int)
    
    # 1. Academic Pressure ve suicidal thoughts'un etkileşimi
    df_new['pressure_thoughts_interaction'] = df_new['Academic Pressure'] * df_new['suicidal_thoughts_numeric']
   
    return df_new

# Feature engineering işlemini uygula
df_engineered = create_depression_features(df)

# Sütunları tamamen silme
columns_to_drop = ['suicidal_thoughts_numeric']
df_engineered = df_engineered.drop(columns=columns_to_drop)


# Çıktı dosyasını kaydetme
output_file_path = r'C:\Users\ABC\Desktop\MachineLearning\VeriOnIsleme\OzellikCikarma\features_with_extraction.csv'

# Dosyanın kaydedileceği klasörü kontrol et
output_dir = os.path.dirname(output_file_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Klasör mevcut değilse oluştur

# Yeni özellikleri içeren DataFrame'i CSV dosyasına kaydet
df_engineered.to_csv(output_file_path, index=False)
print(f"Yeni özelliklerle birlikte sonuçlar '{output_file_path}' dosyasına kaydedildi.")


# Yeni eklenen özellikleri göster
print("\nEklenen yeni özellikler:")
new_features = ['pressure_thoughts_interaction']
print(df_engineered[new_features].head())

# Tüm veri setinin ilk birkaç satırını göster
print("\nTüm güncellenmiş veri setinin ilk birkaç satırı:")
print(df_engineered.head())

# Age sütununun maksimum değerine sahip olan satırı bul ve yazdır
max_age_row = df_engineered.loc[df_engineered['Age'].idxmax()]
print(max_age_row)