
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Veri setini yükleyin
# Dosyanın adını dataset.csv ile değiştirin
data = pd.read_csv(r'datasets/mobile-device-usage-and-user-behavior-dataset/Student Depression Dataset.csv')

# Eksik verilerin genel durumunu kontrol edin
print("Eksik değerlerin sayısı:")
print(data.isnull().sum())

# Kategorik ve sayısal sütunları ayırma
categorical_columns = data.select_dtypes(include=['object']).columns
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

# Sayısal sütunlar için: Ortalama ile doldurma
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = numerical_imputer.fit_transform(data[numerical_columns])
# 'Age' sütununu int değerlere dönüştürme
data['Age'] = data['Age'].astype(int)
# Kategorik sütunlar için: Mod (en sık görülen değer) ile doldurma
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

# Alternatif: İteratif yöntemle eksik değer doldurma (gelişmiş)
iterative_imputer = IterativeImputer(random_state=42)
data[numerical_columns] = iterative_imputer.fit_transform(data[numerical_columns])

# Eksik değerleri kontrol edin
print("\nEksik değerlerin sayısı (doldurduktan sonra):")
print(data.isnull().sum())

# Temizlenmiş veriyi kaydetme
data.to_csv(r'VeriOnIsleme\EksikVeriDoldurma\cleaned_dataset.csv', index=False)
print("\nTemizlenmiş veri datasets/cleaned_dataset.csv olarak kaydedildi.")

