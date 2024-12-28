import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Veriyi yükle
data = pd.read_csv(r'C:\Users\ABC\Desktop\MachineLearning\VeriOnIsleme\OzellikCikarma\features_with_extraction.csv')

# Depresyon sütununu int türüne dönüştür
data['Depression'] = data['Depression'].astype(int)

# Kategorik sütunları belirle
categorical_columns = data.select_dtypes(include=['object']).columns

# Kategorik sütunları sayısal hale getir
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Tamamen eksik sütunları kaldır
data = data.dropna(axis=1, how='all')

# Eksik değerleri doldurma (ortalama ile)
imputer = SimpleImputer(strategy='mean')
X = data.drop(columns=['Depression'])  # Özellikler
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)  # Eksik değerleri doldur

y = data['Depression']  # Hedef değişken

# Sınıf dağılımını hesapla
class_distribution = y.value_counts()

# Age sütununun maksimum değerine sahip olan satırı bul ve yazdır
max_age_row = data.loc[data['Age'].idxmax()]
print(max_age_row)

# Sınıf dağılımını çiz
plt.figure(figsize=(8, 5))
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.title('Sınıf Dağılımı')
plt.xlabel('Depresyon Durumu (0: Yok, 1: Var)')
plt.ylabel('Gözlem Sayısı')
plt.xticks(ticks=[0, 1], labels=['Yok', 'Var'])
plt.show()

# Grafiksel analizler
# 1. Histogram (Örnek: Yaş Dağılımı)
plt.figure(figsize=(8, 5))
sns.histplot(data=X, x='Age', hue=y, kde=True, bins=30)
plt.title('Yaş Dağılımı')
plt.xlabel('Yaş')
plt.ylabel('Frekans')
plt.legend(title='Depression', labels=['Yok', 'Var'])
plt.show()

# Gender sütunundaki 0 ve 1 değerlerini Kadın ve Erkek olarak etiketleme
data['Gender'] = data['Gender'].map({0: 'Kadın', 1: 'Erkek'})

# Cinsiyet ve depresyon durumu ilişkisini sayısal olarak hesapla
gender_depression_distribution = data.groupby(['Gender', 'Depression']).size().unstack(fill_value=0)

# Erkekler ve kadınlar için depresyon durumunun bar grafiği
gender_depression_distribution.plot(kind='bar', figsize=(8, 5), stacked=True, color=['skyblue', 'lightcoral'])

# Grafiği özelleştir
plt.title('Erkek ve Kadınların Depresyon Durumu')
plt.xlabel('Cinsiyet')
plt.ylabel('Sayı')
plt.xticks(rotation=0)
plt.legend(title='Depresyon Durumu', labels=['Yok', 'Var'])
plt.show()
