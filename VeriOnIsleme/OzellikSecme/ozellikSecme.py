
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Veri setini yükleme
file_path = r'VeriOnIsleme\Normalizasyon\Normalizasyon.csv'
data = pd.read_csv(file_path)

# Sütunun sayısallaştırılması
if 'Have you ever had suicidal thoughts ?' in data.columns:
    label_encoder = LabelEncoder()
    # Sayısal sütunu oluşturma ama veri setine eklememe
    data['Suicidal_Thoughts_Numeric'] = label_encoder.fit_transform(data['Have you ever had suicidal thoughts ?'])
else:
    print("Sütun veri setinde mevcut değil.")
    # Sütunun sayısallaştırılması
if 'Family History of Mental Illness' in data.columns:
    label_encoder = LabelEncoder()
    # Sayısal sütunu oluşturma ama veri setine eklememe
    data['Family History of Mental Illness_Numeric'] = label_encoder.fit_transform(data['Family History of Mental Illness'])
else:
    print("Sütun veri setinde mevcut değil.")

# Sadece anlamlı sayısal sütunları seçmek (id gibi gereksiz sütunları çıkarmak)
numeric_data = data.select_dtypes(include=['number']).drop(columns=['id'], errors='ignore')

# İlk korelasyon matrisini çizme
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("İlk Korelasyon Matrisi")
plt.show()

# Sütunları tamamen silme
columns_to_drop = ['Work Pressure', 'Job Satisfaction']
data_dropped = data.drop(columns=columns_to_drop)



# Yeni veri setini kaydetme
output_file_path = r'VeriOnIsleme\OzellikSecme\updated_dataset.csv'
data_dropped.to_csv(output_file_path, index=False)
print(f"Güncellenmiş veri seti '{output_file_path}' olarak kaydedildi.")

# Güncellenmiş veri setinden sadece sayısal ve anlamlı sütunları seçmek
numeric_data_dropped = data_dropped.select_dtypes(include=['number']).drop(columns=['id'], errors='ignore')

# Güncellenmiş korelasyon matrisini çizme
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data_dropped.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Güncellenmiş Korelasyon Matrisi")
plt.show()
# 'Suicidal_Thoughts_Numeric' sütununu veri setinden çıkarma
data_dropped = data_dropped.drop(columns=['Suicidal_Thoughts_Numeric'], errors='ignore')
# 'Family History of Mental Illness_Numeric' sütununu veri setinden çıkarma
data_dropped = data_dropped.drop(columns=['Family History of Mental Illness_Numeric'], errors='ignore')
data_dropped.to_csv(output_file_path, index=False)


#RANDOM FOREST
# Veri dosyasını okuma
file_path1 = r'VeriOnIsleme\OzellikSecme\updated_dataset.csv'
data2 = pd.read_csv(file_path1)

# Hedef sütunu belirleme (örnek olarak 'Depression' seçildi)
target_column = 'Depression'

# Kategorik sütunları tespit etme
categorical_columns = data2.select_dtypes(include=['object']).columns

# Kategorik sütunları sayısal değerlere dönüştürme
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data2[col] = le.fit_transform(data2[col])
    label_encoders[col] = le  # İleride gerekirse kod çözmek için

# Hedef ve bağımsız değişkenlerin belirlenmesi
X = data2.drop(columns=[target_column, 'id'], errors='ignore')
y = data2[target_column]

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model seçimi (sınıflandırma veya regresyon)
if y.nunique() > 10:  # Sürekli değer durumu
    model = RandomForestRegressor(random_state=42)
else:  # Kategorik hedef
    model = RandomForestClassifier(random_state=42)

# Modeli eğitme
model.fit(X_train, y_train)

# Özellik önem derecelerini alma
feature_importances = model.feature_importances_
features = X.columns

# Sonuçları görselleştirme (ilk 5 özellik)
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(5)  # İlk 5 özelliği alma

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color="skyblue")
plt.xlabel("Özellik Önemi")
plt.ylabel("Özellikler")
plt.title("En Önemli 5 Özelliğin Random Forest ile Önemi")
plt.gca().invert_yaxis()  # En önemli özellik üstte olacak şekilde sıralama
plt.show()