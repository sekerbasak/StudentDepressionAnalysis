import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, \
    auc
import matplotlib.pyplot as plt


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    categorical_cols = ['Gender', 'City', 'Profession', 'Sleep Duration',
                        'Dietary Habits', 'Degree',
                        'Have you ever had suicidal thoughts ?',
                        'Family History of Mental Illness']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    return data


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Eğitim doğruluğu
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Test doğruluğu
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Performans raporu
    report = classification_report(y_test, y_test_pred)

    # Metrikler
    metrics = {
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "F1-Score": f1_score(y_test, y_test_pred)
    }

    return report, metrics


def plot_roc_curve(models, X_test, y_test, filename="roc_curve.png"):
    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):  # predict_proba varsa
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):  # SVM gibi durumlar için
            y_pred_proba = model.decision_function(X_test)
        else:
            continue  # ROC çizilemiyorsa geç

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    # Diagonal çizgi: Random tahmin
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - All Models')
    plt.legend(loc="lower right")

    # Görseli dosyaya kaydet
    plt.savefig(filename)
    plt.close()  # Görseli kapat, böylece belleği temizleriz


def plot_accuracy_histogram(results, filename="accuracy_histogram.png"):
    models = list(results.keys())
    train_accuracies = [results[model]["Metrics"]["Train Accuracy"] for model in models]
    test_accuracies = [results[model]["Metrics"]["Test Accuracy"] for model in models]

    bar_width = 0.35
    index = range(len(models))

    plt.figure(figsize=(10, 6))

    # Eğitim doğruluğu (turuncu)
    plt.bar(index, train_accuracies, bar_width, color='orange', label='Train Accuracy')

    # Test doğruluğu (mavi)
    plt.bar([i + bar_width for i in index], test_accuracies, bar_width, color='blue', label='Test Accuracy')

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy for Different Models')
    plt.xticks([i + bar_width / 2 for i in index], models)
    plt.legend()

    # Görseli dosyaya kaydet
    plt.savefig(filename)
    plt.close()  # Görseli kapat, böylece belleği temizleriz


def main():
    filepath = r"C:\Users\ABC\Desktop\MachineLearning\VeriOnIsleme\OzellikCikarma\features_with_extraction.csv"

    data = load_and_preprocess_data(filepath)

    X = data.drop(columns=['id', 'Depression'])
    y = data['Depression']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Random Forest1": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        "Random Forest2": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [100, 200],  # Ağaç sayısını çok yüksek tutmayın
                'max_depth': [3, 5, 10],  # Derinliği sınırlayın
                'min_samples_split': [10, 20],  # Bölme için minimum örnek sayısını artırın
                'min_samples_leaf': [4, 5, 10],  # Yaprak başına daha fazla örnek kullanın
                'max_features': ['sqrt', 'log2'],  # Daha az özellik kullanarak aşırı uyumu engelleyin
                'bootstrap': [True]  # Bootstrap kullanarak modelin genellemesini artırın
            }
        }
    }

    results = {}
    best_models = {}

    for model_name, config in models.items():
        print(f"\nModel: {model_name}")

        grid_search = GridSearchCV(config["model"], config["params"], cv=5, scoring='f1', verbose=2)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"En iyi hiperparametreler: {best_params}")

        report, metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test)
        print(f"Performans Raporu:\n{report}")
        print(f"Eğitim Doğruluğu: {metrics['Train Accuracy']:.4f}")
        print(f"Test Doğruluğu: {metrics['Test Accuracy']:.4f}")

        results[model_name] = {
            "Best Params": best_params,
            "Metrics": metrics
        }

        best_models[model_name] = best_model

    # ROC-AUC eğrisini kaydet
    plot_roc_curve(best_models, X_test, y_test, filename="roc_curvekarsilastirma.png")

    # Eğitim ve test doğruluğu histogramını kaydet
    plot_accuracy_histogram(results, filename="accuracy_histogramkarsilastirma.png")


if __name__ == "__main__":
    main()

