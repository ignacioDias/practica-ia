from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 1. Dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split y escalado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Modelos
models = {
    "MLP (Red Neuronal)": MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42),
    "SVM (Máquina de Vectores de Soporte)": SVC(kernel='rbf', C=1, gamma='scale', random_state=42),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42)
}

# 4. Entrenamiento y evaluación
results = []  # Para guardar los resultados

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("=" * 60)
    print(f"🏷️  Modelo: {name}")
    print(f"📈 Exactitud: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    results.append({"Modelo": name, "Exactitud": acc})

# 5. Mostrar resumen ordenado
print("=" * 60)
print("\n📊 Resumen de exactitudes:\n")
df_results = pd.DataFrame(results).sort_values(by="Exactitud", ascending=False)
print(df_results.to_string(index=False))
