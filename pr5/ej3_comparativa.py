# ===============================================================
# 🌸 Clasificación del dataset Iris con MLP, SVM y Árbol de Decisión
# Este notebook entrena y compara tres modelos distintos de clasificación
# sobre el clásico dataset Iris. Cada sección está comentada para mayor claridad.
# ===============================================================

# --- Importación de librerías necesarias ---
from sklearn.datasets import load_iris                  # Dataset Iris
from sklearn.model_selection import train_test_split    # División train/test
from sklearn.preprocessing import StandardScaler        # Normalización de datos
from sklearn.neural_network import MLPClassifier        # Red neuronal multicapa (MLP)
from sklearn.svm import SVC                             # Máquina de soporte vectorial
from sklearn.tree import DecisionTreeClassifier          # Árbol de decisión
from sklearn.metrics import accuracy_score, classification_report  # Métricas
import pandas as pd                                     # Para mostrar resultados tabulados

# --- 1. Carga del dataset ---
# El dataset Iris contiene 150 muestras con 4 características (longitud/peso de pétalos y sépalos)
# y 3 clases de flores: setosa, versicolor y virginica.
iris = load_iris()
X = iris.data
y = iris.target

# --- 2. División del conjunto de datos y escalado ---
# Se divide el dataset en 70% entrenamiento y 30% test.
# Además, se aplica estandarización para mejorar el rendimiento de SVM y MLP.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # Estratificación mantiene proporción de clases
)

# Escalado de características (media 0, desviación estándar 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3. Definición de modelos ---
# Se seleccionan tres modelos para comparar:
# - MLP: red neuronal con 2 capas ocultas de 10 neuronas cada una.
# - SVM: clasificador de vectores de soporte con kernel RBF.
# - Árbol de decisión: clasificador basado en divisiones jerárquicas.
models = {
    "MLP (Red Neuronal)": MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42),
    "SVM (Máquina de Vectores de Soporte)": SVC(kernel='rbf', C=1, gamma='scale', random_state=42),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42)
}

# --- 4. Entrenamiento y evaluación de los modelos ---
# Se entrena cada modelo y se evalúa con exactitud y reporte de clasificación.
results = []  # Lista para guardar resultados numéricos de cada modelo

for name, model in models.items():
    model.fit(X_train, y_train)                # Entrenamiento
    y_pred = model.predict(X_test)             # Predicciones sobre el test set
    acc = accuracy_score(y_test, y_pred)       # Cálculo de la exactitud (accuracy)
    
    print("=" * 60)
    print(f"🏷️  Modelo: {name}")
    print(f"📈 Exactitud: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Se guarda el resultado para el resumen final
    results.append({"Modelo": name, "Exactitud": acc})

# --- 5. Mostrar resumen ordenado de resultados ---
# Se crea un DataFrame con los resultados y se ordena por exactitud.
print("=" * 60)
print("\n📊 Resumen de exactitudes:\n")
df_results = pd.DataFrame(results).sort_values(by="Exactitud", ascending=False)
print(df_results.to_string(index=False))
