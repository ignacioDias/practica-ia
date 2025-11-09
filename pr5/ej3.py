# ======================================================
# 🧠 Entrenamiento de un Perceptrón Multicapa (MLP) con el dataset Iris
# Este notebook entrena una red neuronal multicapa sobre el dataset Iris,
# evalúa su rendimiento y muestra métricas de clasificación.
# ======================================================

# --- Importación de librerías necesarias ---
from sklearn.datasets import load_iris                   # Dataset Iris
from sklearn.model_selection import train_test_split     # División train/test
from sklearn.preprocessing import StandardScaler         # Normalización
from sklearn.neural_network import MLPClassifier         # Modelo MLP
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Métricas

# --- 1. Cargar dataset ---
# El dataset Iris contiene 150 muestras con 4 características y 3 clases de flores.
iris = load_iris()
X = iris.data        # Características: largo y ancho de pétalos y sépalos
y = iris.target      # Etiquetas de clase (0=setosa, 1=versicolor, 2=virginica)

# --- 2. Separar en conjuntos de entrenamiento y prueba ---
# Separamos 70% de los datos para entrenamiento y 30% para prueba,
# usando estratificación para mantener proporción de clases.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 3. Estandarización de los datos ---
# La red neuronal requiere que los datos estén normalizados para un aprendizaje estable.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Ajusta y transforma los datos de entrenamiento
X_test = scaler.transform(X_test)        # Solo transforma los datos de prueba

# --- 4. Crear el modelo MLP ---
# Se define una red con dos capas ocultas de 10 neuronas cada una,
# activación ReLU y optimizador Adam. Se limita a 1000 iteraciones.
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10),  # dos capas ocultas con 10 neuronas cada una
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# --- 5. Entrenar el modelo ---
# Entrenamos el modelo con los datos de entrenamiento normalizados.
mlp.fit(X_train, y_train)

# --- 6. Evaluar el modelo ---
# Se realizan predicciones y se evalúan las métricas de rendimiento.
y_pred = mlp.predict(X_test)

# --- 7. Mostrar resultados ---
print("Exactitud (accuracy):", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
