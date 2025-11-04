# ======================================================
# Entrenamiento de un MLP con el dataset Iris
# ======================================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Cargar dataset
iris = load_iris()
X = iris.data        # Características
y = iris.target      # Etiquetas

# 2. Separar en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Estandarizar (muy importante para MLP)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Crear el modelo MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10),  # dos capas ocultas con 10 neuronas cada una
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# 5. Entrenar el modelo
mlp.fit(X_train, y_train)

# 6. Evaluar el modelo
y_pred = mlp.predict(X_test)

print("Exactitud (accuracy):", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
