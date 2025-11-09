# ============================================================
# 🧠 CLASIFICACIÓN MULTICLASE CON REGRESIÓN LOGÍSTICA (OvR, OvO, MULTINOMIAL)
# ============================================================
# En este notebook se comparan tres estrategias de clasificación multiclase:
# 1️⃣ One-vs-Rest (OvR)
# 2️⃣ One-vs-One (OvO)
# 3️⃣ Regresión Logística Multinomial
# sobre un conjunto de datos artificial que representa cuatro tipos de cerveza:
# Lager, Stout, IPA y Scottish.
# ============================================================

# ==============================
# 📦 Importación de librerías
# ==============================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ==============================
# 🍺 Datos de ejemplo
# ==============================
# Creamos un dataset sintético con 4 clases de cerveza (Lager, Stout, IPA, Scottish)
# Cada muestra tiene dos características (por ejemplo: color y amargor)

X = np.array([
    [15, 20], [12, 15], [28, 39], [21, 30], [18, 25], [16, 22],  # Lager
    [45, 20], [40, 61], [42, 70], [48, 55], [50, 60],            # Stout
    [55, 25], [60, 18], [72, 22], [65, 20], [70, 19],            # IPA
    [22, 28], [30, 35], [25, 32], [28, 30], [27, 34]             # Scottish
])

# Etiquetas (0 = Lager, 1 = Stout, 2 = IPA, 3 = Scottish)
y = np.array([
    0, 0, 0, 0, 0, 0,  # Lager
    1, 1, 1, 1, 1,     # Stout
    2, 2, 2, 2, 2,     # IPA
    3, 3, 3, 3, 3      # Scottish
])

# ==============================
# 🔀 División en entrenamiento y prueba
# ==============================
# Dividimos el dataset en 70% entrenamiento y 30% prueba
# Stratify asegura que la proporción de clases se mantenga en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==============================
# ⚙️ Definición de modelos
# ==============================
# Definimos tres variantes de la regresión logística para clasificación multiclase:
# - OvR (One-vs-Rest)
# - OvO (One-vs-One)
# - Multinomial (regresión logística softmax)

base_clf = LogisticRegression(max_iter=500)

models = {
    "OvR": OneVsRestClassifier(base_clf),
    "OvO": OneVsOneClassifier(base_clf),
    "Multinomial": LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
}

# ==============================
# 🧪 Entrenamiento y Evaluación
# ==============================
# Entrenamos cada modelo, realizamos predicciones
# y mostramos métricas junto con la matriz de confusión

for name, model in models.items():
    # Entrenamiento
    model.fit(X_train, y_train)
    # Predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)

    # ==============================
    # 📊 Métricas de rendimiento
    # ==============================
    print(f"\n===== {name} =====")
    print("Accuracy:", model.score(X_test, y_test))
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Lager", "Stout", "IPA", "Scottish"]
    ))

    # ==============================
    # 🔍 Matriz de confusión
    # ==============================
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Lager", "Stout", "IPA", "Scottish"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusión — {name}")
    plt.show()
