# ============================================================
# 🌸 Clasificación del dataset Iris con Ensemble Learning (Stacking)
# Este código está preparado para ejecutarse en un Jupyter Notebook.
# Contiene comentarios explicativos en cada bloque para facilitar su comprensión.
# ============================================================

# --- Importación de librerías ---
from sklearn.datasets import load_iris  # Dataset Iris integrado en sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold  # División y validación
from sklearn.preprocessing import StandardScaler  # Escalado de variables
from sklearn.pipeline import Pipeline  # Construcción de pipelines
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay  # Métricas y visualización
from sklearn.tree import DecisionTreeClassifier  # Árbol de decisión
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors
from sklearn.linear_model import LogisticRegression  # Regresión logística
from sklearn.svm import SVC  # Máquina de soporte vectorial
from sklearn.ensemble import RandomForestClassifier, StackingClassifier  # Random Forest y Stacking
import numpy as np  # Cálculos numéricos
import matplotlib.pyplot as plt  # Gráficos
import warnings
warnings.filterwarnings("ignore")  # Ignorar warnings para mantener salida limpia

# --- 1) Carga y división de los datos ---
# Cargamos el dataset Iris (4 features, 3 clases)
X, y = load_iris(return_X_y=True)

# División del dataset: 80% entrenamiento, 20% test
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Definimos validación cruzada estratificada (mantiene proporciones de clases)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- 2) Definición de modelos base ---
# Se usan tres clasificadores base distintos para el ensemble:
# - Árbol de decisión (sin escalado)
# - KNN y Regresión Logística (con escalado)
base_dt  = ('dt',  DecisionTreeClassifier(max_depth=None, random_state=42))
base_knn = ('knn', Pipeline([
    ('scaler', StandardScaler()),  # Escala las características
    ('knn', KNeighborsClassifier(n_neighbors=7))
]))
base_lr  = ('lr',  Pipeline([
    ('scaler', StandardScaler()),  # Escala las características
    ('lr', LogisticRegression(max_iter=1000, multi_class='auto', random_state=42))
]))

# Lista de clasificadores base
base_estimators = [base_dt, base_knn, base_lr]

# --- 3) Definición de meta-modelos (nivel superior del stacking) ---
# Se prueban tres meta-modelos distintos para combinar las predicciones base.
meta_models = {
    'meta_LR'  : LogisticRegression(max_iter=1000, multi_class='auto', random_state=42),
    'meta_SVC' : SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    'meta_RF'  : RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
}

# --- 4) Baselines: rendimiento de cada modelo individual ---
print("== Baselines (modelos individuales) ==")
individuals = {
    'DecisionTree' : base_dt[1],
    'KNN'          : base_knn[1],
    'LogisticReg'  : base_lr[1]
}

# Se evalúa cada modelo base con validación cruzada (5 folds)
for name, clf in individuals.items():
    scores = cross_val_score(clf, Xtr, ytr, cv=cv, scoring='accuracy')
    print(f"{name:13s} | acc cv5 = {scores.mean():.4f} ± {scores.std():.4f}")

# --- 5) Construcción y evaluación de Stacking ---
print("\n== Stacking (DT + KNN + LR como bases) ==")
stack_scores = {}
best_name, best_mean = None, -np.inf
best_model = None

# Se entrena un StackingClassifier con cada meta-modelo definido
for name, meta in meta_models.items():
    stack = StackingClassifier(
        estimators=base_estimators,   # Modelos base
        final_estimator=meta,         # Meta-modelo
        cv=5,                         # Validación interna del meta-modelo
        stack_method='auto'           # Usa predict_proba o decision_function automáticamente
    )
    # Validación cruzada para medir desempeño promedio
    scores = cross_val_score(stack, Xtr, ytr, cv=cv, scoring='accuracy')
    stack_scores[name] = (scores.mean(), scores.std())
    print(f"{name:10s} | acc cv5 = {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Se guarda el mejor modelo
    if scores.mean() > best_mean:
        best_mean, best_name, best_model = scores.mean(), name, stack

# --- 6) Entrenamiento final del mejor modelo y evaluación en test ---
print(f"\n>> Mejor meta-modelo: {best_name} (cv mean acc = {best_mean:.4f})")

# Entrenamos con todos los datos de entrenamiento
best_model.fit(Xtr, ytr)

# Predicciones sobre el conjunto de test
yhat = best_model.predict(Xte)

# Cálculo de la precisión y reporte de clasificación
acc = accuracy_score(yte, yhat)
print(f"Accuracy en test (hold-out 20%): {acc:.4f}\n")
print(classification_report(yte, yhat, target_names=[f"clase_{i}" for i in np.unique(y)]))

# --- 7) Visualización: Matriz de confusión ---
# Se muestra cómo se distribuyeron los aciertos y errores de clasificación.
disp = ConfusionMatrixDisplay.from_predictions(yte, yhat)
plt.title(f"Matriz de confusión - Stacking ({best_name})")
plt.tight_layout()
plt.show()
