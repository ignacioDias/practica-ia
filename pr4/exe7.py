# ============================================================
# 📘 Clasificación de calidad del sueño usando Árboles y Bosques Aleatorios
# Este código está preparado para ejecutarse en un Jupyter Notebook.
# Incluye comentarios explicativos para comprender cada paso del proceso.
# ============================================================

# --- Importación de librerías necesarias ---
import pandas as pd  # Para manejar datasets en formato DataFrame
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba
from sklearn.tree import DecisionTreeClassifier  # Clasificador basado en árboles de decisión
from sklearn.ensemble import RandomForestClassifier  # Clasificador basado en bosques aleatorios (bagging)
from sklearn.metrics import confusion_matrix, classification_report  # Métricas de evaluación
import seaborn as sns  # Librería para visualización
import matplotlib.pyplot as plt  # Librería para gráficos

# --- Cargar el dataset ---
# Asegurate de tener el archivo 'screentime.csv' en el mismo directorio del notebook.
df = pd.read_csv("screentime.csv")

# --- Preprocesamiento de datos ---
# Eliminamos columnas no numéricas o irrelevantes para el modelo.
# Estas columnas pueden no aportar información útil al entrenamiento.
df = df.drop(columns=['user_id', 'gender', 'occupation', 'work_mode'])

# --- Definición de variables ---
# 'X' contiene las características predictoras.
# 'y' contiene la variable objetivo: la calidad del sueño en una escala del 1 al 5.
X = df.drop(columns=['sleep_quality_1_5'])
y = df['sleep_quality_1_5']

# --- División del dataset ---
# Separación en conjunto de entrenamiento (70%) y prueba (30%).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Entrenamiento del modelo de Árbol de Decisión ---
# Se limita la profundidad máxima a 5 para evitar sobreajuste.
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# --- Entrenamiento del modelo de Random Forest ---
# Random Forest combina múltiples árboles entrenados sobre subconjuntos del dataset (bagging).
rf = RandomForestClassifier(n_estimators=100, max_samples=0.8, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# --- Evaluación mediante matrices de confusión ---
# Se generan dos gráficos lado a lado para comparar los modelos.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Matriz de confusión para Árbol de Decisión
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Decision Tree')
axes[0].set_xlabel('Predicho')
axes[0].set_ylabel('Real')

# Matriz de confusión para Random Forest
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest')
axes[1].set_xlabel('Predicho')
axes[1].set_ylabel('Real')

# Ajuste del diseño de los gráficos
plt.tight_layout()
plt.show()

# --- Reportes de clasificación ---
# Muestra métricas como precisión, recall y F1-score para ambos modelos.
print("=== Decision Tree ===")
print(classification_report(y_test, y_pred_dt))

print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))