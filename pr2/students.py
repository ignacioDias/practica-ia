# ============================================
# 📘 REGRESIÓN LINEAL VS POLINOMIAL EN PYTHON
# ============================================

# En este notebook se entrena un modelo para predecir el puntaje de matemáticas ("math score")
# de estudiantes, usando las demás variables del dataset "StudentsPerformance.csv".
# Luego se compara el rendimiento entre un modelo lineal y uno polinomial.

# Importación de librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# 🔹 CARGA DEL DATASET
# ============================================

# Leemos el archivo CSV con la información de los estudiantes
data = pd.read_csv("StudentsPerformance.csv")

# ============================================
# 🔹 DEFINICIÓN DE VARIABLES
# ============================================

# Variable objetivo (target): "math score"
# Variables predictoras (features): todas las demás columnas
X = data.drop("math score", axis=1)
y = data["math score"]

# ============================================
# 🔹 DIVISIÓN ENTRE TRAIN Y TEST
# ============================================

# Dividimos los datos en 80% entrenamiento y 20% prueba
# random_state=42 asegura reproducibilidad de resultados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================
# 🔹 ANÁLISIS OPCIONAL
# ============================================

# Si se desea visualizar las distribuciones de las variables
# descomentar las siguientes líneas:
# data.hist(bins=50, figsize=(12, 8))
# plt.show()

# ============================================
# 🔹 IDENTIFICACIÓN DE VARIABLES CATEGÓRICAS Y NUMÉRICAS
# ============================================

# Detectamos las variables categóricas (tipo texto)
cat_features = X.select_dtypes(include="object").columns

# Detectamos las variables numéricas (si las hubiera)
num_features = X.select_dtypes(exclude="object").columns

# ============================================
# 🔹 MODELO POLINOMIAL (Grado 3)
# ============================================

# Creamos las características polinómicas sobre las variables numéricas
# Esto permite capturar relaciones no lineales
poly = PolynomialFeatures(degree=3, include_bias=False)

# Definimos el preprocesamiento:
# - Codificamos las variables categóricas con OneHotEncoder
# - Estandarizamos las variables numéricas y aplicamos expansión polinómica
preprocessor_poly = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("num", Pipeline([("scaler", StandardScaler()), ("poly", poly)]), num_features)
])

# Creamos el pipeline completo:
# 1. Preprocesamiento de datos
# 2. Regresión lineal sobre las features transformadas
poly_model = Pipeline([
    ("preprocessor", preprocessor_poly),
    ("regressor", LinearRegression())
])

# Entrenamos el modelo polinomial con el conjunto de entrenamiento
poly_model.fit(X_train, y_train)

# ============================================
# 🔹 EVALUACIÓN DEL MODELO POLINOMIAL
# ============================================

# Realizamos predicciones sobre el conjunto de prueba
y_pred_poly = poly_model.predict(X_test)

# ============================================
# 🔹 MODELO LINEAL BASE
# ============================================

# Importamos nuevamente Pipeline (ya está arriba, pero se deja por claridad)
from sklearn.pipeline import Pipeline

# Definimos un modelo lineal básico sin términos polinómicos
lin_model = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features)
    ])),
    ("regressor", LinearRegression())
])

# Entrenamos el modelo lineal
lin_model.fit(X_train, y_train)

# Realizamos predicciones con el modelo lineal
y_pred_lin = lin_model.predict(X_test)

# ============================================
# 🔹 COMPARACIÓN DE MÉTRICAS
# ============================================

# Calculamos el Error Cuadrático Medio (MSE) y el Coeficiente de Determinación (R²)
# para ambos modelos y comparamos sus resultados
print("MSE (Lineal):", mean_squared_error(y_test, y_pred_lin))
print("R² (Lineal):", r2_score(y_test, y_pred_lin))

print("MSE (Polinomial grado 2):", mean_squared_error(y_test, y_pred_poly))
print("R² (Polinomial grado 2):", r2_score(y_test, y_pred_poly))

# ============================================
# 🔹 VISUALIZACIÓN DE RESULTADOS
# ============================================

# Graficamos las predicciones frente a los valores reales para ambos modelos
plt.scatter(y_test, y_pred_lin, alpha=0.6, color="blue", label="Lineal")              # Puntos azules: modelo lineal
plt.scatter(y_test, y_pred_poly, alpha=0.6, color="red", label="Polinomial grado 2")  # Puntos rojos: modelo polinomial
plt.xlabel("Valores reales (math score)")    # Eje X: puntaje verdadero
plt.ylabel("Predicciones")                   # Eje Y: puntaje predicho
plt.title("Comparación Lineal vs Polinomial") # Título del gráfico
plt.legend()                                 # Mostrar leyenda
plt.show()                                   # Mostrar la figura
