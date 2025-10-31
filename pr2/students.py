import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el dataset

data = pd.read_csv("StudentsPerformance.csv")

# predecir el puntaje de matemáticas en base a las otras variables.

X = data.drop("math score", axis=1)
y = data["math score"]

# Separar train/test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# data.hist(bins=50, figsize=(12, 8))
# plt.show()

# Columnas categóricas y numéricas

cat_features = X.select_dtypes(include="object").columns
num_features = X.select_dtypes(exclude="object").columns

# Definir polinomio sobre variables numéricas
poly = PolynomialFeatures(degree=3, include_bias=False)

# Nuevo preprocesador
preprocessor_poly = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("num", Pipeline([("scaler", StandardScaler()), ("poly", poly)]), num_features)
])

poly_model = Pipeline([
    ("preprocessor", preprocessor_poly),
    ("regressor", LinearRegression())
])

poly_model.fit(X_train, y_train)

#Evaluación
y_pred_poly = poly_model.predict(X_test)

# Modelo lineal base
from sklearn.pipeline import Pipeline

lin_model = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features)
    ])),
    ("regressor", LinearRegression())
])

lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

print("MSE (Lineal):", mean_squared_error(y_test, y_pred_lin))
print("R² (Lineal):", r2_score(y_test, y_pred_lin))


print("MSE (Polinomial grado 2):", mean_squared_error(y_test, y_pred_poly))
print("R² (Polinomial grado 2):", r2_score(y_test, y_pred_poly))


#Graficar

plt.scatter(y_test, y_pred_lin, alpha=0.6, color="blue", label="Lineal")
plt.scatter(y_test, y_pred_poly, alpha=0.6, color="red", label="Polinomial grado 2")
plt.xlabel("Valores reales (math score)")
plt.ylabel("Predicciones")
plt.title("Comparación Lineal vs Polinomial")
plt.legend()
plt.show()