import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar dataset
df = pd.read_csv("screentime.csv")

# Eliminar columnas no numéricas o irrelevantes (por ejemplo user_id, gender, occupation, work_mode)
df = df.drop(columns=['user_id', 'gender', 'occupation', 'work_mode'])

# Definir variables
X = df.drop(columns=['sleep_quality_1_5'])
y = df['sleep_quality_1_5']

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Árbol de decisión
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest (Bagging)
rf = RandomForestClassifier(n_estimators=100, max_samples=0.8, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluación: matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Decision Tree')
axes[0].set_xlabel('Predicho')
axes[0].set_ylabel('Real')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest')
axes[1].set_xlabel('Predicho')
axes[1].set_ylabel('Real')

plt.tight_layout()
plt.show()

# Reportes de clasificación
print("=== Decision Tree ===")
print(classification_report(y_test, y_pred_dt))
print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
