from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 1) Datos
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2) Modelos base (DT sin escalar; KNN y LR con escalado)
base_dt  = ('dt',  DecisionTreeClassifier(max_depth=None, random_state=42))
base_knn = ('knn', Pipeline([('scaler', StandardScaler()),
                             ('knn', KNeighborsClassifier(n_neighbors=7))]))
base_lr  = ('lr',  Pipeline([('scaler', StandardScaler()),
                             ('lr', LogisticRegression(max_iter=1000, multi_class='auto', random_state=42))]))

base_estimators = [base_dt, base_knn, base_lr]

# 3) Meta-modelos a probar
meta_models = {
    'meta_LR'  : LogisticRegression(max_iter=1000, multi_class='auto', random_state=42),
    'meta_SVC' : SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    'meta_RF'  : RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
}

# 4) Baselines: cada base por separado
print("== Baselines (modelos individuales) ==")
individuals = {
    'DecisionTree' : base_dt[1],
    'KNN'          : base_knn[1],
    'LogisticReg'  : base_lr[1]
}
for name, clf in individuals.items():
    scores = cross_val_score(clf, Xtr, ytr, cv=cv, scoring='accuracy')
    print(f"{name:13s} | acc cv5 = {scores.mean():.4f} ± {scores.std():.4f}")

# 5) Stacking con distintos meta-modelos
print("\n== Stacking (DT + KNN + LR como bases) ==")
stack_scores = {}
best_name, best_mean = None, -np.inf
best_model = None

for name, meta in meta_models.items():
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        cv=5,               # meta-entrenamiento con k-fold
        stack_method='auto' # usa predict_proba/decision_function según disp.
    )
    scores = cross_val_score(stack, Xtr, ytr, cv=cv, scoring='accuracy')
    stack_scores[name] = (scores.mean(), scores.std())
    print(f"{name:10s} | acc cv5 = {scores.mean():.4f} ± {scores.std():.4f}")
    if scores.mean() > best_mean:
        best_mean, best_name, best_model = scores.mean(), name, stack

# 6) Entrenar el mejor stacking en todo el train y evaluar en test
print(f"\n>> Mejor meta-modelo: {best_name} (cv mean acc = {best_mean:.4f})")
best_model.fit(Xtr, ytr)
yhat = best_model.predict(Xte)
acc = accuracy_score(yte, yhat)
print(f"Accuracy en test (hold-out 20%): {acc:.4f}\n")
print(classification_report(yte, yhat, target_names=[f"clase_{i}" for i in np.unique(y)]))

# 7) Matriz de confusión
disp = ConfusionMatrixDisplay.from_predictions(yte, yhat)
plt.title(f"Matriz de confusión - Stacking ({best_name})")
plt.tight_layout()
plt.show()
