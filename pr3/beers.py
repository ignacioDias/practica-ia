import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Datos
X = np.array([
    [15, 20], [12, 15], [28, 39], [21, 30], [18, 25], [16, 22],  # Lager
    [45, 20], [40, 61], [42, 70], [48, 55], [50, 60],            # Stout
    [55, 25], [60, 18], [72, 22], [65, 20], [70, 19],            # IPA
    [22, 28], [30, 35], [25, 32], [28, 30], [27, 34]             # Scottish
])

y = np.array([
    0, 0, 0, 0, 0, 0,  # Lager
    1, 1, 1, 1, 1,     # Stout
    2, 2, 2, 2, 2,     # IPA
    3, 3, 3, 3, 3      # Scottish
])

# División entrenamiento/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Modelos
base_clf = LogisticRegression(max_iter=500)

models = {
    "OvR": OneVsRestClassifier(base_clf),
    "OvO": OneVsOneClassifier(base_clf),
    "Multinomial": LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
}

# Entrenar y evaluar
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")
    print("Accuracy:", model.score(X_test, y_test))
    print(classification_report(y_test, y_pred, target_names=["Lager", "Stout", "IPA", "Scottish"]))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Lager", "Stout", "IPA", "Scottish"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusión — {name}")
    plt.show()
