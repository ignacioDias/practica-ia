import tensorflow as tf
# ==========================================================
# 🔹 PRÁCTICA IA - CLASIFICACIÓN CON CNN (Fashion-MNIST)
# ==========================================================
# Este Notebook entrena una red neuronal convolucional (CNN)
# para clasificar imágenes del dataset Fashion-MNIST.
# ----------------------------------------------------------

# --- Importación de librerías necesarias ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

# ==========================================================
# 🧩 1. Cargar y Pre-procesar los Datos (Fashion-MNIST)
# ==========================================================

# Cargamos el dataset predividido en entrenamiento y prueba
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Normalizamos los valores de píxeles (0 a 255 → 0.0 a 1.0)
# Esto mejora la estabilidad numérica y acelera el aprendizaje
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Reformateamos las imágenes a 4D (batch, alto, ancho, canales)
# CNN espera imágenes con 1 canal (escala de grises)
X_train = X_train_full.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

# Convertimos las etiquetas a formato categórico (one-hot)
# Ejemplo: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train_full, 10)
y_test_cat = to_categorical(y_test, 10)

# Mostramos tamaños para verificar consistencia
print("Tamaño del set de entrenamiento:", X_train.shape)
print("Tamaño del set de prueba:", X_test.shape)

# ==========================================================
# 🧠 2. Definir la Arquitectura del Modelo CNN
# ==========================================================

# Creamos un modelo secuencial (capa por capa)
model = Sequential()

# --- Etapa de Extracción de Características ---
# Primera capa convolucional: 32 filtros 3x3 con activación ReLU
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Pooling 2x2: reduce dimensiones, mantiene rasgos importantes
model.add(MaxPooling2D((2, 2)))

# Segunda capa convolucional: 64 filtros 3x3
model.add(Conv2D(64, (3, 3), activation='relu'))

# Segundo pooling para reducir tamaño de nuevo
model.add(MaxPooling2D((2, 2)))

# --- Etapa de Clasificación ---
# Aplanamos las características 2D a vector 1D
model.add(Flatten())

# Capa densa oculta de 128 neuronas con ReLU
model.add(Dense(128, activation='relu'))

# Capa de salida: 10 neuronas (una por clase), activación softmax
model.add(Dense(10, activation='softmax'))

# Resumen del modelo (útil para Notebook)
print("📘 Arquitectura del modelo CNN:")
model.summary()

# ==========================================================
# ⚙️ 3. Compilar el Modelo
# ==========================================================
# Definimos el optimizador, la función de pérdida y las métricas
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================================
# 🚀 4. Entrenar el Modelo
# ==========================================================
print("\n--- Iniciando Entrenamiento ---")

# Entrenamos la red:
# - epochs: número de pasadas por todo el dataset
# - batch_size: tamaño de los lotes de entrenamiento
# - validation_split: 10% de los datos para validación
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

print("--- Entrenamiento Finalizado ---")

# ==========================================================
# 📊 5. Evaluar y Registrar Métricas
# ==========================================================
print("\n--- Evaluación en el set de Prueba ---")

# Evaluamos en los datos de prueba no vistos
loss, acc = model.evaluate(X_test, y_test_cat)

# Mostramos resultados finales
print(f"Pérdida (Loss) en Test: {loss:.4f}")
print(f"Precisión (Accuracy) en Test: {acc*100:.2f}%")

# ==========================================================
# 📈 (Opcional) Visualizar el historial de entrenamiento
# ==========================================================
# En una celda extra en Jupyter podés graficar la evolución:
#
# import matplotlib.pyplot as plt
# plt.plot(history.history['accuracy'], label='Entrenamiento')
# plt.plot(history.history['val_accuracy'], label='Validación')
# plt.title('Precisión a lo largo de las épocas')
# plt.xlabel('Época')
# plt.ylabel('Precisión')
# plt.legend()
# plt.show()
#
# Esto te permite visualizar el rendimiento del modelo.
# ==========================================================
