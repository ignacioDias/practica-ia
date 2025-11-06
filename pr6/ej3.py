import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

# --- 1. Cargar y Pre-procesar los Datos (Fashion-MNIST) ---

# Cargamos el dataset (ya viene dividido en train y test)
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Normalizar los datos: Los píxeles van de 0-255. Los escalamos a 0.0-1.0
# para que la red neuronal aprenda más eficientemente.
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Reformar (Reshape) los datos:
# Las capas Conv2D en Keras esperan 4 dimensiones:
# (cantidad_imagenes, alto, ancho, canales_color)
# Como Fashion-MNIST es blanco y negro, usamos 1 canal.
X_train = X_train_full.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

# Convertir las etiquetas (y) a formato one-hot encoding (categórico)
# Ej: La etiqueta '5' se convierte en [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# Esto es necesario para la 'categorical_crossentropy' [cite: 746]
y_train = to_categorical(y_train_full, 10)
y_test_cat = to_categorical(y_test, 10)


# --- 2. Definir la Arquitectura del Modelo CNN ---

# Usamos un modelo Secuencial [cite: 738]
model = Sequential()

# --- Etapa de Extracción de Características [cite: 614] ---

# Capa 1: Convolucional
# 32 filtros (kernels) de tamaño 3x3
# 'relu' como función de activación
# 'input_shape' solo se pone en la primera capa
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Capa 2: Pooling
# Reduce el tamaño a la mitad (2x2) [cite: 657-658]
# Esto ayuda a la red a ser invariante a pequeñas traslaciones
model.add(MaxPooling2D((2, 2)))

# Capa 3: Convolucional (más profunda)
# Añadimos otra capa para aprender patrones más complejos
model.add(Conv2D(64, (3, 3), activation='relu'))

# Capa 4: Pooling
model.add(MaxPooling2D((2, 2)))

# --- Etapa de Clasificación (como en una ANN) ---

# Capa 5: Aplanado (Flatten)
# Convierte los mapas de características 2D en un vector 1D [cite: 634]
# para poder conectarlo a las capas Densas.
model.add(Flatten())

# Capa 6: Densa (Fully Connected)
# Una capa oculta de 128 neuronas para la clasificación
model.add(Dense(128, activation='relu'))

# Capa 7: Salida
# 10 neuronas (1 por cada clase de ropa)
# 'softmax' para obtener probabilidades de cada clase
model.add(Dense(10, activation='softmax'))

# Imprimir un resumen del modelo
print("Arquitectura del modelo CNN:")
model.summary()


# --- 3. Compilar el Modelo ---

# Usamos 'adam' como optimizador y 'categorical_crossentropy' como
# función de pérdida, ya que tenemos múltiples clases [cite: 746, 752]
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# --- 4. Entrenar el Modelo ---

print("\n--- Iniciando Entrenamiento ---")
# Entrenamos el modelo con los datos [cite: 754]
# epochs: cuántas veces ve el dataset completo
# batch_size: cuántas imágenes procesa antes de actualizar pesos
# validation_split: usa un 10% de los datos de entreno para validar
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=64,
                    validation_split=0.1)

print("--- Entrenamiento Finalizado ---")


# --- 5. Evaluar y Registrar Métricas ---

# Evaluamos el rendimiento del modelo con los datos de Test
print("\n--- Evaluación en el set de Prueba ---")
loss, acc = model.evaluate(X_test, y_test_cat)

# Registramos y mostramos las métricas
print(f"Pérdida (Loss) en Test: {loss:.4f}")
print(f"Precisión (Accuracy) en Test: {acc*100:.2f}%")