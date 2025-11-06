import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.utils import to_categorical

# ================================
#  1. FUNCIÓN PARA GENERAR TEXTO
# ================================
def generar_texto(model, char_to_int, int_to_char, vocab_size, seed_pattern, n_chars=40):
    """
    Genera texto carácter a carácter usando un modelo entrenado (LSTM o GRU).
    
    Parámetros:
      - model: modelo entrenado
      - char_to_int: diccionario de caracteres a índices
      - int_to_char: diccionario de índices a caracteres
      - vocab_size: cantidad de caracteres únicos
      - seed_pattern: lista con los índices iniciales
      - n_chars: cantidad de caracteres a generar
    """
    generated_text = ""
    current_pattern = list(seed_pattern)

    for i in range(n_chars):
        x_input = np.zeros((1, seq_length, vocab_size), dtype=bool)
        for t, char_index in enumerate(current_pattern):
            x_input[0, t, char_index] = 1

        prediccion = model.predict(x_input, verbose=0)
        index = np.argmax(prediccion)
        result_char = int_to_char[index]
        generated_text += result_char

        current_pattern.append(index)
        current_pattern = current_pattern[1:len(current_pattern)]
    return generated_text


# ================================
#  2. PREPARACIÓN DE LOS DATOS
# ================================
text = "hola mundo estoy aprendiendo redes neuronales recurrentes con keras"
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

seq_length = 5
epochs = 50

X_data = []
y_data = []
for i in range(0, len(text) - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    X_data.append([char_to_int[char] for char in seq_in])
    y_data.append(char_to_int[seq_out])

n_patterns = len(X_data)
print(f"Total de patrones (secuencias): {n_patterns}")

# One-hot encoding
X = np.zeros((n_patterns, seq_length, vocab_size), dtype=bool)
y = np.zeros((n_patterns, vocab_size), dtype=bool)

for i, pattern in enumerate(X_data):
    for t, char_index in enumerate(pattern):
        X[i, t, char_index] = 1
    y[i, y_data[i]] = 1


# ================================
#  3. MODELOS: LSTM vs GRU
# ================================
def crear_modelo_lstm(seq_length, vocab_size):
    model = Sequential()
    model.add(LSTM(32, input_shape=(seq_length, vocab_size)))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def crear_modelo_gru(seq_length, vocab_size):
    model = Sequential()
    model.add(GRU(32, input_shape=(seq_length, vocab_size)))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# ================================
#  4. ENTRENAMIENTO Y COMPARACIÓN
# ================================
print("\n--- 🚀 ENTRENANDO MODELOS ---\n")

# Modelo LSTM
print("Entrenando modelo LSTM...")
model_lstm = crear_modelo_lstm(seq_length, vocab_size)
model_lstm.fit(X, y, epochs=epochs, batch_size=4, verbose=0)
loss_lstm, acc_lstm = model_lstm.evaluate(X, y, verbose=0)
print(f"✅ LSTM completado. Exactitud final: {acc_lstm:.4f}")

# Modelo GRU
print("\nEntrenando modelo GRU...")
model_gru = crear_modelo_gru(seq_length, vocab_size)
model_gru.fit(X, y, epochs=epochs, batch_size=4, verbose=0)
loss_gru, acc_gru = model_gru.evaluate(X, y, verbose=0)
print(f"✅ GRU completado. Exactitud final: {acc_gru:.4f}")


# ================================
#  5. GENERACIÓN DE TEXTO
# ================================
seed_pattern_int = X_data[0]
seed_text = "".join([int_to_char[val] for val in seed_pattern_int])

texto_generado_lstm = generar_texto(model_lstm, char_to_int, int_to_char, vocab_size, seed_pattern_int, 40)
texto_generado_gru = generar_texto(model_gru, char_to_int, int_to_char, vocab_size, seed_pattern_int, 40)


# ================================
#  6. RESULTADOS FINALES
# ================================
print("\n===============================")
print("📊 RESULTADOS COMPARATIVOS")
print("===============================")
print(f"Texto original base: '{text[:30]}...'")
print(f"Secuencia inicial (semilla): '{seed_text}'\n")

print(f"[LSTM] Exactitud: {acc_lstm:.4f}")
print(f"[LSTM] Texto generado: {texto_generado_lstm}\n")

print(f"[GRU] Exactitud: {acc_gru:.4f}")
print(f"[GRU] Texto generado: {texto_generado_gru}\n")

if acc_gru > acc_lstm:
    print("✅ El modelo GRU tuvo mejor desempeño en este experimento.")
elif acc_lstm > acc_gru:
    print("✅ El modelo LSTM tuvo mejor desempeño en este experimento.")
else:
    print("⚖  Ambos modelos tuvieron un desempeño similar.")