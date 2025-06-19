### **Instrucciones**

#### **Efectos de Label Smoothing en la Confianza y Generalización (Evaluación)**

##### **Descripción**
En esta evaluación, investigarás el efecto de **Label Smoothing**, una técnica de regularización que previene que el modelo se vuelva demasiado confiado en sus predicciones. Para ello, entrenarás dos modelos CNN idénticos en el dataset CIFAR-10: un modelo base con una función de pérdida estándar y un segundo modelo que utiliza la misma pérdida pero con el parámetro de `label_smoothing` activado. El análisis se centrará en comparar no solo la precisión, sino también la distribución de la confianza de las predicciones.

##### **Objetivo**
El objetivo principal es demostrar empíricamente los efectos de Label Smoothing. Deberás:
* Implementar y entrenar un modelo CNN de clasificación de imágenes como línea base.
* Implementar y entrenar una segunda instancia del mismo modelo activando Label Smoothing en la función de pérdida.
* Comparar las curvas de aprendizaje para evaluar el impacto de la técnica en la generalización del modelo.
* Realizar un análisis cuantitativo y visual de la confianza de las predicciones para demostrar cómo Label Smoothing reduce la sobreconfianza.

##### **Instrucciones**

1.  **Carga y Preparación de Datos:**
    * Carga el dataset **CIFAR-10** desde `tensorflow.keras.datasets`.
    * Normaliza las imágenes al rango [0, 1] y convierte las etiquetas a formato categórico (one-hot encoding).

2.  **Construcción y Entrenamiento del Modelo Base:**
    * Define y construye una arquitectura CNN. Puedes usar la proporcionada en el laboratorio de solución o una similar con al menos dos bloques convolucionales y una cabeza clasificadora con `Dropout` y `BatchNormalization`.
    * Compila este modelo utilizando el optimizador `Adam` y la función de pérdida estándar `CategoricalCrossentropy`.
    * Entrena el modelo durante **25 épocas**, guardando el historial de entrenamiento.

3.  **Construcción y Entrenamiento del Modelo con Label Smoothing:**
    * Construye una **instancia idéntica** de la misma arquitectura de red neuronal para asegurar una comparación justa.
    * Compila este segundo modelo con los mismos hiperparámetros que el modelo base, pero con una diferencia clave: instancia la función de pérdida con el parámetro de suavizado, así: `loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)`.
    * Entrena este modelo con los mismos datos y durante el mismo número de épocas.

4.  **Comparación del Rendimiento (Generalización):**
    * Crea una única gráfica que muestre las curvas de **precisión en validación (`val_accuracy`)** de ambos modelos (base y con label smoothing) a lo largo de las épocas.
    * Etiqueta claramente cada curva para una fácil comparación.

5.  **Análisis de la Confianza de las Predicciones:**
    * Obtén las predicciones (probabilidades softmax) para el conjunto de prueba (`x_test`) de ambos modelos.
    * Para cada modelo, identifica las predicciones que fueron **correctas**.
    * De esas predicciones correctas, extrae la probabilidad máxima de cada una (este es el "nivel de confianza" del modelo para esa predicción).
    * Crea un **histograma** que compare la distribución de estos niveles de confianza para el modelo base y el modelo con label smoothing.

6.  **Discusión de Resultados:**
    * Basándote en tus gráficas y resultados, responde a las siguientes preguntas:
        * ¿Cómo afectó Label Smoothing a la precisión final en validación?
        * Describe la diferencia principal que observas en los histogramas de confianza. ¿Qué le hizo Label Smoothing a las predicciones del modelo?
        * Explica por qué un modelo "menos confiado" podría ser más deseable en aplicaciones críticas del mundo real.

##### **Sugerencias**

* La clave de este laboratorio está en la compilación: la única diferencia entre los dos modelos debe ser la inicialización de la función de pérdida.
* Para el análisis de confianza, las funciones `np.argmax()` para encontrar la clase predicha y `np.max()` para encontrar la probabilidad máxima de la predicción serán muy útiles.
* Al graficar el histograma de confianzas, considera usar `plt.xlim(0.8, 1.0)` para hacer zoom en la zona de alta confianza, donde las diferencias son más notorias.

---
## Ejercicio Práctico: Efectos de Label Smoothing en la Confianza y Generalización

### **Objetivo del Ejercicio**
1.  Entrenar dos modelos CNN idénticos en el dataset CIFAR-10.
2.  El primer modelo usará la función de pérdida de entropía cruzada categórica estándar.
3.  El segundo modelo usará la misma función de pérdida pero con **Label Smoothing** activado.
4.  Comparar el rendimiento de ambos modelos (generalización) y, más importante, la **confianza** de sus predicciones.

### **Entorno**
Este laboratorio utiliza **Python**, **TensorFlow** y **Keras**.

---
### **Parte 0: Configuración y Preparación del Dataset**
Comenzamos con la configuración habitual, importando librerías y preparando el dataset CIFAR-10.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Cargar el dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalizar los valores de los píxeles al rango [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convertir las etiquetas a formato one-hot encoding
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print(f"Forma de x_train: {x_train.shape}")
print(f"Forma de y_train (categórica): {y_train_cat.shape}")
```

---
### **Parte 1: Construcción y Entrenamiento del Modelo Base (Sin Label Smoothing)**
Primero, definimos nuestra arquitectura CNN. Será la misma para ambos experimentos para asegurar una comparación justa. Luego, la compilaremos con la función de pérdida estándar.

```python
def build_model(input_shape, num_classes):
    """Construye una CNN simple pero efectiva."""
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model

# --- Entrenamiento del Modelo Base ---
print("--- Entrenando Modelo Base (Sin Label Smoothing) ---")

# Construir el modelo
base_model = build_model(x_train.shape[1:], num_classes)

# Compilar con la pérdida estándar
base_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(), # Sin label_smoothing
    metrics=['accuracy']
)

# Entrenar el modelo
batch_size = 64
epochs = 25 # Un número razonable de épocas para ver los efectos

history_base = base_model.fit(
    x_train, y_train_cat,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test_cat),
    verbose=1 # Usar verbose=1 para ver el progreso
)
```
---

### **Parte 2: Modelo con Label Smoothing**
Ahora, creamos y entrenamos un segundo modelo con la misma arquitectura, pero esta vez, al compilar, activamos `label_smoothing`. Un valor común para el factor de suavizado ($\alpha$) es 0.1.

```python
# --- Entrenamiento del Modelo con Label Smoothing ---
print("\n--- Entrenando Modelo con Label Smoothing ---")

# Construir una instancia idéntica del modelo
ls_model = build_model(x_train.shape[1:], num_classes)

# Compilar con la pérdida y label_smoothing=0.1
ls_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Entrenar el modelo
history_ls = ls_model.fit(
    x_train, y_train_cat,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test_cat),
    verbose=1
)
```

---
### **Parte 3: Comparación de la Generalización (Rendimiento)**
Comparemos las curvas de precisión en el conjunto de validación para ver si Label Smoothing ayudó a la generalización.

```python
plt.figure(figsize=(10, 6))
plt.plot(history_base.history['val_accuracy'], label='Base Model (Validation)', color='blue')
plt.plot(history_ls.history['val_accuracy'], label='Label Smoothing Model (Validation)', color='red', linestyle='--')
plt.title('Comparación de Precisión en Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()
```

---
### **Parte 4: Comparación de la Confianza de las Predicciones**
Esta es la parte más reveladora. Un modelo sobreconfiado hará predicciones correctas con una probabilidad muy cercana a 1.0. Label smoothing busca "calmar" esta confianza.

Analizaremos las probabilidades de las predicciones **correctas** en el conjunto de prueba para ambos modelos.

```python
# Obtener predicciones para el conjunto de prueba
preds_base = base_model.predict(x_test)
preds_ls = ls_model.predict(x_test)

# Identificar las predicciones correctas para cada modelo
correct_preds_base_mask = np.argmax(preds_base, axis=1) == y_test.flatten()
correct_preds_ls_mask = np.argmax(preds_ls, axis=1) == y_test.flatten()

# Obtener la confianza (probabilidad máxima) de las predicciones correctas
confidence_base = np.max(preds_base[correct_preds_base_mask], axis=1)
confidence_ls = np.max(preds_ls[correct_preds_ls_mask], axis=1)

# Graficar un histograma de las confianzas
plt.figure(figsize=(12, 6))
plt.hist(confidence_base, bins=50, density=True, alpha=0.7, label='Modelo Base')
plt.hist(confidence_ls, bins=50, density=True, alpha=0.7, label='Modelo con Label Smoothing')
plt.title('Distribución de la Confianza en Predicciones Correctas')
plt.xlabel('Confianza (Probabilidad de la clase predicha)')
plt.ylabel('Densidad')
plt.xlim(0.8, 1.0) # Enfocamos el gráfico en la zona de alta confianza
plt.legend()
plt.grid(True)
plt.show()

print(f"Confianza promedio (Base Model): {np.mean(confidence_base):.4f}")
print(f"Confianza promedio (Label Smoothing Model): {np.mean(confidence_ls):.4f}")
```