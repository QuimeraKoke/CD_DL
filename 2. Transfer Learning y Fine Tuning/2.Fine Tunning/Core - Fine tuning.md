### **Fine-Tuning de un Modelo Pre-entrenado**

#### **Descripción**
En esta evaluación, aplicarás las técnicas de **Transfer Learning** y **Fine-Tuning** para resolver un problema de clasificación de imágenes utilizando el dataset CIFAR-10. Utilizarás un modelo de vanguardia (como EfficientNetV2B0) pre-entrenado en ImageNet como base. El proceso se dividirá en dos fases: primero, implementarás la extracción de características para establecer una línea base de rendimiento; segundo, aplicarás el fine-tuning para ajustar el modelo y mejorar los resultados.

#### **Objetivo**
El objetivo principal es demostrar la correcta implementación del flujo de trabajo de fine-tuning en dos fases. Deberás cargar un modelo pre-entrenado, entrenar un clasificador personalizado sobre la base congelada, y luego descongelar y ajustar finamente las capas superiores de la base con una tasa de aprendizaje baja para maximizar la precisión. Finalmente, deberás comparar y analizar los resultados de ambas fases.

#### **Instrucciones**

1.  **Carga y Preparación de Datos:**
    * Carga el dataset **CIFAR-10** directamente desde `tensorflow.keras.datasets`.
    * Asegúrate de que los datos tengan el tipo de dato correcto para el modelo pre-entrenado que utilizarás. No es necesario normalizar los píxeles a [0, 1] si usas un modelo como EfficientNetV2, ya que su preprocesamiento interno se encarga de ello.
    * Convierte las etiquetas a formato categórico (one-hot encoding).

2.  **Fase 1: Implementación de Extracción de Características (Línea Base):**
    * Carga la base de un modelo pre-entrenado (se recomienda **EfficientNetV2B0**), con los pesos de `imagenet` y sin la capa superior (`include_top=False`). Adapta el `input_shape` a las imágenes de CIFAR-10 (32, 32, 3).
    * **Congela** la base convolucional para que sus pesos no se actualicen durante esta primera fase (`base_model.trainable = False`).
    * Añade una nueva "cabeza" clasificadora sobre la base congelada. Esta debe consistir en una capa `GlobalAveragePooling2D`, una capa `Dropout` para regularización y una capa `Dense` final con activación `softmax` para las 10 clases de CIFAR-10.
    * Compila el modelo utilizando el optimizador `Adam` y la función de pérdida `CategoricalCrossentropy`.
    * Entrena el modelo durante **15 épocas**. Guarda el historial de entrenamiento.

3.  **Fase 2: Implementación del Fine-Tuning:**
    * Toma el modelo entrenado en la fase anterior.
    * **Descongela** la base del modelo (`base_model.trainable = True`). A continuación, congela de nuevo todas las capas excepto las **últimas 40**. Esto asegurará que solo se ajusten las capas más especializadas.
    * **Re-compila** el modelo. Este es un paso crítico: debes usar un optimizador `Adam` con una **tasa de aprendizaje muy baja** (ej. `1e-5`).
    * Continúa el entrenamiento del modelo por **10 épocas adicionales**. Asegúrate de usar el argumento `initial_epoch` en el método `.fit()` para que el historial y las gráficas continúen desde donde terminó la fase anterior.

4.  **Evaluación y Análisis Comparativo:**
    * Evalúa el rendimiento del modelo en el conjunto de prueba **después de la Fase 1** y anota la precisión. Esta será tu línea base.
    * Evalúa el rendimiento final del modelo en el conjunto de prueba **después de la Fase 2** (fine-tuning).
    * Presenta una comparación clara de la precisión final entre ambas fases y calcula la mejora obtenida.
    * Visualiza las curvas de aprendizaje (precisión y pérdida en validación) para todo el proceso (las 25 épocas totales). Utiliza una línea vertical para marcar claramente el punto donde comenzó el fine-tuning.

5.  **Discusión de Resultados:**
    * Analiza las gráficas y los resultados finales. ¿El fine-tuning mejoró el rendimiento del modelo?
    * En un breve párrafo, explica por qué fue crucial reducir la tasa de aprendizaje para la fase de fine-tuning y qué podría haber sucedido si no lo hubieras hecho.
    * Discute en qué tipo de escenario (considerando el tamaño y la naturaleza del dataset) el esfuerzo extra de implementar fine-tuning es más beneficioso.

---

---
## Laboratorio Práctico: Del Transfer Learning al Fine-Tuning Fino

### **Objetivo del Laboratorio**
* Tomar el modelo entrenado mediante **extracción de características** del laboratorio anterior.
* Aplicar el proceso de **fine-tuning** para re-entrenar las capas superiores del modelo base.
* Comparar la precisión final antes y después del fine-tuning para evaluar si la técnica mejora el rendimiento.
* Entender el flujo de trabajo completo de dos fases: extracción y luego ajuste fino.

### **Entorno**
Este laboratorio utiliza **Python**, **TensorFlow** y **Keras**. Se asume que se parte del código y los resultados del laboratorio anterior.

---
### **Parte 1: Recapitulación - Nuestra Línea Base de Extracción de Características**

Para que este laboratorio sea autocontenido, comenzaremos por ejecutar rápidamente los pasos del laboratorio anterior. Nuestro objetivo es obtener un modelo entrenado únicamente en su "cabeza" clasificadora, que servirá como nuestra línea base para la comparación.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Carga y Preparación de Datos ---
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# --- 2. Cargar y Congelar la Base ---
INPUT_SHAPE = (32, 32, 3)
base_model = keras.applications.EfficientNetV2B0(
    include_top=False,
    weights='imagenet',
    input_shape=INPUT_SHAPE
)
base_model.trainable = False # Congelamos la base

# --- 3. Construir el Modelo Completo ---
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
], name="finetuning_lab_model")

# --- 4. Compilar y Entrenar (Fase de Extracción) ---
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

print("--- FASE 1: Entrenando con Extracción de Características ---")
initial_epochs = 15
history = model.fit(
    x_train,
    y_train_cat,
    epochs=initial_epochs,
    validation_data=(x_test, y_test_cat)
)

# Guardar el rendimiento de nuestra línea base
loss_baseline, accuracy_baseline = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nPrecisión de Línea Base (Extracción de Características): {accuracy_baseline * 100:.2f}%")
```

---
### **Parte 2: El Proceso de Fine-Tuning**

Ahora que tenemos una cabeza clasificadora estable, podemos proceder con el ajuste fino de las capas superiores del modelo base.

#### **Paso 2a: Descongelar las Capas Superiores**
Vamos a "descongelar" la base para que sus pesos puedan ser actualizados. Sin embargo, no queremos re-entrenar toda la red, solo las capas más especializadas (las últimas).

```python
# Descongelamos la base
base_model.trainable = True

# Veamos cuántas capas tiene la base
print(f"La base del modelo tiene {len(base_model.layers)} capas.")

# Decidimos congelar todas las capas excepto las últimas 40
fine_tune_at = -40 # Descongelar las últimas 40 capas

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Fine-tuning se aplicará sobre las últimas {-fine_tune_at} capas del modelo base.")
```

#### **Paso 2b: Re-compilar el Modelo con una Tasa de Aprendizaje Baja**
Este es el paso **más crítico**. Para no destruir el conocimiento pre-entrenado con actualizaciones de gradiente muy grandes, debemos usar una tasa de aprendizaje (learning rate) muy pequeña.

```python
# Re-compilar el modelo con un optimizador y una tasa de aprendizaje muy baja
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), # ¡LR muy bajo!
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

model.summary()
```
**Observación:** Al ver el nuevo resumen, notarás que el número de "Trainable params" ha aumentado, pero sigue siendo mucho menor que el total de parámetros.

---
### **Parte 3: Continuar el Entrenamiento (Fase de Fine-Tuning)**
Ahora continuamos el entrenamiento desde donde lo dejamos. `initial_epoch` le dice a Keras que estamos continuando un proceso de entrenamiento anterior, lo cual es importante para las gráficas de historial.

```python
print("\n--- FASE 2: Continuando el Entrenamiento con Fine-Tuning ---")

fine_tune_epochs = 10 # Entrenar por 10 épocas adicionales
total_epochs = initial_epochs + fine_tune_epochs

# Continuar el entrenamiento
history_fine_tune = model.fit(
    x_train,
    y_train_cat,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1, # Empezar desde la última época de la fase anterior
    validation_data=(x_test, y_test_cat)
)
```

---
### **Parte 4: Comparación Final y Análisis**
Comparemos el rendimiento antes y después del fine-tuning.

```python
# Evaluar el rendimiento final después del fine-tuning
loss_finetune, accuracy_finetune = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nPrecisión después de Fine-Tuning: {accuracy_finetune * 100:.2f}%")
print(f"Mejora con Fine-Tuning: {(accuracy_finetune - accuracy_baseline) * 100:.2f}%")

# Graficar los resultados combinados
# Unimos los historiales de ambas fases de entrenamiento
acc = history.history['val_categorical_accuracy'] + history_fine_tune.history['val_categorical_accuracy']
loss = history.history['val_loss'] + history_fine_tune.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Precisión en Validación')
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Inicio del Fine-Tuning', linestyle='--')
plt.legend(loc='lower right')
plt.title('Precisión en Validación a lo largo del Entrenamiento Completo')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(loss, label='Pérdida en Validación')
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Inicio del Fine-Tuning', linestyle='--')
plt.legend(loc='upper right')
plt.title('Pérdida en Validación a lo largo del Entrenamiento Completo')
plt.xlabel('Épocas')
plt.grid(True)
plt.show()
```