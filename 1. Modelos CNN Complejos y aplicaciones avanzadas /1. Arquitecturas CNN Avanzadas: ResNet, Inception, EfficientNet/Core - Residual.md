### **Instrucciones**

#### **Demostración Práctica del Problema de Degradación y las Conexiones Residuales (Evaluación)**

##### **Descripción**
En esta actividad, verificarás experimentalmente uno de los desafíos más importantes en el entrenamiento de redes profundas: el **problema de degradación**. Construirás y compararás tres modelos: una red superficial que servirá como línea base, una red "plana" muy profunda que debería mostrar problemas para entrenar, y una red residual (tipo ResNet) de profundidad similar que, gracias a las "skip connections", debería superar estos problemas.

##### **Objetivo**
El objetivo es demostrar empíricamente los conceptos teóricos del flujo de identidad y los gradientes. Deberás:
* Construir y entrenar tres arquitecturas CNN de diferente profundidad y tipo (superficial, plana profunda, residual profunda).
* Observar en la práctica cómo apilar capas de forma ingenua puede perjudicar el rendimiento (degradación), incluso en comparación con una red más pequeña.
* Demostrar con código y resultados que las conexiones residuales son una solución efectiva a este problema, permitiendo que las redes más profundas se entrenen correctamente.

##### **Instrucciones**

1.  **Carga y Preparación de Datos:**
    * Carga el dataset **CIFAR-10** desde `tensorflow.keras.datasets`.
    * Normaliza las imágenes al rango [0, 1] y convierte las etiquetas a formato categórico (one-hot encoding).

2.  **Modelo A: Construcción de la Línea Base Superficial:**
    * Implementa una CNN secuencial simple con una profundidad moderada (ej. 6 a 8 capas convolucionales). Puedes agruparlas en 3 bloques con `MaxPooling` entre ellos.
    * Compila y entrena este modelo durante un número suficiente de épocas (ej. 20-30).
    * Registra su precisión final en el conjunto de validación. **Este es el rendimiento a superar.**

3.  **Modelo B: Provocando el Problema de Degradación (Red Plana Profunda):**
    * Construye una nueva CNN secuencial que sea **significativamente más profunda** que el Modelo A (ej. 18-20 capas convolucionales), simplemente apilando más capas `Conv2D` y `BatchNormalization`. **No uses conexiones residuales aquí.**
    * Entrena este modelo utilizando exactamente los mismos hiperparámetros (optimizador, épocas, tamaño de lote) que el Modelo A para una comparación justa.

4.  **Modelo C: La Solución con Conexiones Residuales (Red Residual Profunda):**
    * Primero, implementa una función en Python que construya un **"bloque residual"** (como el que vimos en la teoría, con dos capas `Conv2D` y una skip connection que suma la entrada a la salida).
    * Construye una tercera red neuronal con una profundidad similar a la del Modelo B, pero esta vez, estructurada mediante la repetición de tus bloques residuales.
    * Entrena este modelo con los mismos hiperparámetros que los modelos anteriores.

5.  **Análisis Comparativo y Conclusiones:**
    * Crea una única gráfica que muestre las curvas de **precisión en validación** de los tres modelos a lo largo de las épocas.
    * Presenta una tabla o un resumen con la precisión final de cada modelo en el conjunto de prueba.
    * Escribe una conclusión discutiendo los resultados:
        * ¿Logró el Modelo B (plano y profundo) superar al Modelo A (superficial)? ¿Observaste el problema de degradación?
        * ¿Cómo se compara el rendimiento del Modelo C (residual) con los otros dos?
        * Explica cómo tus resultados experimentales validan la teoría de que las skip connections facilitan el entrenamiento al ayudar con el flujo de gradientes y el aprendizaje de la identidad.

##### **Sugerencias**
* Es muy recomendable usar `BatchNormalization` después de cada capa `Conv2D` (y antes de la activación `ReLU`) en los tres modelos para estabilizar el entrenamiento.
* Para el Modelo B (plano y profundo), puedes simplemente repetir un patrón de `Conv2D -> BatchNormalization -> ReLU` varias veces. El objetivo es ver si "más es mejor" de forma ingenua.
* No te preocupes si el rendimiento del Modelo B es pobre. ¡Ese es el punto del experimento! Demostrar que más profundo no es automáticamente mejor sin la arquitectura correcta.

---
## Laboratorio: Demostración Práctica del Problema de Degradación y ResNet

### **Objetivo del Laboratorio**
Verificar experimentalmente que (a) redes muy profundas "planas" entrenan peor que sus contrapartes superficiales (degradación) y (b) las conexiones residuales solucionan este problema.

---
### **Parte 0: Configuración y Datos**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Cargar y preparar el dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# Hiperparámetros de entrenamiento
EPOCHS = 30
BATCH_SIZE = 64
```

---
### **Parte 1: Modelo A - Línea Base Superficial**
Una CNN estándar y funcional que nos dará una meta de rendimiento.

```python
def build_shallow_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

model_a = build_shallow_model()
model_a.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("--- Entrenando Modelo A (Línea Base Superficial) ---")
history_a = model_a.fit(x_train, y_train_cat, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test_cat), verbose=1)
```

---
### **Parte 2: Modelo B - Red Plana Profunda (Intentando provocar degradación)**
Construimos un modelo mucho más profundo apilando capas de forma secuencial.

```python
def build_deep_plain_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Apilamos muchas capas
    for _ in range(8): # 8 * 2 = 16 capas convolucionales
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

model_b = build_deep_plain_model()
model_b.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\n--- Entrenando Modelo B (Red Plana Profunda) ---")
history_b = model_b.fit(x_train, y_train_cat, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test_cat), verbose=1)
```

---
### **Parte 3: Modelo C - Red Residual Profunda (La Solución)**
Construimos una red de profundidad similar al Modelo B, pero usando bloques residuales.

```python
# Primero, definimos nuestro bloque residual
def residual_block(x, filters):
    fx = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, 3, padding='same')(fx)
    fx = layers.BatchNormalization()(fx)
    out = layers.add([x, fx]) # La skip connection
    out = layers.ReLU()(out)
    return out

def build_deep_resnet_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Apilamos bloques residuales
    for _ in range(8):
        x = residual_block(x, filters=64)
        
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

model_c = build_deep_resnet_model()
model_c.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\n--- Entrenando Modelo C (Red Residual Profunda) ---")
history_c = model_c.fit(x_train, y_train_cat, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test_cat), verbose=1)
```
---
### **Parte 4: Análisis Comparativo de Resultados**

```python
plt.figure(figsize=(12, 8))
plt.plot(history_a.history['val_accuracy'], label='Modelo A (Superficial)', color='blue')
plt.plot(history_b.history['val_accuracy'], label='Modelo B (Plano Profundo)', color='red', linestyle='--')
plt.plot(history_c.history['val_accuracy'], label='Modelo C (Residual Profundo)', color='green')
plt.title('Comparación de Precisión en Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir resultados finales
acc_a = history_a.history['val_accuracy'][-1]
acc_b = history_b.history['val_accuracy'][-1]
acc_c = history_c.history['val_accuracy'][-1]

print(f"\n--- Resultados Finales ---")
print(f"Precisión Modelo A (Superficial): {acc_a*100:.2f}%")
print(f"Precisión Modelo B (Plano Profundo): {acc_b*100:.2f}%")
print(f"Precisión Modelo C (Residual Profundo): {acc_c*100:.2f}%")
```
**Análisis Esperado:**
Los resultados deberían mostrar que el **Modelo B (Plano Profundo)** tiene una precisión final *inferior* a la del **Modelo A (Superficial)**, demostrando el problema de degradación. Por otro lado, el **Modelo C (Residual Profundo)**, gracias a las skip connections, debería igualar o superar fácilmente la precisión del Modelo A, demostrando que es una arquitectura superior para entrenar redes profundas.