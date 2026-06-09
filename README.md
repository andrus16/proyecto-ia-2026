# 🤖 Plataforma de Clasificación con Regresión Logística - IA 2026

Este proyecto consiste en una aplicación web interactiva desarrollada en **Streamlit** y **Python** para el entrenamiento, evaluación y uso práctico
de modelos de clasificación supervisada basados en Regresión Logística, cumpliendo con los objetivos del curso de Inteligencia Artificial y Aprendizaje
de Máquina.

**🔗 Enlace de la Aplicación en la Nube:** [https://proyecto-ia-2026-nimdwqlqlccgp3qyy8c9qu.streamlit.app/]

---

## 🛠️ Arquitectura y Stack Tecnológico
* **Lenguaje principal:** Python 3.x
* **Interfaz de Usuario:** Streamlit (para crear un entorno educativo interactivo, ágil y visual).
* **Modelado y Algoritmos:** Scikit-Learn (`LogisticRegression`, `train_test_split`).
* **Visualización de Datos y Métricas:** Matplotlib y Seaborn (generación de gráficos de la Matriz de Confusión).

---

## 🧠 Bitácora de Uso Responsable de IA Generativa
En cumplimiento estricto de los requerimientos de la rúbrica de evaluación, se detalla a continuación el proceso de co-creación con herramientas de IA
generativa durante el desarrollo del proyecto:

### 1. Qué fue generado:
* La lógica estructural del pipeline de datos y la segmentación interactiva de conjuntos de datos de entrenamiento y validación (Train/Test Split) mediante
un control deslizante (`st.slider`).

* El bloque matemático y el script automático que simula el conjunto de datos de negocio (Churn de Clientes y diagnostico de diabetes), permitiendo
que la aplicación cuente con ejemplos autónomos precargados.

* La representación visual de la matriz de confusión utilizando mapas de calor personalizados integrados en la interfaz de Streamlit.

* Además de los dos ejemplos precargados exigidos por la guía, la plataforma cuenta con un módulo dinámico de ingesta de archivos independientes. Para garantizar el correcto funcionamiento del pipeline de clasificación, el usuario debe cargar un archivo con las siguientes especificaciones técnicas:
  1. **Formato:** Archivo separado por comas del tipo `.csv`.
  2. **Variables independientes ($X$):** Las columnas iniciales deben contener exclusivamente datos de tipo numérico continuo o entero
  (características del modelo).
  3. **Variable objetivo ($y$):** La **última columna** del archivo debe corresponder estrictamente a la clase objetivo binaria, estructurada con valores 
  numéricos de `0` (clase negativa) o `1` (clase positiva).

### 2. Qué fue corregido:
* **Error de dependencias local (`ModuleNotFoundError: No module named 'matplotlib'`):** Al incorporar los gráficos en la app local, el sistema falló.
Se solucionó deteniendo el servidor local, instalando `matplotlib` y `seaborn` mediante la consola, y registrándolos en el archivo de configuración.

* **Error de despliegue en la nube (`ModuleNotFoundError` en Streamlit Cloud):** La aplicación web en internet falló inicialmente al leer las librerías gráficas. Tras inspeccionar los archivos, se identificó que el archivo de configuración se había guardado accidentalmente con doble extensión (`requirements.txt.txt`). Se corrigió renombrándolo a `requirements.txt` en la raíz del repositorio web y reiniciando el servidor cloud, logrando un despliegue exitoso.

### 3. Qué fue comprendido:
* Se interiorizó cómo la **Regresión Logística** adapta la ecuación lineal multivariable pasándola a través de la **Función Sigmoide** ($1 / (1 + e^{-z})$).
Esto curva los resultados para mapear cualquier valor numérico a una probabilidad acotada estrictamente entre 0 y 1, ideal para la clasificación binaria.

* Se comprendió la interpretación directa de los coeficientes: un peso con valor positivo indica que el aumento de esa característica eleva la probabilidad
de que el registro pertenezca a la Clase 1, mientras que un coeficiente negativo ejerce el efecto contrario.

* Se asimiló la importancia de evaluar un modelo usando la **Matriz de Confusión** (analizando falsos positivos y falsos negativos) en lugar de depender 
únicamente de la exactitud general (*Accuracy*).
