import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuración de la página
st.set_page_config(page_title="Plataforma de Clasificación - Regresión Logística", layout="wide")
st.title("🤖 Plataforma de Clasificación con Regresión Logística")

# --- BARRA LATERAL: CONFIGURACIÓN DE DATOS ---
st.sidebar.header("⚙️ Configuración de Datos")
opcion_data = st.sidebar.selectbox(
    "Selecciona el conjunto de datos:",
    ["Ejemplo 2: Rotación de Clientes (Churn)", "Subir mi propio CSV"]
)

df = None

# Opción 1: Dataset Simulado Autónomo (Churn de Clientes)
if "Ejemplo 2" in opcion_data:
    np.random.seed(42)
    n_samples = 400
    edad = np.random.randint(18, 70, n_samples)
    ingresos = np.random.randint(15, 120, n_samples) * 1000
    reclamos = np.random.randint(0, 10, n_samples)
    antiguedad = np.random.randint(1, 60, n_samples)
    
    # Lógica para definir la clase objetivo (0 o 1)
    score = 0.05 * edad - 0.00002 * ingresos + 0.6 * reclamos - 0.04 * antiguedad
    prob = 1 / (1 + np.exp(-score))
    churn = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'Edad': edad,
        'Ingresos_Anuales': ingresos,
        'Reclamos_Soporte': reclamos,
        'Antiguedad_Meses': antiguedad,
        'Churn': churn
    })
    st.sidebar.success("✅ Cargado: Ejemplo 2 (Churn de Clientes)")

# Opción 2: Cargar archivo propio con validación robusta de encabezados
else:
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            # Inspeccionamos la primera fila para saber si tiene títulos o son números directos
            first_row = pd.read_csv(uploaded_file, nrows=1, header=None)
            uploaded_file.seek(0)
            
            # Intentamos convertir el primer elemento a número
            float(first_row.iloc[0, 0])
            
            # Si no falla, significa que NO tiene encabezados. Asignamos nombres automáticos.
            df_raw = pd.read_csv(uploaded_file, header=None)
            num_cols = df_raw.shape[1]
            columnas_nuevas = [f"Variable_{i+1}" if i < num_cols - 1 else "Clase_Objetivo" for i in range(num_cols)]
            df_raw.columns = columnas_nuevas
            df = df_raw.copy()
            st.sidebar.info("💡 CSV sin encabezados detectado. Columnas nombradas automáticamente.")
        except ValueError:
            # Si falla la conversión a float, la primera fila es texto (tiene encabezados normales)
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ CSV con encabezados cargado exitosamente.")
    else:
        st.info("👈 Por favor, sube un archivo CSV en la barra lateral para comenzar.")

# --- PROCESAMIENTO Y MODELADO (Si los datos existen) ---
if df is not None:
    # Mostrar vista previa de los datos cargados
    st.subheader("📋 Vista Previa de los Datos")
    st.datasource = st.dataframe(df.head(10))
    
    # Definición de X (características) e y (clase objetivo)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Control deslizante interactivo para el Train/Test Split
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Parámetros del Modelo")
    test_size_percentage = st.sidebar.slider("Porcentaje de datos de prueba (Test Split):", 10, 50, 20, step=5)
    test_size = test_size_percentage / 100.0
    
    # Separación de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Entrenamiento del modelo de Regresión Logística
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # --- INTERFAZ DE PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs(["📊 Entrenamiento y Coeficientes", "📈 Evaluación del Desempeño", "🔮 Realizar Predicciones"])
    
    # PESTAÑA 1: Coeficientes del modelo ajustado
    with tab1:
        st.subheader("🧮 Coeficientes e Intercepto Calculados")
        st.write("A continuación se muestran los pesos asignados por el modelo a cada variable matemática:")
        
        coef_df = pd.DataFrame({
            'Característica (Variable)': X.columns,
            'Coeficiente (Peso)': model.coef_[0]
        })
        st.table(coef_df)
        st.metric(label="Intercepto (Beta_0)", value=f"{model.intercept_[0]:.4f}")
        st.info("💡 Un coeficiente positivo incrementa la probabilidad de pertenecer a la Clase 1. Un coeficiente negativo la disminuye.")
        
    # PESTAÑA 2: Métricas de rendimiento y gráficos
    with tab2:
        st.subheader("🎯 Métricas de Validación")
        acc = accuracy_score(y_test, y_pred)
        st.metric(label="Exactitud General (Accuracy)", value=f"{acc * 100:.2f}%")
        
        # Reporte de Clasificación en formato texto estructurado
        st.markdown("**📋 Reporte Detallado de Clasificación:**")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        st.json(report_dict)
        
        # Gráfico dinámico de la Matriz de Confusión
        st.markdown("**🧩 Matriz de Confusión Visual:**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicho 0', 'Predicho 1'],
                    yticklabels=['Real 0', 'Real 1'], ax=ax)
        plt.ylabel('Realidad')
        plt.xlabel('Predicción')
        st.pyplot(fig)

    # PESTAÑA 3: Formulario interactivo para nuevas predicciones independientes
    with tab3:
        st.subheader("🔮 Ingreso de Datos en Tiempo Real")
        st.write("Modifica los siguientes valores numéricos para evaluar cómo respondería el modelo ante un nuevo registro:")
        
        inputs = {}
        # Recorremos dinámicamente las variables usando el dataframe original
        for col in X.columns:
            val_min = float(df[col].min())
            val_max = float(df[col].max())
            val_mean = float(df[col].mean())
            
            # Creamos un campo interactivo adaptado al rango real de cada variable
            inputs[col] = st.number_input(f"Valor para '{col}':", min_value=val_min, max_value=val_max, value=val_mean)
            
        # Conversión de las entradas a matriz para Scikit-Learn
        input_data = np.array([list(inputs.values())])
        
        if st.button("🚀 Ejecutar Clasificación"):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            st.markdown("---")
            st.subheader("🎯 Resultado del Diagnóstico Predictivo")
            if prediction == 1:
                st.error(f"🔴 Clasificación Asignada: **Clase 1** (Probabilidad calculada: {probability * 100:.2f}%)")
            else:
                st.success(f"🟢 Clasificación Asignada: **Clase 0** (Probabilidad calculada: {(1 - probability) * 100:.2f}%)")
