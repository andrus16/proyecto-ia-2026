import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuración de la página
st.set_page_config(page_title="Clasificador IA 2026", layout="wide")
st.title("🤖 Plataforma de Clasificación con Regresión Logística")
st.write("Proyecto Final - Inteligencia Artificial y Aprendizaje De Máquina")

# Asegurar que exista la carpeta datasets
if not os.path.exists("datasets"):
    os.makedirs("datasets")

# --- GENERADOR DEL DATASET CHURN ---
if not os.path.exists("datasets/churn_ejemplo.csv"):
    np.random.seed(42)
    n_muestras = 400
    antiguedad_meses = np.random.randint(1, 72, size=n_muestras)
    factura_mensual = np.random.uniform(20.0, 120.0, size=n_muestras)
    reclamos_soporte = np.random.randint(0, 6, size=n_muestras)
    
    score = (reclamos_soporte * 1.8) + (factura_mensual * 0.02) - (antiguedad_meses * 0.06)
    probabilidad = 1 / (1 + np.exp(-score))
    churn = (probabilidad > 0.5).astype(int)
    
    df_churn = pd.DataFrame({
        'Antiguedad_Meses': antiguedad_meses,
        'Factura_Mensual': np.round(factura_mensual, 2),
        'Reclamos_Soporte': reclamos_soporte,
        'Cancelo_Servicio': churn
    })
    df_churn.to_csv("datasets/churn_ejemplo.csv", index=False)

# --- INTERFAZ DE USUARIO (BARRA LATERAL) ---
st.sidebar.header("⚙️ Configuración de Datos")
opcion_datos = st.sidebar.selectbox(
    "Selecciona el conjunto de datos:",
    ("Ejemplo 2: Rotación de Clientes (Churn)", "Ejemplo 1: Diagnóstico de Diabetes", "Subir mi propio CSV")
)

df = None

# --- LÓGICA DE CARGA ---
if opcion_datos == "Ejemplo 2: Rotación de Clientes (Churn)":
    df = pd.read_csv("datasets/churn_ejemplo.csv")
    st.subheader("📊 Dataset: Rotación de Clientes / Churn")

elif opcion_datos == "Ejemplo 1: Diagnóstico de Diabetes":
    if os.path.exists("datasets/diabetes_ejemplo.csv"):
        columnas = ['Embarazos', 'Glucosa', 'PresionArterial', 'GrosorPiel', 'Insulina', 'IMC', 'PedigreeDiabetes', 'Edad', 'Resultado']
        df = pd.read_csv("datasets/diabetes_ejemplo.csv", names=columnas)
        st.subheader("📊 Dataset: Diagnóstico de Diabetes")
    else:
        st.error("Por favor, asegúrate de haber descargado 'diabetes_ejemplo.csv' en la carpeta 'datasets'.")

elif opcion_datos == "Subir mi propio CSV":
    archivo_subido = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo_subido is not None:
        df = pd.read_csv(archivo_subido)
        st.subheader("📥 Dataset Personalizado Cargado")

# --- PROCESAMIENTO Y ENTRENAMIENTO ---
if df is not None:
    st.dataframe(df.head(10))
    st.success(f"Filas detectadas: {df.shape[0]} | Columnas detectadas: {df.shape[1]}")
    
    # Separar características (X) y variable objetivo (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    st.divider()
    
    # Crear pestañas para organizar la app de forma profesional
    tab1, tab2, tab3 = st.tabs(["🧠 Entrenamiento y Coeficientes", "📈 Evaluación del Desempeño", "🔮 Realizar Predicciones"])
    
    # Inicializar el modelo en el estado de la sesión para que persista entre pestañas
    if 'modelo_entrenado' not in st.session_state:
        st.session_state.modelo_entrenado = None
        st.session_state.X_columns = list(X.columns)
    
    with tab1:
        st.header("Configuración del Entrenamiento")
        porcentaje_test = st.slider("Porcentaje de datos para Prueba (Test Split):", 10, 40, 20) / 100
        
        if st.button("🚀 Entrenar Regresión Logística"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=porcentaje_test, random_state=42)
            
            modelo = LogisticRegression()
            modelo.fit(X_train, y_train)
            
            # Guardar en sesión
            st.session_state.modelo_entrenado = modelo
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = modelo.predict(X_test)
            
            st.success("¡Modelo entrenado exitosamente!")
            
            # Mostrar Coeficientes
            st.subheader("📋 Coeficientes del Modelo ($\beta$)")
            df_coef = pd.DataFrame({
                'Característica (Variable)': X.columns,
                'Coeficiente (Peso)': modelo.coef_[0]
            })
            st.table(df_coef)
            st.write(f"**Intercepto ($\beta_0$):** {modelo.intercept_[0]:.4f}")
            st.info("💡 Un coeficiente positivo significa que al aumentar esa variable, aumenta la probabilidad de ser Clase 1. Un coeficiente negativo indica lo contrario.")

    with tab2:
        st.header("Métricas de Validación")
        if st.session_state.modelo_entrenado is not None:
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            
            exactitud = accuracy_score(y_test, y_pred)
            st.metric(label="Exactitud General (Accuracy)", value=f"{exactitud * 100:.2f}%")
            
            # Gráfico de Matriz de Confusión
            st.subheader("📊 Matriz de Confusión")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=['Clase 0 (Negativo)', 'Clase 1 (Positivo)'],
                        yticklabels=['Clase 0 (Negativo)', 'Clase 1 (Positivo)'])
            plt.ylabel('Clase Real')
            plt.xlabel('Clase Predicha')
            st.pyplot(fig)
            
            # Reporte de Clasificación detallado
            st.subheader("📝 Reporte de Clasificación Completo")
            reporte = classification_report(y_test, y_pred, output_dict=True)
            df_reporte = pd.DataFrame(reporte).transpose()
            st.dataframe(df_reporte.style.format(precision=2))
        else:
            st.warning("⚠️ Primero debes ir a la pestaña de entrenamiento y presionar el botón de entrenar.")

    with tab3:
        st.header("🔮 Clasificador en Tiempo Real")
        if st.session_state.modelo_entrenado is not None:
            st.write("Modifica los valores de las variables para simular un nuevo caso:")
            
            # Crear entradas dinámicas según las columnas del dataset cargado
            datos_nuevos = {}
            for col in st.session_state.X_columns:
                # Valores por defecto inteligentes según el nombre de la columna
                val_min = float(df[col].min())
                val_max = float(df[col].max())
                val_promedio = float(df[col].mean())
                datos_nuevos[col] = st.number_input(f"Valor para '{col}':", min_value=val_min, max_value=val_max, value=val_promedio)
            
            if st.button("🔮 Clasificar Registro"):
                # Darle formato de fila al input
                input_df = pd.DataFrame([datos_nuevos])
                
                # Predecir clase y probabilidad
                prediccion = st.session_state.modelo_entrenado.predict(input_df)[0]
                probabilidad = st.session_state.modelo_entrenado.predict_proba(input_df)[0][1]
                
                st.subheader("Resultado de la Predicción:")
                if prediccion == 1:
                    st.error(f"🔴 **Clasificado como CLASE 1** (Probabilidad: {probabilidad*100:.2f}%)")
                else:
                    st.success(f"🟢 **Clasificado como CLASE 0** (Probabilidad: {probabilidad*100:.2f}%)")
        else:
            st.warning("⚠️ Primero debes ir a la pestaña de entrenamiento y presionar el botón de entrenar.")