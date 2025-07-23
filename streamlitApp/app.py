import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# streamlit recarga el script en cada interaccion
def main():
    st.title("App de Machine Learning: Regresión Lineal y Logística")

    # 1. Subir archivo
    st.header("1. Sube tu archivo CSV")
    file = st.file_uploader("Elige un archivo CSV", type=["csv"])
    if file is not None:
        data = pd.read_csv(file)
        st.write("Vista previa de los datos:")
        st.dataframe(data.head())

        # Paso extra: Verificar y eliminar valores nulos
        st.subheader("Verificación de valores nulos")
        nulos = data.isnull().sum()
        if nulos.any():
            st.warning("Se encontraron valores nulos en las siguientes columnas:")
            st.write(nulos[nulos > 0])
            data = data.dropna()
            st.success("Filas con valores nulos eliminadas.")
        else:
            st.info("No se encontraron valores nulos en los datos.")

        # 2. Seleccion de columnas
        st.header("2. Selecciona las columnas para el modelo")
        columnas_numericas = data.select_dtypes(include=np.number).columns.tolist()
        columnas_categoricas = data.select_dtypes(exclude=np.number).columns.tolist()

        # Regresiion Lineal
        st.subheader("Regresion (Modelos)")
        modelos_reg = {
            "Lineal" : LinearRegression,
            "Ridge":Ridge,
            "Lasso" : Lasso,
            "Arbol de Decision":DecisionTreeRegressor,
            "Random Forest" : RandomForestRegressor 
        }
        modelo_elegido = st.selectbox("Selecciona el modelo de regresion",list(modelos_reg.keys()))
        
        target_reg = st.selectbox("Selecciona la columna objetivo (numérica)", columnas_numericas)
        features_reg = st.multiselect("Selecciona las columnas predictoras (numéricas)", 
                                      [col for col in columnas_numericas if col != target_reg]) #lista de compresion para no repetir variable dependiente

        if st.button("Entrenar Regresión Lineal"):
            if features_reg and target_reg:
                X = data[features_reg]
                y = data[target_reg]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = modelos_reg[modelo_elegido]()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                st.write(f"**MSE:** {mse:.2f}")
                st.write(f"**RMSE:** {rmse:.2f}")
                
                st.session_state['modelo_reg'] = model
                st.session_state['features_reg'] = features_reg
                st.session_state['X'] = X
                
                if hasattr(model, "intercept_"): #verifica si el objeto tiene un atributo dispoonible
                    st.write(f"Intercepto del modelo: {model.intercept_}")
                if hasattr(model,"coef_"):
                    st.write(f"Coeficientes del modelo o Betas {model.coef_}")

                # Gráfico de predicciones vs reales
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.set_xlabel("Valores reales")
                ax.set_ylabel("Predicciones")
                ax.set_title("Regresión Lineal: Reales vs Predichos")
                st.pyplot(fig)     
            else:
                st.warning("Selecciona al menos una columna predictora y una objetivo.")
                
        if 'modelo_reg' in st.session_state and 'features_reg' in st.session_state:
            st.subheader("Prueba el modelo con tus propios datos")
            user_input = {}
            for feature in st.session_state['features_reg']:
                valor = st.number_input(f"Ingrese un valor para {feature}", value=float(st.session_state['X'][feature].mean()))
                user_input[feature] = valor

            if st.button("Predecir con mis datos"):
                input_df = pd.DataFrame([user_input])
                prediccion = st.session_state['modelo_reg'].predict(input_df)[0]
                st.success(f"La predicción del modelo para tus datos es: {prediccion:.2f}")
        
        # Regresión Logística
        st.subheader("Regresión Logística (Clasificación)")
        if columnas_categoricas:
            target_log = st.selectbox("Selecciona la columna objetivo (categórica)", columnas_categoricas)
            features_log = st.multiselect("Selecciona las columnas predictoras (numéricas)", columnas_numericas)
            if st.button("Entrenar Regresión Logística"):
                if features_log and target_log:
                    if data[target_log].nunique() == 2:
                        X = data[features_log]
                        y = data[target_log].astype('category').cat.codes
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = LogisticRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        st.write(f"**Precisión:** {acc:.2f}")

                        # Matriz de confusión
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_xlabel("Predicción")
                        ax.set_ylabel("Real")
                        ax.set_title("Matriz de Confusión")
                        st.pyplot(fig)

                        # Guarda el modelo y las features para predicción personalizada
                        st.session_state['modelo_log'] = model
                        st.session_state['features_log'] = features_log
                        st.session_state['X_log'] = X
                        st.session_state['target_log'] = target_log
                        st.session_state['clases_log'] = list(data[target_log].astype('category').cat.categories)
                    else:
                        st.warning("La columna objetivo debe tener solo 2 clases para regresión logística.")
                else:
                    st.warning("Selecciona al menos una columna predictora y una objetivo.")

        else:
            st.info("No se encontraron columnas categóricas para clasificación.")

        # --- Bloque de predicción personalizada para clasificación ---
        if 'modelo_log' in st.session_state and 'features_log' in st.session_state:
            st.subheader("Prueba el modelo de clasificación con tus propios datos")
            user_input_log = {}
            for feature in st.session_state['features_log']:
                valor = st.number_input(
                    f"Ingrese un valor para {feature} (clasificación)",
                    value=float(st.session_state['X_log'][feature].mean()),
                    key=f"user_input_log_{feature}"
                )
                user_input_log[feature] = valor

            if st.button("Predecir clase con mis datos"):
                input_df_log = pd.DataFrame([user_input_log])
                proba = st.session_state['modelo_log'].predict_proba(input_df_log)[0]
                prediccion_log = st.session_state['modelo_log'].predict(input_df_log)[0]  # me da la prediccion        
                clase_predicha = st.session_state['clases_log'][prediccion_log]
                porcentaje = proba[prediccion_log] * 100
                st.success(f"La clase predicha para tus datos es: {clase_predicha} (código: {prediccion_log})")
                st.info(f"Probabilidad asignada a esta clase: {porcentaje:.2f}%")

if __name__ == "__main__":
    main()