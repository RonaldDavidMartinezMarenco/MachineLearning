import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

        # 2. Selección de columnas
        st.header("2. Selecciona las columnas para el modelo")
        columnas_numericas = data.select_dtypes(include=np.number).columns.tolist()
        columnas_categoricas = data.select_dtypes(exclude=np.number).columns.tolist()

        # Regresión Lineal
        st.subheader("Regresión Lineal")
        target_reg = st.selectbox("Selecciona la columna objetivo (numérica)", columnas_numericas)
        features_reg = st.multiselect("Selecciona las columnas predictoras (numéricas)", 
                                      [col for col in columnas_numericas if col != target_reg]) #lista de compresion

        if st.button("Entrenar Regresión Lineal"):
            if features_reg and target_reg:
                X = data[features_reg]
                y = data[target_reg]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                st.write(f"**MSE:** {mse:.2f}")
                st.write(f"**RMSE:** {rmse:.2f}")

                # Gráfico de predicciones vs reales
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.set_xlabel("Valores reales")
                ax.set_ylabel("Predicciones")
                ax.set_title("Regresión Lineal: Reales vs Predichos")
                st.pyplot(fig)
            else:
                st.warning("Selecciona al menos una columna predictora y una objetivo.")

        # Regresión Logística
        st.subheader("Regresión Logística (Clasificación)")
        if columnas_categoricas:
            target_log = st.selectbox("Selecciona la columna objetivo (categórica)", columnas_categoricas)
            features_log = st.multiselect("Selecciona las columnas predictoras (numéricas)", columnas_numericas)
            if st.button("Entrenar Regresión Logística"):
                if features_log and target_log:
                    # Solo funciona para clasificación binaria
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
                    else:
                        st.warning("La columna objetivo debe tener solo 2 clases para regresión logística.")
                else:
                    st.warning("Selecciona al menos una columna predictora y una objetivo.")
        else:
            st.info("No se encontraron columnas categóricas para clasificación.")

if __name__ == "__main__":
    main()