# Proyecto de Machine Learning – Predicción de Cumplimiento de Pago

## Propósito del proyecto
Este proyecto tiene como finalidad desarrollar un modelo de **Machine Learning** que permita estimar si un cliente cumplirá con el pago oportuno de un crédito, a partir de variables financieras, historial crediticio y características sociodemográficas.

La solución fue construida bajo un enfoque modular, aplicando buenas prácticas de ciencia de datos que abarcan desde la ingestión de la información hasta el entrenamiento, evaluación y elección del modelo con mejor rendimiento.

---

## Descripción de los componentes

### Ingesta de datos (`cargar_datos.py`)
- Lectura de la base de datos en formato Excel.  
- Estandarización de los nombres de las columnas.  
- Procesamiento y conversión de variables de tipo fecha.  
- Verificación de la presencia de la variable objetivo `Pago_atiempo`.

---

### Ingeniería de características (`ft_engineering.py`)
- Identificación y separación de variables numéricas y categóricas.  
- Construcción de un `ColumnTransformer` que incluye:
  - Imputación de valores faltantes.
  - Escalamiento de características numéricas.
  - Codificación de variables categóricas mediante One-Hot Encoding.
- Manejo preventivo de errores comunes como:
  - Columnas ausentes en el dataset.
  - Valores infinitos o fuera de rango.

---

### Entrenamiento y evaluación de modelos (`model_training_evaluation.py`)
- División del conjunto de datos en entrenamiento y prueba.  
- Entrenamiento de distintos algoritmos de clasificación, entre ellos:
  - Regresión Logística  
  - Random Forest  
  - Gradient Boosting  
- Evaluación comparativa utilizando métricas como:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
- Selección del modelo con mejor desempeño global.  
- Almacenamiento del modelo final entrenado en formato `.pkl`.

---

## Variable objetivo
- **Pago_atiempo**
  - `1` → El cliente realiza el pago dentro del plazo establecido.  
  - `0` → El cliente no cumple con el pago a tiempo.

---

## Herramientas y tecnologías empleadas
- **Python** - V3.10
- **Pandas** – Procesamiento y análisis de datos  
- **NumPy** – Cálculo numérico  
- **Scikit-learn** – Construcción de modelos, pipelines y evaluación  
- **Joblib** – Persistencia del modelo  
- **Excel** – Fuente de datos  

---

## Resultado obtenido
El proyecto entrega:

- Un modelo de Machine Learning entrenado y validado, listo para realizar predicciones.  
- Un archivo `.pkl` preparado para su integración en una aplicación desarrollada con **Streamlit**.  
- Un flujo de trabajo estructurado, reproducible y alineado con las buenas prácticas de Machine Learning.