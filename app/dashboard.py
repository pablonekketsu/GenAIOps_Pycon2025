# app/dashboard.py

import mlflow
import pandas as pd
import streamlit as st

st.set_page_config(page_title="📊 Dashboard General de Evaluación", layout="wide")
st.title("📈 Evaluación Completa del Chatbot por Pregunta")

# ✅ Buscar todos los experimentos que comienzan con "eval_"
client = mlflow.tracking.MlflowClient()
experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

if not experiments:
    st.warning("No se encontraron experimentos de evaluación.")
    st.stop()

# Mostrar opciones
exp_names = [exp.name for exp in experiments]
selected_exp_name = st.selectbox("Selecciona un experimento para visualizar:", exp_names)

experiment = client.get_experiment_by_name(selected_exp_name)
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

if not runs:
    st.warning("No hay ejecuciones registradas en este experimento.")
    st.stop()

# Convertir runs a DataFrame
data = []
for run in runs:
    params = run.data.params
    metrics = run.data.metrics
    data.append({
        "pregunta": params.get("question"),
        "prompt_version": params.get("prompt_version"),
        "chunk_size": int(params.get("chunk_size", 0)),
        "chunk_overlap": int(params.get("chunk_overlap", 0)),
        "correctness_score": metrics.get("correctness_score", metrics.get("lc_is_correct", 0)),
        "relevance_score": metrics.get("relevance_score", 0),
        "coherence_score": metrics.get("coherence_score", 0),
        "toxicity_score": metrics.get("toxicity_score", 0),
        "harmfulness_score": metrics.get("harmfulness_score", 0),
    })

df = pd.DataFrame(data)

# Mostrar tabla completa
st.subheader("📋 Resultados individuales por pregunta")
st.dataframe(df)

# Agrupación para análisis
criteria_columns = [
    "correctness_score",
    "relevance_score",
    "coherence_score",
    "toxicity_score",
    "harmfulness_score",
]
grouped = df.groupby(["prompt_version", "chunk_size"])[criteria_columns].mean().reset_index()
grouped["preguntas"] = df.groupby(["prompt_version", "chunk_size"]).size().values

st.subheader("📊 Desempeño agrupado por configuración")
st.dataframe(grouped)

# Gráfico
grouped["config"] = grouped["prompt_version"] + " | " + grouped["chunk_size"].astype(str)
selected_criterion = st.selectbox("Criterio para el gráfico:", criteria_columns)
st.bar_chart(grouped.set_index("config")[selected_criterion])
