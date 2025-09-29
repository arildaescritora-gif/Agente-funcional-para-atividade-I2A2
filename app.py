```python
import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from openai import OpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from langchain.tools import StructuredTool

# ===========================================================
# Configuração inicial
# ===========================================================
st.set_page_config(page_title="Agente com Gráficos", layout="wide")
st.title("🤖 Agente Inteligente com Gráficos (Plotly + Matplotlib)")

# Inicializa cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===========================================================
# Carregando o DataFrame (substitua pelo seu dataset)
# ===========================================================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(
        {
            "Time": np.random.randint(0, 150000, 1000),
            "V1": np.random.randn(1000),
            "V2": np.random.randn(1000),
            "V3": np.random.randn(1000),
            "V4": np.random.randn(1000),
            "V5": np.random.randn(1000),
            "Amount": np.random.randint(1, 500, 1000),
        }
    )

# ===========================================================
# Funções do agente
# ===========================================================
def generate_histograms(*args):
    """
    Gera histogramas para todas as colunas numéricas do dataframe (versão matplotlib).
    """
    df = st.session_state.df
    
    numeric_cols = df.select_dtypes(include=["number"]).columns
    n_cols = 5  # quantidade de gráficos por linha
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=30, color="steelblue", edgecolor="black")
        axes[i].set_title(col)
    
    # Remove subplots extras
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    return {
        "status": "success",
        "mpl_figure": fig,
        "message": "Histogramas gerados com sucesso (matplotlib)."
    }

def generate_histogram_plotly(column_name: str):
    """
    Gera histograma de uma coluna usando Plotly.
    """
    df = st.session_state.df
    if column_name not in df.columns:
        return {"status": "error", "message": f"Coluna '{column_name}' não encontrada."}

    fig = px.histogram(df, x=column_name, nbins=30, title=f"Histograma de {column_name}")

    return {
        "status": "success",
        "plotly_figure": fig,
        "message": f"Histograma da coluna '{column_name}' gerado com sucesso."
    }

# ===========================================================
# Interface do Streamlit
# ===========================================================
pergunta = st.text_input("Digite sua pergunta para o agente:")

if pergunta:
    # Exemplo simples: escolha da função baseada no texto
    if "todas" in pergunta.lower() and "coluna" in pergunta.lower():
        response_content = generate_histograms()
    else:
        # pega uma coluna aleatória só como exemplo
        col = np.random.choice(st.session_state.df.columns)
        response_content = generate_histogram_plotly(col)

    # Exibir mensagem de status
    if "message" in response_content:
        st.success(response_content["message"])

    # Exibir gráfico Plotly
    if "plotly_figure" in response_content:
        st.plotly_chart(response_content["plotly_figure"], use_container_width=True)

    # Exibir gráfico Matplotlib
    if "mpl_figure" in response_content:
        st.pyplot(response_content["mpl_figure"])
```
