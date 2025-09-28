# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re 
import plotly.express as px 
# -------------------- CORREÇÃO DO NAMERROR: CLASSE LONGA --------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate 

# --- Configuração da Chave de API do Google ---
try:
    google_api_key = st.secrets["google_ai"]["google_api_key"]
except KeyError:
    st.error("Chave de API do Google não encontrada. Certifique-se de adicioná-la nos 'Secrets' da sua aplicação.")
    st.stop()

# --- Definição das Ferramentas (Tools) ---

def show_descriptive_stats(*args):
    """
    Gera estatísticas descritivas para todas as colunas de um DataFrame.
    Retorna um dicionário com o resumo estatístico.
    """
    df = st.session_state.df
    stats = df.describe(include='all').to_markdown(tablefmt="pipe")
    return {"status": "success", "data": stats, "message": "Estatísticas descritivas geradas."}


def generate_histogram(column: str, *args):
    """
    Gera um histograma interativo Plotly para uma coluna numérica específica do DataFrame.
    A entrada deve ser o nome da coluna (ex: 'amount', 'v5', 'time').
    """
    df = st.session_state.df
    column = column.lower()
    
    if column not in df.columns:
        return {"status": "error", "message": f"Erro: A coluna '{column}' não foi encontrada no DataFrame. Por favor, verifique se o nome está correto."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"status": "error", "message": f"Erro: A coluna '{column}' não é numérica. Forneça uma coluna numérica para gerar um histograma."}
    
    # Usando Plotly Express
    fig = px.histogram(df, x=column, title=f'Distribuição de {column}')
    return {"status": "success", "plotly_figure": fig, "message": f"O histograma da coluna '{column}' foi gerado com sucesso. Analise a distribuição dos dados e procure por assimetrias ou picos."}


def generate_correlation_heatmap(*args):
    """
    Calcula a matriz de correlação entre as variáveis numéricas do DataFrame
    e gera um mapa de calor (heatmap) interativo Plotly.
    """
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        return {"status": "error", "message": "Erro: O DataFrame não tem colunas numéricas suficientes para calcular a correlação."}
    
    correlation_matrix = df[numeric_cols].corr()
    
    # Usando Plotly Express
    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        title='Mapa de Calor da Matriz de Correlação',
        color_continuous_scale='RdBu_r'
    )
    fig.update_xaxes(side="top")
    return {"status": "success", "plotly_figure": fig, "message": "O mapa de calor da correlação interativo foi gerado. Analise o padrão de cores para identificar relações fortes (vermelho/azul escuro) ou fracas (cinza claro)."}


def
