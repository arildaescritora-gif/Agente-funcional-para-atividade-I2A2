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


def generate_scatter_plot(columns_str: str, *args):
    """
    Gera um gráfico de dispersão (scatter plot) interativo Plotly para visualizar 
    a relação entre duas colunas numéricas.
    A entrada DEVE ser uma string contendo os nomes das duas colunas SEPARADAS por um espaço, 
    vírgula ou 'e' (ex: 'time, amount' ou 'v1 e v2').
    """
    df = st.session_state.df
    
    col_names = re.split(r'[,\s]+', columns_str.lower())
    col_names = [col for col in col_names if col and col != 'e'] 
    
    if len(col_names) < 2:
         return {"status": "error", "message": f"Erro de Argumentos: O agente precisa de pelo menos DOIS nomes de coluna para o gráfico de dispersão. Foi encontrado apenas: {col_names}"}

    x_col = col_names[0]
    y_col = col_names[1]

    if x_col not in df.columns or y_col not in df.columns:
        return {"status": "error", "message": f"Erro: Uma ou ambas as colunas ('{x_col}', '{y_col}') não existem no DataFrame."}
    
    # Usando Plotly Express
    fig = px.scatter(df, x=x_col, y=y_col, title=f'Gráfico de Dispersão: {x_col} vs {y_col}')
    return {"status": "success", "plotly_figure": fig, "message": f"O gráfico de dispersão interativo para '{x_col}' vs '{y_col}' foi gerado. Use-o para visualizar a forma e a densidade da relação entre essas variáveis."}


def detect_outliers_isolation_forest(*args):
    """
    Detecta anomalias (outliers) no DataFrame usando o algoritmo Isolation Forest.
    A análise é aplicada às colunas V1 a V28, 'time' e 'amount'.
    Retorna o número de anomalias detectadas e uma amostra dos outliers.
    """
    try:
        df = st.session_state.df
        feature_cols = [col for col in df.columns if col.startswith('v')] + ['time', 'amount']
        
        existing_features = [col for col in feature_cols if col in df.columns]
        if not existing_features:
             return {"status": "error", "message": "Erro ao detectar anomalias: Não foram encontradas colunas V*, 'time' ou 'amount' no DataFrame."}

        df_features = df[existing_features]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_features)
        model = IsolationForest(contamination=0.01, random_state=42)
        df['anomaly_score'] = model.fit_predict(df_scaled)
        outliers = df[df['anomaly_score'] == -1]
        
        message = f"O algoritmo Isolation Forest detectou {len(outliers)} transações atípicas (outliers)."
        if not outliers.empty:
            message += "\nAmostra das transações detectadas como anomalias:\n" + outliers.head().to_markdown(tablefmt="pipe")
            
        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao detectar an
