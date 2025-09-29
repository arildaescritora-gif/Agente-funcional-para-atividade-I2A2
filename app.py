# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re 
# Removido plotly.express, substituído por Matplotlib/Seaborn
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
# Removido sklearn para compactação, focado em EDA

# --- Configuração da Chave de API do Google ---
try:
    google_api_key = st.secrets["google_ai"]["google_api_key"]
except KeyError:
    st.error("Chave de API do Google não encontrada. Adicione-a nos 'Secrets'.")
    st.stop()

# --- Funções Auxiliares de Carregamento ---

@st.cache_data(show_spinner=False)
def load_and_extract_data(uploaded_file):
    """Carrega dados de ZIP ou CSV, garantindo colunas minúsculas."""
    if uploaded_file is None:
        return {"status": "error", "message": "Nenhum arquivo enviado."}

    try:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
        else: # Assumindo CSV
            df = pd.read_csv(uploaded_file)
        
        # Se você voltar a ter problemas de desempenho/memória, ative esta linha:
        # df = df.sample(frac=0.1, random_state=42)

        df.columns = [col.lower() for col in df.columns]

        return {"status": "success", "df": df, "message": f"Arquivo '{uploaded_file.name}' carregado com sucesso. DataFrame pronto para análise."}

    except Exception as e:
        return {"status": "error", "message": f"Erro ao processar o arquivo: {e}"}


# --- Definição das Ferramentas Customizadas (Tools) ---

def show_descriptive_stats(*args):
    """Gera estatísticas descritivas para todas as colunas do DataFrame."""
    df = st.session_state.df
    stats = df.describe(include='all').to_markdown(index=True)
    return {"status": "success", "data": stats, "message": "Estatísticas descritivas geradas e exibidas na tabela abaixo."}


def generate_histogram(column: str, *args):
    """
    Gera um histograma Matplotlib para uma coluna numérica.
    A entrada deve ser o nome da coluna (ex: 'amount').
    """
    df = st.session_state.df
    column = column.lower()
    
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return {"status": "error", "message": f"Erro: A coluna '{column}' não é válida ou não é numérica."}
    
    # 1. Limpa figuras Matplotlib antigas
    plt.close('all')
    
    # 2. Cria a figura Matplotlib (usando plt e sns para leveza)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f'Distribuição de {column}')
    ax.set_xlabel(column)
    
    # 3. Retorna a figura para exibição
    return {"status": "success", "matplotlib_figure": fig, "message": f"O histograma da coluna '{column}' foi gerado e está sendo exibido acima."}


def generate_correlation_heatmap(*args):
    """
    Calcula a matriz de correlação e gera um mapa de calor Matplotlib/Seaborn.
    """
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        return {"status": "error", "message": "Erro: DataFrame não tem colunas numéricas suficientes para correlação."}
    
    correlation_matrix = df[numeric_cols].corr()
    
    # 1. Limpa figuras Matplotlib antigas
    plt.close('all')
    
    # 2. Cria a figura Matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        ax=ax,
        linewidths=.5,
        linecolor='black'
    )
    ax.set_title('Mapa de Calor da Matriz de Correlação')
    
    # 3. Retorna a figura para exibição
    return {"status": "success", "matplotlib_figure": fig, "message": "O mapa de calor da correlação foi gerado e está sendo exibido acima. Analise as cores para relações fortes."}


def generate_scatter_plot(columns_str: str, *args):
    """
    Gera um gráfico de dispersão Matplotlib para duas colunas numéricas.
    A entrada DEVE ser uma string contendo os nomes das duas colunas SEPARADAS por vírgula (ex: 'time, amount').
    """
    df = st.session_state.df
    
    col_names = [c.strip().lower() for c in columns_str.split(',') if c.strip()]
    
    if len(col_names) != 2:
         return {"status": "error", "message": f"Erro de Argumentos: O agente precisa de exatamente DUAS colunas separadas por vírgula."}

    x_col, y_col = col_names[0], col_names[1]

    if x_col not in df.columns or y_col not in df.columns:
        return {"status": "error", "message": f"Erro: As colunas ('{x_col}', '{y_col}') não existem no DataFrame."}
    
    # 1. Limpa figuras Matplotlib antigas
    plt.close('all')
    
    # 2. Cria a figura Matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
    ax.set_title(f'Gráfico de Dispersão: {x_col} vs {y_col}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    # 3. Retorna a figura para exibição
    return {"status": "success", "matplotlib_figure": fig, "message": f"O gráfico de dispersão para '{x_col}' vs '{y_col}' foi gerado e está sendo exibido acima."}


# --- Inicialização do Agente ---

def initialize_agent(tools_list, system_prompt_text):
    """Inicializa o Agente de Chamada de Ferramenta (Tool Calling Agent) com Gemini Flash."""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        google_api_key=google_api_key,
        temperature=0.0
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

    agent = create_tool_calling_agent(llm, tools_list, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        verbose=True,
        memory=memory,
        max_iterations=15
    )
    return agent_executor


# --- Interface do Streamlit ---

st.set_page_config(page_title="Agente EDA Customizado (Gemini Flash)", layout="wide")

st.title("🤖 Agente de Análise de Dados Customizado (V24)")
st.markdown("Agente único usando ferramentas customizadas e o modelo Gemini 1.5 Flash para análise de EDA e geração de gráficos leves (Matplotlib).")

# Inicializa o estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_
