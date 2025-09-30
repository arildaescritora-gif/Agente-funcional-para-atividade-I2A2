# -*- coding: utf-8 -*-
"""Agente de Análise Exploratória de Dados com Streamlit e Gemini"""

import streamlit as st
import numpy as np
import zipfile
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

# --------------------------------------------------------------------------------------
# --- CONFIGURAÇÃO INICIAL ---
# --------------------------------------------------------------------------------------

try:
    google_api_key = st.secrets["google_ai"]["google_api_key"]
except KeyError:
    st.error("Chave de API do Google não encontrada. Configure em 'Secrets'.")
    st.stop()

# --------------------------------------------------------------------------------------
# --- FERRAMENTAS (TOOLS) ---
# --------------------------------------------------------------------------------------

def show_descriptive_stats(*args):
    df = st.session_state.df
    stats = df.describe(include='all').to_markdown(tablefmt="pipe")
    return {"status": "success", "data": stats, "message": "Estatísticas descritivas geradas."}

def generate_histogram(column: str, *args):
    df = st.session_state.df
    column = column.lower()
    if column not in df.columns:
        return {"status": "error", "message": f"Coluna '{column}' não encontrada."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"status": "error", "message": f"A coluna '{column}' não é numérica."}

    fig = px.histogram(df, x=column, title=f'Distribuição de {column}')
    return {"status": "success", "fig": fig, "type": "plotly", "message": f"Histograma da coluna '{column}' gerado."}

def generate_correlation_heatmap(*args):
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        return {"status": "error", "message": "Não há colunas numéricas suficientes."}

    correlation_matrix = df[numeric_cols].corr()
    fig = px.imshow(correlation_matrix, text_auto=".2f", aspect="auto",
                    title='Mapa de Calor da Matriz de Correlação',
                    color_continuous_scale='RdBu_r')
    fig.update_xaxes(side="top")
    return {"status": "success", "fig": fig, "type": "plotly", "message": "Mapa de calor gerado."}

def generate_scatter_plot(columns_str: str, *args):
    df = st.session_state.df
    col_names = re.split(r'[,\s]+', columns_str.lower())
    col_names = [col for col in col_names if col and col != 'e']
    if len(col_names) < 2:
        return {"status": "error", "message": "Forneça pelo menos duas colunas."}

    x_col, y_col = col_names[:2]
    if x_col not in df.columns or y_col not in df.columns:
        return {"status": "error", "message": f"Colunas '{x_col}' ou '{y_col}' não existem."}

    fig = px.scatter(df, x=x_col, y=y_col, title=f'Dispersão: {x_col} vs {y_col}')
    return {"status": "success", "fig": fig, "type": "plotly", "message": f"Dispersão '{x_col}' vs '{y_col}' gerada."}

def detect_outliers_isolation_forest(*args):
    try:
        df = st.session_state.df
        feature_cols = [col for col in df.columns if col.startswith('v')] + ['time', 'amount']
        existing_features = [col for col in feature_cols if col in df.columns]
        if not existing_features:
            return {"status": "error", "message": "Não há colunas válidas para análise."}

        df_features = df[existing_features]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_features)
        model = IsolationForest(contamination=0.01, random_state=42)
        df['anomaly_score'] = model.fit_predict(df_scaled)
        outliers = df[df['anomaly_score'] == -1]

        msg = f"Isolation Forest detectou {len(outliers)} anomalias."
        if not outliers.empty:
            msg += "\nAmostra:\n" + outliers.head().to_markdown(tablefmt="pipe")
        return {"status": "success", "message": msg}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def find_clusters_kmeans(n_clusters: str, *args):
    try:
        n_clusters = int(n_clusters)
    except ValueError:
        return {"status": "error", "message": f"O valor '{n_clusters}' não é inteiro."}

    try:
        df = st.session_state.df
        feature_cols = [col for col in df.columns if col.startswith('v')] + ['time', 'amount']
        existing_features = [col for col in feature_cols if col in df.columns]
        if not existing_features:
            return {"status": "error", "message": "Não há colunas válidas para clusterização."}

        df_features = df[existing_features]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        df['cluster'] = kmeans.fit_predict(df_scaled)

        cluster_summary = df.groupby('cluster').agg({
            'amount': ['mean', 'min', 'max'],
            'time': ['min', 'max']
        }).to_markdown(tablefmt="pipe")

        msg = f"K-Means com {n_clusters} clusters concluído.\n{cluster_summary}"
        return {"status": "success", "message": msg}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def generate_matplotlib_figure(column_x: str, column_y: str = None, chart_type: str = 'scatter', *args):
    df = st.session_state.df
    col_x = column_x.lower()
    if col_x not in df.columns:
        return {"status": "error", "message": f"Coluna '{col_x}' não encontrada."}

    fig = plt.figure(figsize=(10, 6))
    try:
        if chart_type == 'hist':
            sns.histplot(df[col_x], kde=True, ax=plt.gca())
            plt.title(f'Histograma de {col_x}')
        elif chart_type == 'scatter' and column_y:
            col_y = column_y.lower()
            if col_y not in df.columns:
                return {"status": "error", "message": f"Coluna Y '{col_y}' não encontrada."}
            sns.scatterplot(x=df[col_x], y=df[col_y], ax=plt.gca())
            plt.title(f'Dispersão: {col_x} vs {col_y}')
        else:
            return {"status": "error", "message": "Parâmetros inválidos."}

        return {"status": "success", "fig": fig, "type": "matplotlib", "message": "Gráfico Matplotlib gerado."}
    except Exception as e:
        plt.close(fig)
        return {"status": "error", "message": str(e)}

# --------------------------------------------------------------------------------------
# --- CARREGAMENTO DE DADOS ---
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_extract_data(uploaded_file):
    if uploaded_file is None:
        return {"status": "error", "message": "Nenhum arquivo enviado."}
    try:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            return {"status": "error", "message": "Formato não suportado."}

        df.columns = [col.lower() for col in df.columns]
        return {"status": "success", "df": df, "message": f"Arquivo '{uploaded_file.name}' carregado com sucesso."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def initialize_agent(tools_list, system_prompt_text):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.0
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(llm, tools_list, prompt)
    return AgentExecutor(agent=agent, tools=tools_list, verbose=True, memory=memory, max_iterations=15)

# --------------------------------------------------------------------------------------
# --- INTERFACE STREAMLIT ---
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados (EDA) com Gemini Flash")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

with st.sidebar:
    st.header("Upload do Arquivo de Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV ou ZIP", type=["csv", "zip"])
    if st.button("Carregar Dados e Inicializar Agente") and uploaded_file is not None:
        with st.spinner("Carregando..."):
            load_result = load_and_extract_data(uploaded_file)
        if load_result["status"] == "success":
            st.session_state.df = load_result["df"]
            tools_with_df = [
                Tool(name=f.__name__, description=f.__doc__, func=f) for f in [
                    show_descriptive_stats,
                    generate_histogram,
                    generate_correlation_heatmap,
                    generate_scatter_plot,
                    detect_outliers_isolation_forest,
                    find_clusters_kmeans,
                    generate_matplotlib_figure
                ]
            ]
            system_prompt = (
                "Você é um agente de EDA. Sempre use as ferramentas disponíveis. "
                "Use Plotly para gráficos interativos e Matplotlib apenas se solicitado."
            )
            st.session_state.agent_executor = initialize_agent(tools_with_df, system_prompt)
            st.success("Agente inicializado!")
        else:
            st.error(load_result["message"])
    if st.session_state.df is not None:
        st.success(f"DataFrame com {len(st.session_state.df)} linhas e {len(st.session_state.df.columns)} colunas.")
        st.dataframe(st.session_state.df.head())

# Histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"])
        else:
            st.markdown(str(message["content"]))

# Entrada do usuário
if prompt_input := st.chat_input("Qual análise você gostaria de fazer?"):
    with st.chat_message("user"):
        st.markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    if st.session_state.agent_executor is not None:
        with st.chat_message("assistant"):
            st_callback = st.container()
            try:
                full_response = st.session_state.agent_executor.invoke({"input": prompt_input})
                response_content = full_response["output"]

                if isinstance(response_content, dict) and response_content.get("status") in ["success", "error"]:
                    if "fig" in response_content:
                        if response_content.get("type") == "plotly":
                            st_callback.plotly_chart(response_content["fig"], use_container_width=True)
                        elif response_content.get("type") == "matplotlib":
                            st_callback.pyplot(response_content["fig"])
                    if "message" in response_content:
                        st_callback.markdown(response_content["message"])
                        st.session_state.messages.append({"role": "assistant", "content": response_content["message"]})
                    if "data" in response_content:
                        df_display = pd.read_markdown(response_content["data"])
                        st_callback.dataframe(df_display)
                        st.session_state.messages.append({"role": "assistant", "content": df_display})
                    if response_content.get("status") == "error":
                        st_callback.error(response_content["message"])
                else:
                    st_callback.markdown(str(response_content))
                    st.session_state.messages.append({"role": "assistant", "content": str(response_content)})
            except Exception as e:
                error_message = f"Erro inesperado: {e}"
                st_callback.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
