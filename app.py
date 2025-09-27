# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
# O DataFrame (df) é lido diretamente de st.session_state.df dentro da função.

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
    Gera um histograma para uma coluna numérica específica do DataFrame.
    A entrada deve ser o nome da coluna (ex: 'amount', 'v5', 'time').
    """
    df = st.session_state.df
    # Forçando a coluna para minúsculas
    column = column.lower()
    
    if column not in df.columns:
        return {"status": "error", "message": f"Erro: A coluna '{column}' não foi encontrada no DataFrame. Por favor, verifique se o nome está correto."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"status": "error", "message": f"Erro: A coluna '{column}' não é numérica. Forneça uma coluna numérica para gerar um histograma."}
    
    fig, ax = plt.subplots()
    df[column].hist(ax=ax)
    ax.set_title(f'Distribuição de {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequência')
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return {"status": "success", "image": buf, "message": f"O histograma da coluna '{column}' foi gerado com sucesso."}


def generate_correlation_heatmap(*args):
    """
    Calcula a matriz de correlação entre as variáveis numéricas do DataFrame
    e gera um mapa de calor (heatmap) para visualização.
    """
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        return {"status": "error", "message": "Erro: O DataFrame não tem colunas numéricas suficientes para calcular a correlação."}
    
    correlation_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Mapa de Calor da Matriz de Correlação')
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return {"status": "success", "image": buf, "message": "O mapa de calor da correlação foi gerado com sucesso."}


def generate_scatter_plot(x_col: str, y_col: str, *args):
    """
    Gera um gráfico de dispersão (scatter plot) para visualizar a relação entre duas colunas numéricas.
    As entradas DEVE ser os nomes das colunas X e Y SEPARADAMENTE (ex: x_col='time', y_col='amount').
    """
    df = st.session_state.df
    
    # CORRIGIDO: O agente falhou porque a entrada foi ambígua. A validação Pydantic precisa dos dois
    # argumentos. Vamos garantir que eles existam antes de prosseguir.
    if not x_col or not y_col:
         return {"status": "error", "message": "Erro de Argumentos: Para gerar o gráfico de dispersão, o agente precisa de DOIS nomes de coluna distintos para os eixos X e Y."}

    # Forçando as colunas para minúsculas
    x_col = x_col.lower()
    y_col = y_col.lower()
    
    if x_col not in df.columns or y_col not in df.columns:
        return {"status": "error", "message": f"Erro: Uma ou ambas as colunas ('{x_col}', '{y_col}') não existem no DataFrame."}
    
    fig, ax = plt.subplots()
    df.plot.scatter(x=x_col, y=y_col, ax=ax)
    ax.set_title(f'Gráfico de Dispersão: {x_col} vs {y_col}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return {"status": "success", "image": buf, "message": f"O gráfico de dispersão para '{x_col}' vs '{y_col}' foi gerado com sucesso."}


def detect_outliers_isolation_forest(*args):
    """
    Detecta anomalias (outliers) no DataFrame usando o algoritmo Isolation Forest.
    A análise é aplicada às colunas V1 a V28, 'time' e 'amount'.
    Retorna o número de anomalias detectadas e uma amostra dos outliers.
    """
    try:
        df = st.session_state.df
        # Nomes das colunas ajustados para minúsculas ('time', 'amount')
        feature_cols = [col for col in df.columns if col.startswith('v')] + ['time', 'amount']
        
        # Filtra apenas colunas que realmente existem no DF
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
        return {"status": "error", "message": f"Erro ao detectar anomalias: {e}"}


def find_clusters_kmeans(n_clusters: str, *args):
    """
    Realiza agrupamento (clustering) nos dados usando o algoritmo K-Means.
    A análise é aplicada às colunas V1 a V28, 'time' e 'amount'.
    A entrada DEVE ser o número de clusters desejado (como string, ex: "5").
    Retorna uma descrição dos clusters encontrados.
    """
    try:
        # Converte o input (que é uma string por causa do bug de LLM) para int
        n_clusters = int(n_clusters)
    except ValueError:
         return {"status": "error", "message": f"O número de clusters deve ser um número inteiro, mas o valor recebido foi '{n_clusters}'."}

    try:
        df = st.session_state.df
        # Nomes das colunas ajustados para minúsculas ('time', 'amount')
        feature_cols = [col for col in df.columns if col.startswith('v')] + ['time', 'amount']
        
        existing_features = [col for col in feature_cols if col in df.columns]
        if not existing_features:
             return {"status": "error", "message": "Erro ao encontrar clusters: Não foram encontradas colunas V*, 'time' ou 'amount' no DataFrame."}

        df_features = df[existing_features]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        df['cluster'] = kmeans.fit_predict(df_scaled)
        
        cluster_summary = df.groupby('cluster').agg({
            'amount': ['mean', 'min', 'max'],
            'time': ['min', 'max']
        }).to_markdown(tablefmt="pipe")
        
        message = f"O agrupamento K-Means com {n_clusters} clusters foi concluído."
        message += "\nCaracterísticas dos Clusters:\n" + cluster_summary
        
        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao realizar o agrupamento com K-Means: {e}"}


# --- Funções de Carregamento e Interface ---

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
            return {"status": "error", "message": "Formato de arquivo não suportado. Por favor, envie um arquivo ZIP ou CSV."}

        # CRÍTICO: Renomear colunas para minúsculas para padronização.
        df.columns = [col.lower() for col in df.columns]

        return {"status": "success", "df": df, "message": f"Arquivo '{uploaded_file.name}' carregado com sucesso. DataFrame pronto para análise."}

    except Exception as e:
        return {"status": "error", "message": f"Erro ao processar o arquivo: {e}"}


def initialize_agent(tools_list, system_prompt_text):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
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

st.set_page_config(page_title="Agente de Análise de Dados (Gemini/LangChain)", layout="wide")

st.title("🤖 Agente de Análise de Dados (EDA) com Gemini")
st.markdown("Envie um arquivo CSV (ou ZIP com CSV) e pergunte ao agente para realizar análises, como correlação, estatísticas descritivas ou detecção de anomalias.")

# Inicializa o estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

# Sidebar para upload de arquivo
with st.sidebar:
    st.header("Upload do Arquivo de Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV ou ZIP", type=["csv", "zip"])

    # Botão para carregar dados e inicializar o agente
    if st.button("Carregar Dados e Inicializar Agente") and uploaded_file is not None:
        with st.spinner("Carregando e preparando dados..."):
            load_result = load_and_extract_data(uploaded_file)

        if load_result["status"] == "success":
            st.session_state.df = load_result["df"]

            # 2. Definir a lista final de ferramentas
            tools_with_df = [
                Tool(
                    name=show_descriptive_stats.__name__,
                    description=show_descriptive_stats.__doc__,
                    func=show_descriptive_stats 
                ),
                Tool(
                    name=generate_histogram.__name__,
                    description=generate_histogram.__doc__,
                    func=generate_histogram
                ),
                Tool(
                    name=generate_correlation_heatmap.__name__,
                    description=generate_correlation_heatmap.__doc__,
                    func=generate_correlation_heatmap
                ),
                Tool(
                    name=generate_scatter_plot.__name__,
                    description=generate_scatter_plot.__doc__,
                    func=generate_scatter_plot
                ),
                Tool(
                    name=detect_outliers_isolation_forest.__name__,
                    description=detect_outliers_isolation_forest.__doc__,
                    func=detect_outliers_isolation_forest
                ),
                Tool(
                    name=find_clusters_kmeans.__name__,
                    description=find_clusters_kmeans.__doc__,
                    func=find_clusters_kmeans
                )
            ]

            system_prompt = (
                "Você é um agente de Análise Exploratória de Dados (EDA) altamente proficiente, "
                "especializado em datasets de transações financeiras. Seu objetivo é ajudar o usuário a "
                "entender o dataset, usando as ferramentas disponíveis para gerar estatísticas, gráficos e "
                "modelos de clustering/anomalias. "
                "Sempre que o usuário solicitar uma análise de dados, use a ferramenta apropriada. "
                "Para análises que requerem colunas (como histograma), **você deve** perguntar ao usuário qual coluna ele deseja, se ele não especificar. "
                "Ao receber o resultado de uma ferramenta (markdown, gráfico ou mensagem), sintetize a informação de forma clara e profissional. "
                "Lembre-se: todas as colunas V* e 'Time' e 'Amount' foram convertidas para minúsculas ('v*', 'time', 'amount') no DataFrame. "
                "Sua resposta final deve sempre ser em Português e oferecer insights."
            )

            st.session_state.agent_executor = initialize_agent(tools_with_df, system_prompt)
            st.success("Dados carregados e agente inicializado! Você pode começar a perguntar.")

        else:
            st.error(load_result["message"])

    if st.session_state.df is not None:
        st.success(f"DataFrame carregado com {len(st.session_state.df)} linhas e {len(st.session_state.df.columns)} colunas.")
        st.subheader("Visualização dos Dados (Amostra)")
        st.dataframe(st.session_state.df.head())


# Exibir histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], pd.DataFrame):
             st.dataframe(message["content"])
        elif isinstance(message["content"], str):
             st.markdown(message["content"])

# Tratamento de entrada do usuário
if prompt_input := st.chat_input("Qual análise você gostaria de fazer? (Ex: 'Gere um mapa de calor da correlação')"):
    
    # 1. Adicionar input do usuário ao chat
    with st.chat_message("user"):
        st.markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    
    # 2. Executar o agente
    if st.session_state.agent_executor is not None:
        with st.chat_message("assistant"):
            st_callback = st.container()
            
            try:
                full_response = st.session_state.agent_executor.invoke({"input": prompt_input})
                response_content = full_response["output"]

                if isinstance(response_content, dict) and response_content.get("status") in ["success", "error"]:
                    
                    if "message" in response_content:
                        # Exibe e salva a mensagem de texto
                        st_callback.markdown(response_content["message"])
                        st.session_state.messages.append({"role": "assistant", "content": response_content["message"]})
                    
                    if "data" in response_content:
                        # Exibe o DataFrame de estatísticas
                        df_display = pd.read_markdown(response_content["data"])
                        st_callback.dataframe(df_display)
                        # Salva o DataFrame como objeto DataFrame (para exibição futura)
                        st.session_state.messages.append({"role": "assistant", "content": df_display})
                        
                    if "image" in response_content:
                        # Exibe a imagem/gráfico no Streamlit
                        st_callback.image(response_content["image"], use_column_width=True)
                        # Não salva o objeto BytesIO na memória do chat.
                    
                    if response_content.get("status") == "error":
                         st_callback.error(response_content["message"])
                    
                else:
                    # Resposta direta do LLM (sem uso de ferramenta)
                    st_callback.markdown(str(response_content))
                    st.session_state.messages.append({"role": "assistant", "content": str(response_content)})

            except Exception as e:
                error_message = f"Desculpe, ocorreu um erro na análise: {e}"
                st_callback.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
