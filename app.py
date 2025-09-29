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
    """Carrega dados de
