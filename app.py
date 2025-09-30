# ... [código anterior omitido]

# Tratamento de entrada do usuário
if prompt_input := st.chat_input("Qual análise você gostaria de fazer? (Ex: 'Gere um mapa de calor da correlação' ou 'Gere um histograma Matplotlib da coluna amount')"):
    
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
                    
                    # RENDERIZAÇÃO DE GRÁFICO PLOTLY (AGORA USANDO st.plotly_chart)
                    if "plotly_figure" in response_content:
                        # CORREÇÃO AQUI: Usando a função dedicada do Streamlit para Plotly
                        st_callback.plotly_chart(response_content["plotly_figure"], use_container_width=True)
                        
                    # RENDERIZAÇÃO DE GRÁFICO MATPLOTLIB 
                    if "matplotlib_figure" in response_content:
                        st_callback.pyplot(response_content["matplotlib_figure"])
                        plt.close(response_content["matplotlib_figure"]) 
                    
# ... [resto do código omitido]
