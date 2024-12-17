import streamlit as st

# ==================================================================
# Configurações
# ==================================================================
st.set_page_config(page_title = 'Home',
                  layout= 'centered')

# ==================================================================
# Barra Lateral
# ==================================================================
st.sidebar.markdown( '# Celpe-Bras 2015/2' ) 
st.sidebar.markdown( """---""" )

# ==================================================================
# Layout
# ==================================================================

st.header('Celpe-Bras 2015/2')

st.subheader('Um projeto de Visual Analytics', divider='gray')

st.markdown("""
           A partir do conhecimento obtido na disciplina de Visual Analytics, foram desenvolvidos dois dashboards. Assim, aqui você encontrará uma análise realizada a partir dos textos da Tarefa 4 do exame Celpe-Bras do semestre 2015/2

           Na aba lateral das demais páginas você encontrará filtros. Selecione as opções desejadas e veja gráficos e métricas alterarem.
           
           Em Tokens e Types você obterá insights acerca do número de tokens e types dos textos de acordo com cada nota.
           
           Em Keywords você visualizará as principais palavras encontradas nos textos de cada nota.  
"""
)
st.markdown('')
st.markdown('')
st.markdown('Dados não estão publicamente disponíveis')