import pandas as pd
import numpy as np
import nltk
from collections import Counter #uso para n-gramas
import re # talvez regex vai ajudar na normalizaçao
import plotly.graph_objects as go
import streamlit as st

# ==================================================================
# Configurações da Página 
# ==================================================================
st.set_page_config(page_title = 'Frequência das Palavras',
                  layout= 'centered')

# ==================================================================
#Import dataset
# ==================================================================
csv_file_path = 'corpus_completo.csv'

# Lendo o csv como um df
df = pd.read_csv(csv_file_path)

#Criando uma cópia
df_va = df.copy()

# ==================================================================
# Pré-Processamento
# ==================================================================
df_va.groupby('nota').count()

nltk.download('stopwords')
nltk.download('punkt') # é um tokenizador, importante para nltk.word_tokenize
# nltk.download('rslp') é um stemmer. acho q nao vou usar
nltk.download('punkt_tab') # Precisei que devia fazer download desse pacote pq o punkt nao tava funcionando

# Pré-processamento

# Devo tirar acentos?? Ex: Belém
# Retirar números? Na duvida, retirei.
# Não fiz stemming visto que poderia perder erros ortográficos

#Retirada de sinais gráficos, pontuações e espaços
def clean_cp(text):
    cleaned = text.lower() #Deixando tudo minúsculo
    cleaned = re.sub('[^\w\s]', '', cleaned) # Removendo pontuacao
    #cleaned = re.sub('[0-9]+', '', cleaned) # Removendo números 
    cleaned = re.sub('\d+', '', cleaned) # Removendo números NÃO TÁ FUNCIONANDO AAAAAAAAAAAAAAAAA Começou a funcionar qdo pus o lower como primeiro comando da funçao
    cleaned = re.sub('\s+', ' ', cleaned) # Removendo espaços extras
    cleaned = re.sub('\s+', ' ', cleaned)
    return cleaned.strip() # Removendo tabs

#Retirada de stopwords.. Essa parte deve ser importante na hora de ver as palavras mais frequentes (heatmap)
#def stopwords_cp(text):
  #  stopwords = nltk.corpus.stopwords.words('portuguese')
   # tokenized = nltk.word_tokenize(text, language='portuguese')
   # sem_stopwords = [token for token in tokenized if token not in stopwords]
   # return ' '.join(sem_stopwords)

df_va['content'] = df_va['content'].apply(clean_cp)
#df_va['content'] = df_va['content'].apply(stopwords_cp)

# Tokenizando e retirando stopwords: importante para ver a frequencia das palavras

def tokenized_cp(text):
   stopwords = nltk.corpus.stopwords.words('portuguese') # Carregando as stopwords do português
   tokenized = nltk.word_tokenize(text, language='portuguese') #Transforma o texto em tokens
   text_sem_stopwords = [token for token in tokenized if token not in stopwords] # Deixando somente o que nao é stopword no texto
   return text_sem_stopwords

df_va['tokenized_content'] = df_va['content'].apply(tokenized_cp)

# Contando o número de tokens SEM STOPWORDS (diferente do typestokenttr.ipynb) para cada texto: usarei isso para o gráfico 2, que demonstra o número mínimo e número máximo de tokens

# Contando tokens sem considerar a nota
df_va['num_tokens'] = df_va['tokenized_content'].apply(len)

# Agrupando de acordo com cada nota
qtde_tokens_nota = df_va.groupby('nota')['num_tokens'].sum()

# Verificando a ocorrência dos tokens

# Criando uma lista só com todos os tokens (?)
all_tokens = [token for tokens in df_va['tokenized_content'] for token in tokens]

# Contando a ocorrencia de cada token
token_counts = Counter(all_tokens)

# Agrupando a contagem de tokens de acordo com cada nota
nota_token_counts = (
    df_va.groupby('nota')['tokenized_content']
    .apply(lambda texts: Counter([token for text in texts for token in text]))
)

# Convertendo cada token e contagem em um dataframe
df_frequency = nota_token_counts.reset_index()
df_frequency.columns = ['nota', 'token', 'token_frequency']
sorted_tokens = df_frequency.sort_values('token_frequency', ascending=False)

df_words = pd.DataFrame()

# Agrupando a contagem de tokens por nota de forma a mostrar os 15 mais comuns para cada nota ( 15 * 6 = 90 linhas portanto )
top_tokens_per_grade = (
    sorted_tokens.groupby('nota')
    .head(20)  # Select the top 15 tokens per grade
    .reset_index(drop=True)
)
df_words = top_tokens_per_grade

df_words_sorted = (df_words.sort_values(by=['nota', 'token_frequency'], ascending=[True, False]))
df_words_0 = df_words_sorted.loc[df_words_sorted['nota'] == 0, :]
df_words_1 = df_words_sorted.loc[df_words_sorted['nota'] == 1, :]
df_words_2 = df_words_sorted.loc[df_words_sorted['nota'] == 2, :]
df_words_3 = df_words_sorted.loc[df_words_sorted['nota'] == 3, :]
df_words_4 = df_words_sorted.loc[df_words_sorted['nota'] == 4, :]
df_words_5 = df_words_sorted.loc[df_words_sorted['nota'] == 5, :]

# ==================================================================
# Barra Lateral no Streamlit 
# ==================================================================
st.sidebar.markdown( '## Filtro' )

lista_notas = list(df_words_sorted['nota'].unique())
opcao_notas = st.sidebar.multiselect(
    'Escolha a nota que deseja visualizar',
    lista_notas,
    default = [0, 1, 2, 3, 4, 5] #nao sei se fiz certo...
)
nota_selec = df_words_sorted['nota'].isin(opcao_notas)
df_words_sorted = df_words_sorted.loc[nota_selec, :]

# ==================================================================
# Layout no Streamlit
# ==================================================================
st.header( 'Frequência das palavras' )

with st.container():
    st.subheader( 'Heatmap para cada nota' )
    col1, col2, col3 = st.columns( 3,  gap='large' )

    with col1:

        # Pivot the data for heatmap
        heatmap_data = df_words_0.pivot(index='token', columns='nota', values='token_frequency').fillna(0)

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,  # Frequency matrix
            x=heatmap_data.columns,  # Grades (x-axis)
            y=heatmap_data.index,    # Tokens (y-axis)
            colorscale='Reds',       # Heatmap color scale
            #colorbar=dict(title="Frequência"),  # Colorbar title
            hoverongaps=False,       # Ensure no gaps show on hover
            hovertemplate=(
                "Token: %{y}<br>" + #Exibir título pelo Streamlit, q aí fica título pra cada coluna
                "Frequência: %{z}<extra></extra>"
            )
        ))

        # Update layout for better visualization
        fig.update_layout(
        # title="20 Tokens Mais Frequentes", >> Por título no streamlit
            height=500,  # Adjust height to make the heatmaps narrow
            width=300,  # Adjust width for each heatmap ....  * len(grades)
            template="plotly_white",
            plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
            paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
            font=dict(color="white")  # White font for all text
        )
        #fig.update_yaxes(title="Tokens", automargin=True)
        fig.update_xaxes(title="", showticklabels=False)

        # Show the figure
        fig.show()
        st.plotly_chart( fig )

    with col2:

        # Pivot the data for heatmap
        heatmap_data = df_words_1.pivot(index='token', columns='nota', values='token_frequency').fillna(0)

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,  # Frequency matrix
            x=heatmap_data.columns,  # Grades (x-axis)
            y=heatmap_data.index,    # Tokens (y-axis)
            colorscale='Reds',       # Heatmap color scale
            #colorbar=dict(title="Frequência"),  # Colorbar title
            hoverongaps=False,       # Ensure no gaps show on hover
            hovertemplate=(
                "Token: %{y}<br>" + #Exibir título pelo Streamlit, q aí fica título pra cada coluna
                "Frequência: %{z}<extra></extra>"
            )
        ))
        st.plotly_chart( fig )

        # Update layout for better visualization
        fig.update_layout(
        # title="20 Tokens Mais Frequentes", >> Por título no streamlit
            height=500,  # Adjust height to make the heatmaps narrow
            width=300,  # Adjust width for each heatmap ....  * len(grades)
            template="plotly_white",
            plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
            paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
            font=dict(color="white")  # White font for all text
        )
        #fig.update_yaxes(title="Tokens", automargin=True)
        fig.update_xaxes(title="", showticklabels=False)

        # Show the figure
        fig.show()
        st.plotly_chart( fig )


    with col3:

        # Pivot the data for heatmap
        heatmap_data = df_words_2.pivot(index='token', columns='nota', values='token_frequency').fillna(0)

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,  # Frequency matrix
            x=heatmap_data.columns,  # Grades (x-axis)
            y=heatmap_data.index,    # Tokens (y-axis)
            colorscale='Reds',       # Heatmap color scale
            #colorbar=dict(title="Frequência"),  # Colorbar title
            hoverongaps=False,       # Ensure no gaps show on hover
            hovertemplate=(
                "Token: %{y}<br>" + #Exibir título pelo Streamlit, q aí fica título pra cada coluna
                "Frequência: %{z}<extra></extra>"
            )
        ))

        # Update layout for better visualization
        fig.update_layout(
        # title="20 Tokens Mais Frequentes", >> Por título no streamlit
            height=500,  # Adjust height to make the heatmaps narrow
            width=300,  # Adjust width for each heatmap ....  * len(grades)
            template="plotly_white",
            plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
            paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
            font=dict(color="white")  # White font for all text
        )
        #fig.update_yaxes(title="Tokens", automargin=True)
        fig.update_xaxes(title="", showticklabels=False)

        # Show the figure
        fig.show()
        st.plotly_chart( fig )

with st.container():
    st.subheader( 'Heatmap para cada nota' )
    col1, col2, col3 = st.columns( 3,  gap='large' )
    with col1:

        # Pivot the data for heatmap
        heatmap_data = df_words_3.pivot(index='token', columns='nota', values='token_frequency').fillna(0)

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,  # Frequency matrix
            x=heatmap_data.columns,  # Grades (x-axis)
            y=heatmap_data.index,    # Tokens (y-axis)
            colorscale='Reds',       # Heatmap color scale
            #colorbar=dict(title="Frequência"),  # Colorbar title
            hoverongaps=False,       # Ensure no gaps show on hover
            hovertemplate=(
                "Token: %{y}<br>" + #Exibir título pelo Streamlit, q aí fica título pra cada coluna
                "Frequência: %{z}<extra></extra>"
            )
        ))

        # Update layout for better visualization
        fig.update_layout(
            # title="20 Tokens Mais Frequentes", >> Por título no streamlit
            height=500,  # Adjust height to make the heatmaps narrow
        width=300,  # Adjust width for each heatmap ....  * len(grades)
        template="plotly_white",
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
        font=dict(color="white")  # White font for all text
        )
        #fig.update_yaxes(title="Tokens", automargin=True)
        fig.update_xaxes(title="", showticklabels=False)

        # Show the figure
        fig.show()

    with col2:

        # Pivot the data for heatmap
        heatmap_data = df_words_4.pivot(index='token', columns='nota', values='token_frequency').fillna(0)

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,  # Frequency matrix
            x=heatmap_data.columns,  # Grades (x-axis)
            y=heatmap_data.index,    # Tokens (y-axis)
            colorscale='Reds',       # Heatmap color scale
            #colorbar=dict(title="Frequência"),  # Colorbar title
            hoverongaps=False,       # Ensure no gaps show on hover
            hovertemplate=(
                "Token: %{y}<br>" + #Exibir título pelo Streamlit, q aí fica título pra cada coluna
                "Frequência: %{z}<extra></extra>"
            )
        ))

        # Update layout for better visualization
        fig.update_layout(
            # title="20 Tokens Mais Frequentes", >> Por título no streamlit
            height=500,  # Adjust height to make the heatmaps narrow
            width=300,  # Adjust width for each heatmap ....  * len(grades)
            template="plotly_white",
            plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
            paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
            font=dict(color="white")  # White font for all text
        )
        #fig.update_yaxes(title="Tokens", automargin=True)
        fig.update_xaxes(title="", showticklabels=False)

        # Show the figure
        fig.show()
        st.plotly_chart( fig )

    with col3:
        # Pivot the data for heatmap
        heatmap_data = df_words_5.pivot(index='token', columns='nota', values='token_frequency').fillna(0)

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,  # Frequency matrix
            x=heatmap_data.columns,  # Grades (x-axis)
            y=heatmap_data.index,    # Tokens (y-axis)
            colorscale='Reds',       # Heatmap color scale
            #colorbar=dict(title="Frequência"),  # Colorbar title
            hoverongaps=False,       # Ensure no gaps show on hover
            hovertemplate=(
                "Token: %{y}<br>" + #Exibir título pelo Streamlit, q aí fica título pra cada coluna
                "Frequência: %{z}<extra></extra>"
            )
        ))

        # Update layout for better visualization
        fig.update_layout(
            # title="20 Tokens Mais Frequentes", >> Por título no streamlit
            height=500,  # Adjust height to make the heatmaps narrow
            width=300,  # Adjust width for each heatmap ....  * len(grades)
            template="plotly_white",
            plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
            paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
            font=dict(color="white")  # White font for all text
        )
        #fig.update_yaxes(title="Tokens", automargin=True)
        fig.update_xaxes(title="", showticklabels=False)

        # Show the figure
        fig.show()
        st.plotly_chart( fig )