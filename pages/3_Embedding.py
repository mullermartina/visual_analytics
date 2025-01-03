import pandas as pd
import numpy as np
import nltk
from collections import Counter #uso para n-gramas
import re 
import plotly.express as px #biblio q pus pra essa viz
from sklearn.manifold import TSNE #biblio q pus pra essa viz
from gensim.models import Word2Vec
import streamlit as st
import plotly.graph_objects as go

# ==================================================================
# Configurações da Página 
# ==================================================================
st.set_page_config(page_title = 'Embedding Fixa',
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

nltk.download('stopwords')
nltk.download('punkt') # é um tokenizador, importante para nltk.word_tokenize
nltk.download('punkt_tab') # download desse pacote pq o punkt sozinho nao tava funcionando

# Pré-processamento
#Retirada de sinais gráficos, pontuações e espaços
def clean_cp(text):
    cleaned = text.lower() #Deixando tudo minúsculo
    cleaned = re.sub('[^\w\s]', '', cleaned) # Removendo pontuacao
    #cleaned = re.sub('[0-9]+', '', cleaned) # Removendo números 
    cleaned = re.sub('\d+', '', cleaned) # Removendo números NÃO TAVA FUNCIONANDO. Começou a funcionar qdo pus o lower como primeiro comando da funçao
    cleaned = re.sub('\s+', ' ', cleaned) # Removendo espaços extras
    cleaned = re.sub('\s+', ' ', cleaned)
    return cleaned.strip() # Removendo tabs

df_va['content'] = df_va['content'].apply(clean_cp)

# Tokenizando e retirando stopwords: retiro stopwords pq embeddings fixas nao consideram contexto
def tokenized_cp(text):
   stopwords = nltk.corpus.stopwords.words('portuguese') # Carregando as stopwords do português
   tokenized = nltk.word_tokenize(text, language='portuguese') #Transforma o texto em tokens
   text_sem_stopwords = [token for token in tokenized if token not in stopwords] # Deixando somente o que nao é stopword no texto
   return text_sem_stopwords

df_va['tokenized_content'] = df_va['content'].apply(tokenized_cp)

# Prepare the tokenized corpus
tokenized_corpus = df_va['tokenized_content'].tolist()

# ==================================================================
# Treinamento da Embedding
# ==================================================================
model = Word2Vec( # Se ficar pesado, inserir no streamlit somente o modelo treinado já
    sentences=tokenized_corpus,  # Input tokenized sentences
    vector_size=100,            # Dimensionality of the embedding vectors
    window=5,                   # Context window size
    min_count=1,                # Minimum word frequency
    sg=0,                       # CBOW (0) or Skip-gram (1)
    workers=1,                  # workers=1 to eliminate ordering jitter from OS thread scheduling. To do a reproducible run
    seed=42                     # To do a reproducible run
)

# ==================================================================
# Layout no Streamlit
# ==================================================================
st.header( 'Embedding Fixa ' )

with st.container():
    st.markdown("""---""")
    st.subheader('Embedding das Palavras do Corpus')

     # TSNE visualization function
    def visualize_embeddings(model, selected_word=None, num_neighbors=10, num_points=1000):
        words = list(model.wv.index_to_key)[:num_points]  # Limit the number of words to display
        vectors = model.wv[words]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_vectors = tsne.fit_transform(vectors)

        df_fe = pd.DataFrame({
            'Token': words,
            'x': reduced_vectors[:, 0],
            'y': reduced_vectors[:, 1]
        })

        # Define colors
        if selected_word:
            neighbors = model.wv.most_similar(selected_word, topn=num_neighbors)
            neighbor_words = [selected_word] + [word for word, _ in neighbors]
            df_fe['color'] = df_fe['Token'].apply(
                lambda x: '#FF4B4B' if x == selected_word else 
                        ('#FF904B' if x in neighbor_words else '#A0A0A0')  # Gray for others
            )
            df_filtered = df_fe[df_fe['Token'].isin(neighbor_words + [selected_word])]
        else:
            df_fe['color'] = '#FF4B4B'  # Red for all points
            df_filtered = df_fe

        # Create the scatter plot with Plotly Graph Objects
        fig = go.Figure()

        # Add points
        fig.add_trace(go.Scatter(
            x=df_filtered['x'],
            y=df_filtered['y'],
            mode='markers+text',
            marker=dict(
                color=df_filtered['color'],
                size=10,
            ),
            text=df_filtered['Token'],
            textposition='top center'
        ))

        # Customize layout
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
            paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent outer background
            font=dict(color='white'),  # White font for axis labels and title
            hoverlabel=dict(
                bgcolor='white',  # White background for tooltips
                font_size=14,     # Tooltip font size
                font_color='black'  # Tooltip font color
            ),
            xaxis_title='Dimensão 1',
            yaxis_title='Dimensão 2'
        )
    
        return fig

    # Streamlit Filters
    st.markdown('<p style="font-size:20px;">Você quer visualizar todas as palavras ou apenas uma palavra específica e seus vizinhos?</p>', unsafe_allow_html=True)
    visualization_choice = st.radio(
        "",
        options=["Visualizar todas as palavras", "Visualizar uma palavra específica e seus vizinhos"]
    )

    st.write("")  # Adding a blank line

    if visualization_choice == "Visualizar todas as palavras":
        st.write("Visualizando todas as palavras:")
        fig = visualize_embeddings(model)
    else:
        st.markdown('<p style="font-size:18px;">Escolha uma palavra para visualizar:</p>', unsafe_allow_html=True)
        word_list = list(model.wv.index_to_key)  # Full list of vocabulary
        selected_word = st.selectbox('', word_list)
        st.write("")  # Adding a blank line

        if selected_word:
            st.write(f'Visualizando a palavra **{selected_word}** e seus vizinhos mais próximos:')
            fig = visualize_embeddings(model, selected_word)
        else:
            st.write("Escolha uma palavra para visualizar:")

    st.plotly_chart(fig, use_container_width=True)