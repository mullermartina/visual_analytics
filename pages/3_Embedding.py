import pandas as pd
import numpy as np
import nltk
from collections import Counter #uso para n-gramas
import re 
import plotly.express as px #biblio q pus pra essa viz
from sklearn.manifold import TSNE #biblio q pus pra essa viz
from gensim.models import Word2Vec
import streamlit as st

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