import pandas as pd
import numpy as np
import nltk
from collections import Counter
import re
import plotly.graph_objects as go
import streamlit as st

# ==================================================================
# Page Settings
# ==================================================================
st.set_page_config(page_title = 'Frequência das Palavras',
                  layout= 'centered')

# ==================================================================
# Import dataset
# ==================================================================
csv_file_path = 'complete_corpus.csv'

# Read csv
df = pd.read_csv(csv_file_path)

# Create a copy
df_va = df.copy()

# ==================================================================
# Pré-Processamento
# ==================================================================
df_va.groupby('nota').count()

nltk.download('stopwords')
nltk.download('punkt') # Tokenizer, important to nltk.word_tokenize
nltk.download('punkt_tab') # Because just the line above was not working

# Removal of graphic symbols, punctuation, and spaces. I kept accents and removed numbers
def clean_cp(text):
    cleaned = text.lower() # All lowercase
    cleaned = re.sub('[^\w\s]', '', cleaned) # Removing punctuation
    cleaned = re.sub('\d+', '', cleaned) # Removing numbers
    cleaned = re.sub('\s+', ' ', cleaned) # Removing extra spaces
    cleaned = re.sub('\s+', ' ', cleaned)
    return cleaned.strip() # Removing tabs

df_va['content'] = df_va['content'].apply(clean_cp)

# Tokenizing removing stopwords: important to token frequency

def tokenized_cp(text):
   stopwords = nltk.corpus.stopwords.words('portuguese') # Loading the Portuguese stopwords
   tokenized = nltk.word_tokenize(text, language='portuguese') #Tokenizing
   text_sem_stopwords = [token for token in tokenized if token not in stopwords] # Removing stopwords
   return text_sem_stopwords

df_va['tokenized_content'] = df_va['content'].apply(tokenized_cp)


# ==================================================================
# Specific processing for this visualization
# ==================================================================

# Count tokens without stopwords

# Count tokens without considerate grade
df_va['num_tokens'] = df_va['tokenized_content'].apply(len)

# Group by grade
qtde_tokens_nota = df_va.groupby('nota')['num_tokens'].sum()

nota_token_counts = (
    df_va.groupby('nota')['tokenized_content']
    .apply(lambda texts: Counter([token for text in texts for token in text]))
)

# Create dataframe to visualization
df_frequency = nota_token_counts.reset_index()
df_frequency.columns = ['nota', 'token', 'token_frequency']
sorted_tokens = df_frequency.sort_values('token_frequency', ascending=False)

df_words = pd.DataFrame()

top_tokens_per_grade = (
    sorted_tokens.groupby('nota')
    .head(100)  # Assim mostra os 100 tokens mais comuns. Já vai ficar péssimo na visualização entao mantenho esse head e insiro outro abaixo.
    .reset_index(drop=True)
)
df_words = top_tokens_per_grade

# ==================================================================
# Specific processing for the second heatmap
# ==================================================================
file_path = 'task4_text.txt'

# Read the content of the file
with open(file_path, "r", encoding="utf-8") as file:
    given_text = file.read()

# Preprocessing of the task 4
cleaned_text = clean_cp(given_text)
tokenized_text = tokenized_cp(cleaned_text)

# Removing tokens of the task 4 from the set of tokens of the candidates' texts
# Ensure tokenized_text is a set for efficient subtraction
tokenized_text_set = set(tokenized_text)

# Subtract tokenized_text from each row in 'tokenized_content'
df_va['filtered_content'] = df_va['tokenized_content'].apply(
    lambda tokens: [token for token in tokens if token not in tokenized_text_set]
)

# Count the tokens number without stopwords for each text
# Count tokens without grade
df_va['num_tokens'] = df_va['filtered_content'].apply(len)

# Group by each grade
qtde_tokens_nota = df_va.groupby('nota')['num_tokens'].sum()

# Group token count according to each grade
nota_token_counts = (
    df_va.groupby('nota')['filtered_content']
    .apply(lambda texts: Counter([token for text in texts for token in text]))
)

# Converting each token and its count into a DataFrame
df_frequency = nota_token_counts.reset_index()
df_frequency.columns = ['nota', 'token', 'token_frequency']
sorted_tokens = df_frequency.sort_values('token_frequency', ascending=False)

df_words_ta = pd.DataFrame()

# Grouping the token count by grade to display the most common ones for each grade
top_tokens_per_grade = (
    sorted_tokens.groupby('nota')
    .head(100)  # Assim mostra os 100 tokens mais comuns. Já vai ficar péssimo na visualização entao mantenho esse head e insiro outro abaixo.
    .reset_index(drop=True)
)
df_words_ta = top_tokens_per_grade


# ==================================================================
# Streamlit Sidebar 
# ==================================================================
st.sidebar.markdown( '## Filtros' )

# Sidebar filter for grade
selected_grades = st.sidebar.multiselect(
    "Escolha a nota que deseja visualizar", [0, 1, 2, 3, 4, 5], default=[0, 1, 2, 3, 4, 5]
)

# Filter Dataframes based on selected grades
filtered_df_words = df_words[df_words['nota'].isin(selected_grades)]

st.markdown("")

# Filter to select the top x most frequent words
number = st.sidebar.slider(
    "Escolha o número de palavras mais frequentes que você deseja exibir",
    min_value=1,  # Minimum value of the slider
    max_value=100,  # Maximum value of the slider
    value=50,  # Default value
    step=1  # Step size
)

# ==================================================================
# Layout no Streamlit
# ==================================================================
st.header( 'Frequência das palavras' )

with st.container():
    st.markdown( """---""" )
    st.subheader(f'{number} Palavras mais frequentes')

    # Pivot the data for heatmap
    heatmap_data = filtered_df_words.pivot(index='token', columns='nota', values='token_frequency').fillna(0).head(number)

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,  # Frequency matrix
        x=heatmap_data.columns,  # Grades (x-axis)
        y=heatmap_data.index,    # Tokens (y-axis)
        colorscale='Reds',       # Heatmap color scale
        colorbar=dict(title="Frequência", titlefont=dict(size=14), tickfont=dict(size=12)),  # Colorbar title
        hoverongaps=False,       # Ensure no gaps show on hover
        hovertemplate=(
            "<span style='font-size:14px'><b>Token</b>: %{y}<br>" +  # Tooltip with larger font
            "<b>Nota</b>: %{x}<br>" +
            "<b>Frequência</b>: %{z}</span><extra></extra>"
        )
    ))

    # Update layout for better visualization
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
        font=dict(color="white", size=14),  # White font for all text
        height=600, # Increase height for better visualization
        width=1000
    )


    #fig.update_yaxes(title="Tokens", tickfont=dict(size=14), automargin=True)
    fig.update_yaxes(title="Tokens", showticklabels=False) # showticklabels=False to show the words just in tooltip
    fig.update_xaxes(title="", showticklabels=False)

    # Show the figure
    fig.show()
    st.plotly_chart( fig, use_container_width = True )


with st.container():
    st.subheader(f'{number} Palavras Mais Frequentes ao Retirar Tarefa 4')

    # Pivot the data for heatmap
    heatmap_data = df_words_ta.pivot(index='token', columns='nota', values='token_frequency').fillna(0).head(number)

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,  # Frequency matrix
        x=heatmap_data.columns,  # Grades (x-axis)
        y=heatmap_data.index,    # Tokens (y-axis)
        colorscale='Reds',       # Heatmap color scale
        colorbar=dict(title="Frequência", titlefont=dict(size=14), tickfont=dict(size=12)),  # Legend title and font size
        hoverongaps=False,       # Ensure no gaps show on hover
        hovertemplate=(
            "<span style='font-size:14px'><b>Token</b>: %{y}<br>" +  # Tooltip with larger font
            "<b>Nota</b>: %{x}<br>" +
            "<b>Frequência</b>: %{z}</span><extra></extra>"
        )
    ))

    # Update layout for better visualization
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
        font=dict(color="white", size=14),  # Update general font size
        height=600,  # Adjust height
        width=1000,   # Keep width consistent
    )

    # Update y-axis to hide tokens
    fig.update_yaxes(
        title="",  # No y-axis title
        showticklabels=False,  # Hide tokens from y-axis
    )

    # Update x-axis for better styling
    fig.update_xaxes(title="", showticklabels=False)

    # Show the figure
    fig.show()
    st.plotly_chart( fig, use_container_width = True )