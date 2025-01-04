import pandas as pd
import numpy as np
import nltk
from collections import Counter
import re 
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==================================================================
# Page Settings
# ==================================================================
st.set_page_config(page_title = 'Tokens e Types',
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
# Preprocessing
# ==================================================================
df_va.groupby('grade').count()

df_chart = pd.DataFrame()

# Number of texts
df_chart['qty_texts'] = df_va['grade'].value_counts()

nltk.download('stopwords')
nltk.download('punkt') # Tokenizer, important to nltk.word_tokenize
nltk.download('punkt_tab') # Because just the line above was not working

# Removal of graphic symbols, punctuation, and spaces: I kept accents, removed numbers, and did not apply stemming
def clean_cp(text):
    cleaned = text.lower() # All lowercase
    cleaned = re.sub('[^\w\s]', '', cleaned) # Removing punctuation
    cleaned = re.sub('\d+', '', cleaned) # Removing numbers
    cleaned = re.sub('\s+', ' ', cleaned) # Removing extra spaces
    return cleaned.strip() # Removing tabs

df_va['content'] = df_va['content'].apply(clean_cp)

# Tokenizing WITHOUT removing stopwords
def tokenized_cp(text):
   tokenized = nltk.word_tokenize(text, language='portuguese') # Text to tokens
   return tokenized

df_va['tokenized_content'] = df_va['content'].apply(tokenized_cp)

# Verifying the number of tokens for each grade (I use the total for each grade to calculate TTR)

# Count the number of tokens for each text and create a column with the count
df_va['token_count'] = df_va['tokenized_content'].apply(len)

# Count the number of tokens accordingly to each grade       
df_chart['qty_total_tokens_grade'] = df_va.groupby('grade')['token_count'].sum()

# Verifying the average number os tokens for each grade
df_chart['qty_avg_tokens'] = round((df_chart['qty_total_tokens_grade'] / df_chart['qty_texts']),2)

# Count types (tokens without repetition)
df_va['types_content'] = df_va['tokenized_content'].apply(lambda tokens: list(set(tokens)))

# Verifying the number of types for each grade

# Count the number of types for each text and create a column with the count
df_va['types_count'] = df_va['types_content'].apply(len)

# Count the number of types accordingly to each grade
df_chart['qty_total_types_grade'] = df_va.groupby('grade')['types_count'].sum()

# Verifying the average number os types for each grade
df_chart['qty_avg_types'] = round((df_chart['qty_total_types_grade'] / df_chart['qty_texts']),2)

# TTR
df_chart['TTR'] = round(((df_chart['qty_total_types_grade'] / df_chart['qty_total_tokens_grade']) * 100),2)

# Minimum number os tokens to each grade

# Number os tokens to each text
df_va['token_count'] = df_va['tokenized_content'].apply(len)

# Minimum number os tokens to each grade     
df_chart['qty_tokens_min'] = df_va.groupby('grade')['token_count'].min()

# Maximum number os tokens to each grade

# Number os tokens to each text ## remove this line. i already calculated
df_va['token_count'] = df_va['tokenized_content'].apply(len)

# Maximum number os tokens to each grade      
df_chart['qty_tokens_max'] = df_va.groupby('grade')['token_count'].max()

# Standard deviation of the number of tokens by grade

df_va['token_count'] = df_va['tokenized_content'].apply(len)

df_chart['std_qty_tokens'] = round((df_va.groupby('grade')['token_count'].std()),2)

df_chart['real_grade'] = [3, 2, 4, 5, 1, 0]
df_chart.sort_values('real_grade')

# ==================================================================
# Sidebar
# ==================================================================
st.sidebar.markdown( '## Filtro' )

# Sidebar filter for grade
selected_grades = st.sidebar.multiselect(
    "Escolha a nota que deseja visualizar", [0, 1, 2, 3, 4, 5], default=[0, 1, 2, 3, 4, 5]
)

# Filter DataFrames based on selected grades
filtered_df_chart = df_chart[df_chart['real_grade'].isin(selected_grades)]
filtered_df_va = df_va[df_va['grade'].isin(selected_grades)]

# ==================================================================
# Layout no Streamlit
# ==================================================================
st.header( 'Tokens e Types' )

with st.container():
    st.markdown( """---""" )
    st.subheader( 'Número de Textos' )

    # Create the bar chart
    fig = px.bar(
      filtered_df_chart,  # Pass the entire DataFrame
      x='real_grade',  # X-axis
      y='qty_texts',  # Y-axis
      labels={'qty_texts': 'Quantidade', 'real_grade': 'Nota'},  # Custom axis labels
      color_discrete_sequence=['#FF4B4B']  # Streamlit red color
    )

    # Update the tooltip
    fig.update_traces(
       hovertemplate="%{y} textos<extra></extra>"  # Custom tooltip  
    )

    # Update layout for transparent background and tooltip styles
    fig.update_layout(
        #title="Quantidade de Textos por Nota",
        xaxis_title="Nota",
        yaxis_title="Quantidade",
        template="plotly_white",
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
        font=dict(color="white"),  # White font for titles, labels, and text
        xaxis=dict(
        showgrid=False,  # Remove vertical gridlines
        color="white"    # White font for x-axis labels
        ),
        yaxis=dict(
        #showgrid=False,  # Remove horizontal gridlines
        color="white"    # White font for y-axis labels
        ),
        hoverlabel=dict(
        bgcolor="#FFFFFF",  # Light red background
        font_size=14,       # Tooltip font size
        font_color="black"  # Tooltip text color
        )
    )

    # Show the figure
    fig.show()
    st.plotly_chart( fig )

with st.container():
    st.markdown( """---""" )
    st.subheader( 'Número Máximo e Mínimo de Tokens ')
    
    customdata = filtered_df_chart[['qty_avg_tokens', 'std_qty_tokens']].values

    # Create the figure
    fig = go.Figure()

    # Add the bars for max tokens
    fig.add_trace(go.Bar(
        x=filtered_df_chart['real_grade'], 
        y=filtered_df_chart['qty_tokens_max'], 
        name='Máximo de tokens', 
        marker_color='#FF4B4B',  # Streamlit-like red
        hovertemplate=(
            "Máximo de tokens: %{y}<br>" +
            "Número médio: %{customdata[0]:.2f}<br>" +
            "Desvio padrão: %{customdata[1]:.2f}<extra></extra>"     
        ),
        customdata=customdata
    ))

    # Add the bars for min tokens
    fig.add_trace(go.Bar(
        x=filtered_df_chart['real_grade'], 
        y=filtered_df_chart['qty_tokens_min'], 
        name='Mínimo de tokens', 
        marker_color='#FF904B',  # New color: orange
        hovertemplate=(
            "Mínimo de tokens: %{y}<br>" +
            "Número médio: %{customdata[0]:.2f}<br>" +
            "Desvio padrão: %{customdata[1]:.2f}<extra></extra>"       
        ),
        customdata=customdata
    ))

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Nota",
        yaxis_title="Quantidade",
        barmode='group',  # Grouped bars
        legend_title="Legenda",
        template='plotly_white',
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
        font=dict(color="white"),  # White font for titles, labels, and text
        xaxis=dict(
            showgrid=False,  # Remove vertical gridlines
            color="white"    # White font for x-axis labels
        ),
        yaxis=dict(
            #showgrid=False,  # Remove horizontal gridlines
            color="white"    # White font for y-axis labels
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",  # Light red background for tooltip
            font_size=14,       # Tooltip font size
            font_color="black"  # Tooltip text color
        )
    )

    # Show the figure
    fig.show()
    st.plotly_chart( fig )

              
with st.container():
    st.markdown( """---""" )
    st.subheader( 'Número Médio de Tokens, Types e TTR' )

    # Create the figure
    fig = go.Figure()

    # Add the bars for avg tokens
    fig.add_trace(go.Bar(
        x=filtered_df_chart['real_grade'], 
        y=filtered_df_chart['qty_avg_tokens'], 
        name='Média de tokens', 
        marker_color='#FF4B4B',  # Streamlit-like red
        hovertemplate=(
            "Média de tokens: %{y}<br>" +
            "TTR: %{customdata:.2f}<extra></extra>%"       
        ),
        customdata=filtered_df_chart['TTR']
    ))

    # Add the bars for avg types
    fig.add_trace(go.Bar(
        x=filtered_df_chart['real_grade'], 
        y=filtered_df_chart['qty_avg_types'], 
        name='Média de types', 
        marker_color='#FF904B',  # New color: orange
        hovertemplate=(
            "Média de types: %{y}<br>" +
            "TTR: %{customdata:.2f}<extra></extra>%"       
        ),
        customdata=filtered_df_chart['TTR']

    ))

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Nota",
        yaxis_title="Quantidade",
        barmode='group',  # Grouped bars
        legend_title="Legenda",
        template='plotly_white',
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
        font=dict(color="white"),  # White font for titles, labels, and text
        xaxis=dict(
            showgrid=False,  # Remove vertical gridlines
            color="white"    # White font for x-axis labels
        ),
        yaxis=dict(
            #showgrid=False,  # Remove horizontal gridlines
            color="white"    # White font for y-axis labels
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",  # Light red background for tooltip
            font_size=14,       # Tooltip font size
            font_color="black"  # Tooltip text color
        )

    )

    # Show the figure
    fig.show()
    st.plotly_chart( fig )

with st.container():
    st.markdown( """---""" )
    st.subheader( 'Distribuição de Types e Tokens por Notas' )

    # Create the scatterplot
    fig = px.scatter(
        filtered_df_va,
        x='token_count',                # x-axis: total tokens
        y='types_count',                # y-axis: unique tokens
        color='grade',                   # Color the dots based on the 'grade' column
        hover_data=['token_count', 'types_count', 'grade'],  # Tooltip includes 'grade' #title="Quantidade de Tokens e Types segundo Nota"
        labels={
            'token_count': 'Tokens',
            'types_count': 'Types',
            'grade': 'Nota'
        },
        color_continuous_scale='Agsunset' 
    )

    # Customize layout
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
        font=dict(color='white'),       # White font for titles, labels, and text
        #title_font=dict(size=18),       # Title font size
        xaxis_title_font=dict(size=16), # X-axis label font size
        yaxis_title_font=dict(size=16), # Y-axis label font size
    )

    # Customize tooltip
    fig.update_traces(
        hoverlabel=dict(
            bgcolor='white',            # White background for tooltip
            font_size=14,               # Font size for tooltip text
            font_color='black'          # Black font color
        )
    )

    # Show the figure
    fig.show()
    st.plotly_chart( fig, use_container_width = True )

