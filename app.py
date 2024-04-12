import os 
import pandas as pd 

dat_apr_24 = pd.read_json("Data/April 2024/dat_apr_24.json")
XF = pd.read_csv("Data/April 2024/XF_apr_24.csv")
XF = XF.rename(columns={'Unnamed: 0':'word'}).set_index('word')

YY = " ".join(dat_apr_24.text)


from wordcloud import WordCloud, ImageColorGenerator
#import matplotlib.pyplot as plt

#Define a list of stop words
#stopwords = ['general','conference']
stopwords = ['s','t','m'] #does this work for the ALL bigrams from words with contractions
#A function to generate the word cloud from text. Modify this function to give options for bigrams, words, and document frequency 

import pandas as pd
import dash
#import dash_html_components as html
from dash import html, dcc 
#import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from io import BytesIO
import base64
import dash.dependencies as dd
from wordcloud import WordCloud
import numpy as np

##

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LUX])

server = app.server

app.layout = html.Div([
    html.H1(children='General Confernce Wordclouds: October 2023'),
    html.Div(children='''
        A web application to visualize word frequencies in general conference addresses.
    '''),
    html.Br(),
    html.Label('Choose a Speaker or All Talks in the Session'),
    dcc.RadioItems(
        id='check',
        options=[
            {'label': 'All Talks in Session', 'value': 'all'},
            {'label': 'Speaker', 'value': 'Speaker'},
            # Add more options as needed
        ],
        value=['Speaker'],  # Initial checked items
    ),

    html.Br(),
    html.Label('Choose Speaker'),
    dcc.Dropdown(id='dropdown',
                 #options=[{'label':x, 'value':x} for x in speaker_key],
                 #switch to speaker_name 
                 options=[{'label': dat_apr_24.speaker_name[x] , 'value': dat_apr_24.speaker_key[x]} for x in range(0,len(dat_apr_24.speaker_key))],
                 value='nelson'),
    html.Label('Choose Method:*'),
    dcc.RadioItems(id='radio',options=['words', 'bigrams', 'tf-idf'], value='bigrams'),

    html.Br(),
    html.H2(id='title'),
    html.Img(id='image_wc'),
    html.Div(children='''
        *Words shows the most frequent words, bigrams show the most frequent pairs of words, and tf-idf measuers the term importance within a talk relative to the session. 
    ''')
])

def generate_wordcloud(data, title, method='words'):
    if method=="words":
        cloud = WordCloud(width=576,
                        height=480,
                        #max_words=150,
                        colormap='Paired',
                        random_state=42,
                        stopwords=stopwords,
                        collocations=False).generate_from_text(data)
    if method=="bigrams":
        cloud = WordCloud(width=576,
                        height=480,
                        #max_words=150,
                        colormap='Paired',
                        random_state=42,
                        stopwords=stopwords,
                        collocations=True).generate_from_text(data)
    if method == "tf-idf":
        cloud = WordCloud(width=576,
                        height=480,
                        #max_words=150,
                        colormap='Paired',
                        random_state=42,
                        stopwords=stopwords,
                        collocations=False).generate_from_frequencies(data)

    # plt.figure(figsize=(10,8))
    # plt.imshow(cloud)
    # plt.axis('off')
    # plt.title(title, fontsize=13)
    # plt.show()
    return cloud.to_image()

@app.callback(
    Output(component_id='title', component_property='children'),
    Input(component_id='dropdown', component_property='value')
)
def update_output_div(input_value):
    #return f'Output: {input_value}'
    nkey = np.where(input_value == pd.Series(speaker_key))[0][0]
    return titles[nkey]

@app.callback(Output('image_wc', 'src'),
              [Input('image_wc', 'id'),
               Input('dropdown', 'value'),
               Input('radio','value'),
               Input('check','value')])
def make_image(b,n,y,z):
    img = BytesIO()
    #dff = df[df['Location'] == n]
    #dff2 = dff.groupby(["Title"])["Title"].count().reset_index(name="count")
    #plot_wordcloud(data=dff2).save(img, format='PNG')
    nkey = np.where(n == pd.Series(dat_apr_24.speaker_key))[0][0]
    if z == "all":
        if y == "tf-idf":
            generate_wordcloud(XF.sum(axis=1),'All',method='tf-idf').save(img, format='PNG')  
        else:
            generate_wordcloud(YY,'All',method=y).save(img, format='PNG')
    else:
        if y == "tf-idf":
           generate_wordcloud(XF[n],'dropdown',method='tf-idf').save(img, format='PNG')
            #generate_wordcloud(XF[nkey],'dropdown',method='tf-idf').save(img, format='PNG')
              #currently not printing with title
        else:
            #generate_wordcloud(Y[nkey],'dropdown',method=y).save(img, format='PNG')
            generate_wordcloud(dat_apr_24.text[nkey],'dropdown',method=y).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
#Dropdown key wasn't doing anything 


if __name__ == "__main__":
    app.run_server(debug=False,host='0.0.0.0')
