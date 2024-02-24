#Read the data 
data_links = ["https://www.churchofjesuschrist.org/study/general-conference/2023/10/11bednar?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/12wright?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/13daines?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/14godoy?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/15christofferson?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/16ardern?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/17oaks?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/22andersen?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/23newman?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/24costa?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/25stevenson?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/26choi?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/27phillips?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/28rasband?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/31sabin?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/32koch?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/33runia?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/34soares?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/41ballard?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/42freeman?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/43parrella?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/44cook?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/45uchtdorf?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/46waddell?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/47eyring?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/57renlund?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/52pingree?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/53cordon?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/55esplin?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/54gong?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/56giraud-carrier?lang=eng","https://www.churchofjesuschrist.org/study/general-conference/2023/10/51nelson?lang=eng"]
speaker_key = ["bednar","wright","daines","godoy","christofferson","arden","oaks","andersen","newman","costa","stevenson","choi","phillips","rasband","sabin","koch","runia","soares","ballard","freeman","parrella","cook","uchtdorf","waddell","eyring","renlund","pingree","cordon","esplin","gong","giraud-carrier","nelson"]
speaker_name = ['Elder David A. Bednar','Sister Amy A. Wright','Elder Robert M. Daines','Elder Carlos A. Godoy','Elder D. Todd Christofferson','Elder Ian S. Ardern','President Dallin H. Oaks','Elder Neil L. Andersen','Brother Jan E. Newman','Elder Joaquin E. Costa','Elder Gary E. Stevenson','Elder Yoon Hwan Choi','Elder Alan T. Phillips','Elder Ronald A. Rasband','Elder Gary B. Sabin','Elder Joni L. Koch','Sister Tamara W. Runia','Elder Ulisses Soares','President M. Russell Ballard','President Emily Belle Freeman','Elder Adilson de Paula Parrella','Elder Quentin L. Cook','Elder Dieter F. Uchtdorf','Bishop W. Christopher Waddell','President Henry B. Eyring','Elder Dale G. Renlund','Elder John C. Pingree Jr.','Elder Valeri V. Cordón','Elder J. Kimo Esplin','Elder Gerrit W. Gong','Elder Christophe G. Giraud-Carrier','President Russell M. Nelson']
titles = ['In the Path of Their Duty','Abide the Day in Christ','Sir, We Would Like to See Jesus','For the Sake of Your Posterity','The Sealing Power','Love Thy Neighbor','Kingdoms of Glory','Tithing: Opening the Windows of Heaven','Preserving the Voice of the Covenant People in the Rising Generation','The Power of Jesus Christ in Our Lives Every Day','Promptings of the Spirit','Do You Want to Be Happy?','God Knows and Loves You','How Great Will Be Your Joy','Hallmarks of Happiness','Humble to Accept and Follow','Seeing God’s Family through the Overview Lens','Brothers and Sisters in Christ','Praise to the Man','Walking in Covenant Relationship with Christ','Bearing Witness of Jesus Christ in Word and Actions','Be Peaceable Followers of Christ','The Prodigal and the Road That Leads Home','More Than a Hero','Our Constant Companion','Jesus Christ Is the Treasure','Eternal Truth','Divine Parenting Lessons','The Savior’s Healing Power upon the Isles of the Sea','Love Is Spoken Here','We Are His Children','Think Celestial!']

#Read the data text 
from urllib.request import urlopen
from bs4 import BeautifulSoup

def get_text_html(url):

    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()
    soup.body.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    soup.get_text(separator='\n', strip=True)

    return text 

X = [get_text_html(i) for i in data_links] 

#remove the header string 
str_header = "Authenticating...\nGeneral Conference\nOctober 2023 general conferenceContentsSaturday Morning SessionIn the Path of Their DutyDavid A. BednarAbide the Day in ChristAmy A. WrightSir, We Would Like to See JesusRobert M. DainesFor the Sake of Your PosterityCarlos A. GodoyThe Sealing PowerD. Todd ChristoffersonLove Thy NeighbourIan S. ArdernKingdoms of GloryDallin H. OaksSaturday Afternoon SessionSustaining of General Authorities, Area Seventies, and General OfficersHenry B. EyringTithing: Opening the Windows of HeavenNeil L. AndersenPreserving the Voice of the Covenant People in the Rising GenerationJan E. NewmanThe Power of Jesus Christ in Our Lives Every DayJoaquin E. CostaPromptings of the SpiritGary E. StevensonDo You Want to Be Happy?Yoon Hwan ChoiGod Knows and Loves YouAlan T. PhillipsHow Great Will Be Your JoyRonald A. RasbandSaturday Evening SessionHallmarks of HappinessGary B. SabinHumble to Accept and FollowJoni L. KochSeeing God’s Family through the Overview LensTamara W. RuniaBrothers and Sisters in ChristUlisses SoaresSunday Morning SessionPraise to the ManM. Russell BallardWalking in Covenant Relationship with ChristEmily Belle FreemanBearing Witness of Jesus Christ in Word and ActionsAdilson de Paula ParrellaBe Peaceable Followers of ChristQuentin L. CookThe Prodigal and the Road That Leads HomeDieter F. UchtdorfMore Than a HeroW. Christopher WaddellOur Constant CompanionHenry B. EyringSunday Afternoon SessionJesus Christ Is the TreasureDale G. RenlundEternal TruthJohn C. Pingree Jr.Divine Parenting LessonsValeri V. CordónThe Savior’s Healing Power upon the Isles of the SeaJ. Kimo EsplinLove Is Spoken HereGerrit W. GongWe Are His ChildrenChristophe G. Giraud-CarrierThink Celestial!Russell M. Nelson\n"

X = [X[i].replace(str_header,"") for i in range(0,len(X))]

#Text processing to determine string similarity 

import pandas as pd
import re
import string
#These are only used when make_lemmas=False. Commented out for speed of download 
#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#from nltk.corpus import stopwords

#pip install spacy 
#python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

#Import visualization libraries
import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

#Clustering libraries
#pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#For calculating the distance metrics
#from spicy.spatial.distance import pdist, squareform
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


#Function to process the text: 
def preprocess_text(text: str, make_lemmas: bool) -> str:
    """This function cleans the input text by
    - removing links
    - removing special chars
    - removing numbers
    - removing stopwords
    - transforming in lower case
    - removing excessive whitespaces
    Arguments:
        text (str): text to clean
        remove_stopwords (bool): remove stopwords or not
    Returns:
        str: cleaned text
    """
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove numbers and special chars
    text = re.sub("[^A-Za-z]+", " ", text)
    #Remove numbers, special characters and punctuation
    text = re.sub(r'[^\w\s]'," ",text)
    # remove stopwords
    if make_lemmas:
        doc = nlp(text.lower())
        text = " ".join([token.lemma_ for token in doc if not token.is_stop]) 
    else:
        # 1. creates tokens
        tokens = nltk.word_tokenize(text)
        # 2. checks if token is a stopword and removes it
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. joins all tokens again
        text = " ".join(tokens)
    # returns cleaned text
    text = text.lower().strip()
    return text

#Y0 = [preprocess_text(i,make_lemmas=False) for i in X] #unlemmatized text 
Y = [preprocess_text(i,make_lemmas=True) for i in X]


#Compile all documents for the entire conference 
YY = " ".join(Y)

# Document term matrix 
vectorizer = TfidfVectorizer(sublinear_tf=True)

# fit_transform applies TF-IDF to clean texts - save the array of vectors in X
Ymat = vectorizer.fit_transform(Y) #do individually as well 

#xdat2 is best 

XF = pd.DataFrame((Ymat.toarray().T),index=vectorizer.get_feature_names_out(),columns=speaker_key)

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
                 options=[{'label': speaker_name[x] , 'value': speaker_key[x]} for x in range(0,len(speaker_key))],
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
        cloud = WordCloud(width=960,
                        height=768,
                        max_words=150,
                        colormap='tab20c',
                        random_state=42,
                        stopwords=stopwords,
                        collocations=False).generate_from_text(data)
    if method=="bigrams":
        cloud = WordCloud(width=960,
                        height=768,
                        max_words=150,
                        colormap='tab20c',
                        random_state=42,
                        stopwords=stopwords,
                        collocations=True).generate_from_text(data)
    if method == "tf-idf":
        cloud = WordCloud(width=960,
                        height=768,
                        max_words=150,
                        colormap='tab20c',
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
    nkey = np.where(n == pd.Series(speaker_key))[0][0]
    if z == "all":
        if y == "tf-idf":
            generate_wordcloud(XF.sum(axis=1),'All',method='idf').save(img, format='PNG')  
        else:
            generate_wordcloud(YY,'All',method=y).save(img, format='PNG')
    else:
        if y == "tf-idf":
            generate_wordcloud(XF[n],'dropdown',method='idf').save(img, format='PNG')  #currently not printing with title
        else:
            generate_wordcloud(Y[nkey],'dropdown',method=y).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

if __name__ == "__main__":
    #app.run_server(debug=False,port=1217)
    app.run_server(debug=False)
