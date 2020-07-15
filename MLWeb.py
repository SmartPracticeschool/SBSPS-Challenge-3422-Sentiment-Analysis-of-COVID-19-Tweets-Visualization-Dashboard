import pickle
from flask import Flask, request, render_template, send_file
from twitterscraper import query_tweets
import datetime as dt
import pandas as pd
import random
from gtts import gTTS 
import os
from playsound import playsound
import matplotlib.pyplot as plt
import numpy as np

z=1
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
#CORS(app)

vectorize = pickle.load(open('VECT.sav', 'rb'))
classifier = pickle.load(open('SVM.sav', 'rb'))
@app.route("/", methods=['GET','POST'])
def return_sentiment():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':       
        sentence = request.form['sentence']
        if(sentence):
            sentence_vector = vectorize.transform([sentence])
            sentiment = classifier.predict(sentence_vector).tolist()
            return render_template('index.html', original_input= {'sentence':sentence}, result = sentiment[0])

@app.route("/scrape/",methods=['GET','POST'])
def return_data():
    if request.method == 'POST':
        today = dt.date.today()
        yesterday = today - dt.timedelta(days = 1)
        begin_date=yesterday
        end_date = today 
        limit = 100
        lang='en'
        tweets = query_tweets("corona",begindate = begin_date,enddate = end_date ,limit=limit ,lang=lang)
        df = pd.DataFrame(t.__dict__ for t in tweets)
        i=df.text.str.strip()
        j=df.timestamp
        l=random.randint(1,len(i))
        tweet=i[l]
        tweet_vector = vectorize.transform([tweet])
        sent = classifier.predict(tweet_vector).tolist()
        mytext = tweet
        language = 'en'
        myobj = gTTS(text = mytext, lang = language, slow = False)
        myobj.save("audio.mp3")

        
        return render_template('index.html', tw = tweet, time=j[l], sent = sent[0])
    else:
        return render_template('index.html')
   
@app.route("/scrape/audio/",methods=['GET'])
def return_audio():
    playsound("audio.mp3")
    os.remove("audio.mp3")
    return render_template('index.html')

@app.route("/dashboard/",methods=['GET'])
def return_plots():
    today = dt.date.today()
    yesterday = today - dt.timedelta(days = 1)
    x=[0 for t in range(7)]
    y=[0 for t in range(7)]
    for m in range(7):
        begin_date = yesterday - dt.timedelta(days = m)
        end_date = today - dt.timedelta(days = m)
        lang='en'
        limit=500
        tweets = query_tweets("COVID",begindate = begin_date, enddate = end_date ,limit = limit ,lang=lang)
        df = pd.DataFrame(t.__dict__ for t in tweets)
        i=df.text
        j=df.timestamp
        n = random.sample(range(0,len(i)),100)
        
        for k in n:
            date = j[k]
            tweet= i[k]
            tweet_vector = vectorize.transform([tweet])
            sent = classifier.predict(tweet_vector).tolist()
            if(sent[0]==1):
                x[m]=x[m]+1
            else:
                y[m]=y[m]+1
    dates = [today-dt.timedelta(days = z) for z in range (7)]
    x=np.array(x)
    y=np.array(y)
    plt.bar(dates, x, width=0.8, label='p', color='gold', bottom=y)
    plt.bar(dates, y, width=0.8, label='n', color='silver')
     
    plt.ylim(0,120)
    plt.ylabel("No. of Sentiments")
    plt.xlabel("Date")
    plt.xticks(rotation=25)
    plt.legend(loc='centre', bbox_to_anchor=(1,0.60), ncol = 1)
    plt.title("Sentiment Analysis")
    
    plt.savefig('C:/Users/KISHAN/Documents/Sentiment Analysis IBM/IBM/static/img/plot2.png')
    plt.show()
    colors = ['lightskyblue', 'lightcoral']
    labels = ['Positive', 'Negative',]
    sizes=[sum(x),sum(y)]
    explode = (0.2, 0)
    patches, texts = plt.pie(sizes, explode=explode, colors=colors, shadow=True, startangle=45)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('C:/Users/KISHAN/Documents/Sentiment Analysis IBM/IBM/static/img/plot3.png')
    plt.show()
    return render_template('index.html')
    
if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)