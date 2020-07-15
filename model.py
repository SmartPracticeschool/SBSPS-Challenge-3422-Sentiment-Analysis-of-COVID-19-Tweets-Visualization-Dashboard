import pandas as pd
import numpy as np
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection,svm
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("tweets2.csv",encoding='latin-1',low_memory=False)
#dataset.head()

dataset['sentiment_text'] = dataset['sentiment_text'].str.lower() 

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt    
dataset['sentiment_text'] = np.vectorize(remove_pattern)(dataset['sentiment_text'], "@[\w]*")
dataset['sentiment_text'] = dataset['sentiment_text'].str.replace("[^a-zA-Z#]", " ")
dataset['sentiment_text'] = dataset['sentiment_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
print("Done")

dataset['sentiment_text']=[word_tokenize(entry) for entry in dataset['sentiment_text']]
print("Done")
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
i=0
for index,entry in enumerate(dataset['sentiment_text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    dataset.loc[index,'text_final'] = str(Final_words)
    i=i+1
    print(i)
#dataset['sentiment_text']=LancasterStemmer.stem(dataset['sentiment_text'])
print(dataset['text_final'])

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(dataset['text_final'],dataset['sentiment'],test_size=0.3)

Tfidf_vect = TfidfVectorizer(max_features=100000)
Tfidf_vect.fit(dataset['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

pickle.dump(Tfidf_vect, open('VECT.sav','wb'))
print("Done")

print(Tfidf_vect.vocabulary_)
print(Train_X_Tfidf)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma = 'auto')
SVM.fit(Train_X_Tfidf,Train_Y)

pickle.dump(SVM, open('SVM.sav','wb'))
print("Done")    
    
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
