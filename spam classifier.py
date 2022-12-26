# SPAM CLASSIFIER

# Importing the dataset
import pandas as pd
messages=pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])
messages.head()

# Data cleaning and Data preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]

for i in range(len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    

# Creating BAG OF WORDS
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
x=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

# Train Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Training the model using Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_model=MultinomialNB().fit(x_train,y_train)
y_predict=spam_model.predict(x_test)

# finding accuracy and confusion matrix
from sklearn.metrics import confusion_matrix
conf_m=confusion_matrix(y_test, y_predict)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_predict)


