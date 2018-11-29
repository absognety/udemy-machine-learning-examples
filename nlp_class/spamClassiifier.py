#Building a spam classifier

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data = pd.read_csv('spambase.data').as_matrix()
data_ = pd.read_csv('spambase.data')

np.random.shuffle(data)

#input and label sets split
X = data[:,:48]
Y = data[:,-1]

trainX = X[:-100,]
trainY = Y[:-100,]

testX = X[-100:,]
testY = Y[-100:,]

ml_model = MultinomialNB()
ml_model.fit(trainX,trainY)

train_score = ml_model.score(trainX,trainY)
test_score = ml_model.score(testX,testY)

print ("classification rate for training set:" ,train_score)
print ("classification rate for test set:" ,test_score)

from sklearn.ensemble import AdaBoostClassifier

adB_model = AdaBoostClassifier()
adB_model.fit(trainX,trainY)
print ("Training Score:",adB_model.score(trainX,trainY))
print ("Test Score:",adB_model.score(testX,testY))


 
 



