from sklearn import naive_bayes
import numpy as np

#load data
def loadDataSet():
    postingList = [
        ['my','dog','has','flea','problems','help','please'],
        ['maybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid'],
    ]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

#create vocabulary list
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#convert a vocab list into vector
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:   print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

listOPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)

#create a training model
trainClasses = np.array(listClasses).reshape(-1,1)
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))

clf = naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
clf.fit(trainMat,trainClasses)

#test the model
testEntry = ['love','my','dalmation']
thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry)).reshape(1,-1)
print(testEntry,'classified as:',clf.predict(thisDoc))
testEntry = ['stupid','garbage']
thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry)).reshape(1,-1)
print(testEntry,'classified as:',clf.predict(thisDoc))
