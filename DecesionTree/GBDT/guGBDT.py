'''
Author: GuPeng

Time:2019/2/28

Theme: GBDT

'''
import pandas as pd
import numpy as np 
import random
import math
import copy
import sys,os
import re
from sklearn.externals import joblib
dirpath= os.path.abspath("..")+"\\DecisionTree"
sys.path.append(dirpath)

import DTregress as tree
from DTregress import strtofloat,classify


def classifyall(dataset,tree):
    allresult=[]
#    afactresult=[]
    for eachdata in dataset:
        resultone=classify(eachdata,tree)
        allresult.append(resultone)
#        afactresult.extend(eachdata[-1])
    return allresult
def genGBDT(dataset,features,treenum=500,maxdepth=6,lr=0.1):
    gblist=[]
    for i in xrange(treenum):
          print "tree is ",i 
          treechild=tree.buildTree(dataset,features,maxdepth=maxdepth)
          gblist.append(treechild)
          childresult=classifyall(dataset,treechild)
#          realreasult=realreasult-lr*childresult
          for i in xrange(len(dataset)):
              dataset[i][-1]=dataset[i][-1]-lr*childresult[i]
    return gblist
def initcal(dataset):
    meanre=np.mean([row[-1] for row in dataset]) 
    return meanre        
def testOneGBDT(observation,treelist,lr=0.1):
    pridict=sum(lr*classify(observation,tree) for tree in treelist)
    return pridict         
def modelsave(model,filename):
    modelfile=open(filename, 'w')
    lable=[i+1 for i in xrange(len(model))]
    savefile=lambda x:modelfile.write("Tree{} is :{}\n".format(x[0],x[1]))
    [savefile(moll) for moll in zip(lable,model)]
    print "{} has been saved".format(filename)

def modelload(filename):
    modelfile=open(filename, 'r')
    ss=modelfile.readlines()
    result=[]
    for i in range(0, len(ss)):
        ss[i] = ss[i].strip('\n')
        rr=re.search(r":[\S]+",(ss[i])).group()
        result.append(rr)
    return result


if __name__=="__main__":   
    df = pd.read_csv(os.path.abspath("..")+"\\housing.txt", header=None)[1:]   
    labels = df.columns.values.tolist()
    (traindata,testdata)=tree.splitData(df,splitrate=0.3)
    trainlabel=traindata.columns.values.tolist()
    train=strtofloat(traindata.values.tolist())
    testlabel=testdata.columns.values.tolist()
    test=strtofloat(testdata.values.tolist())    
#    model=genGBDT(train,trainlabel,treenum=500)
    print "testresult:",testOneGBDT(test[1],model)
    print "realresult",test[1][-1]