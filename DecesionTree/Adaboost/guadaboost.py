# -*- coding: utf-8 -*-
'''
Author: GuPeng

Time:2019/3/5

Theme: Adaboost

'''
import pandas as pd
import numpy as np 
import random
import math
import copy
import sys,os
import re
dirpath= os.path.abspath("..")+"\\RandomForest"
sys.path.append(dirpath)
#import DTclass as dtclass

from guRandomForest import classify,splitData,genRandomForest,classifyifyweakfilter
def findfault(data,weakfilter,tesyfunc=classify):
    TP=0.0
    faultloca=[]
    correctloca=[]
    for row in xrange(len(data)):
        oneresult=classify(data[row],weakfilter)
        if oneresult == data[row][-1]:
            TP+=1
            correctloca.append(row)
        else:
            faultloca.append(row)        
    accu = TP/len(data)
    return (accu,correctloca,faultloca,len(data)-TP)

def weakresult(testdata,weakmodel):
    result=[classifyifyweakfilter(testdata.values.tolist(),weakfilter) for weakfilter in weakmodel]
    resultfunc=lambda x: "weakfilter{} accu is : {}".format(x[0],x[1])
    weakresult=[resultfunc(lis) for lis in zip([i+1 for i in xrange(len(result))],result)]
    print weakresult
    return result


def findbestfilter(traindata,weakfilters,weight,filterlist,count):
#    least_wrong= float('inf')
    currentweight=0.0
    correctloca=[]
    faultloca=[]
    bestcorrectloca=[]
    bestfaultloca=[]
    bestweight=float('inf')
    bestnum=-1
#    bestFT=0
    bestaccu=0.0
    for weakfilternum in xrange(len(weakfilters)):
        (accu,correctloca,faultloca,FT)=findfault(traindata,weakfilters[weakfilternum])
        currentweight=sum([weight[i]for i in faultloca])
#        print "currentweight::",currentweight
#        print "accu",accu
        if currentweight < bestweight:
            bestnum=weakfilternum
            bestcorrectloca=correctloca
            bestfaultloca=faultloca
            bestfilter=weakfilters[weakfilternum]
            bestweight=currentweight
            bestaccu=accu
        elif currentweight == bestweight:
            equalnum=random.choice([weakfilternum,bestnum])
            bestfilter=weakfilters[equalnum]
            bestcorrectloca=correctloca
            bestfaultloca=faultloca
            bestweight=currentweight
            bestaccu=currentweight    
    filterlist.append(bestfilter)
    count.append(0.5*math.log((1-bestaccu)/bestaccu))
    return (bestfilter,bestcorrectloca,bestfaultloca,0.5*math.log(bestaccu/(1-bestaccu)))
#def calelpha(accu,dataset,weakmodel):
#    resultlist=[classify(data,weakmodel) for data in dataset]
##    resultlist=weakresult(dataset,weakmodel)
#    weight=[math.exp(-accu*dataset[i][-1]*resultlist[i])for i in xrange(len(resultlist))]
#    weightnorm=[weight[i]/sum(weight) for i in xrange(len(weight))]
#    return weightnorm
#def updatefunc(dataset,num,bestcorrectloca,bestfaultloca,bestaccu,weakmodel):
#    if num in bestcorrectloca:
#        return calelpha(bestaccu,dataset,weakmodel)
#    elif num in bestfaultloca:
#        return calelpha(-bestaccu,dataset,weakmodel)
#    else:
#        raise("Wrong number")

def updatelist(accu,dataset,correctloca,faultloca,weakmodel):    
    resultlist=[classify(data,weakmodel) for data in dataset] 
    accufunc= lambda x,i:x if i in correctloca else -x
    weight=[math.exp(-accufunc(accu,m)*dataset[m][-1]*resultlist[m])for m in xrange(len(resultlist))] 
    weightnorm=[weight[t]/sum(weight) for t in xrange(len(weight))]
    return weightnorm

def updateweight(accu,dataset,weight,correctloca,faultloca,weakmodel):
#    updatefunc=lambda x:x in correctloca
    listlist=updatelist(accu,dataset,correctloca,faultloca,weakmodel)
    nextweight=[listlist[i]*weight[i] for i in xrange(len(weight)) ]
    return nextweight
def genadaboost(traindata,weakfilters):
    initweight=[float(1.0/len(traindata)) for i in xrange(len(traindata))]
#    print "initweight",initweight
    filterlist=[]
    count=[]
    for filt in xrange(len(weakfilters)):#循环的次数如何确定？？？
        
        (bestfilter,correctloca,faultloca,accu)=findbestfilter(traindata,
                                                            weakmodel,
                                                            initweight,
                                                            filterlist,count)
#        print "current accu",accu
        nextweight=updateweight(accu,train,initweight,correctloca,faultloca,bestfilter)
        initweight=nextweight
#        print "the nextweight is ::::",nextweight
    
    return (filterlist,count)

if __name__=="__main__":    
    filepath= os.path.abspath("..")+"\\RandomForest\\wine.txt"
    df = pd.read_csv(filepath, header=None) 
    (traindata,testdata)=splitData(df,splitrate=0.3)
    weakmodel=genRandomForest(traindata,treenum=4,rowrate=0.8,colrate=0.4,maxdepth=5)
    weakresult(testdata,weakmodel)
#    (s1,s2,s3,s4)=findbestfilter(traindata.values.tolist(),
#                                 weakmodel,
#                                 [0.1 for  i in xrange(len(traindata.values.tolist()))],
#                                 [])
    train=traindata.values.tolist()
#    initweight=[float(1.0/len(train)) for i in xrange(len(train))]
#    
#    filterlist=[]
#    
#    (bestfilter,correctloca,faultloca,accu)=findbestfilter(train,
#                                                    weakmodel,
#                                                    initweight,
#                                                    filterlist)
#    nextweight=updateweight(accu,train,initweight,correctloca,faultloca,weakmodel[1])
    (modellist,weight)=genadaboost(train,weakmodel)
    
#    calelpha(s4,traindata.values.tolist(),weakmodel[1])
#    classifyifyweakfilter(testdata.values.tolist(),weakmodel[1])
    
#    tree = buildtree(traindata) 
    
#    print "The accuarcy is : ",accu
#    drawtree(tree,"beforeprune.jpg")
#    prune(tree,0.25)
#    drawtree(tree,"afterprune.jpg")
#    accuprune=accuracy(testdata,tree)
#    print "The accuarcy is :",accuprune

'''
adaboost是一个二分类问题 ！ 如果使用多分类 则需要使用OVA或者OVO 

'''











