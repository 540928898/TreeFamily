# -*- coding: utf-8 -*-

'''
Author: GuPeng

Time:2019/3/4

Theme: Regress

'''

import pandas as pd
import numpy as np
#import copy
import random
import math
import os
print os.path.abspath("..")
from PIL import Image,ImageDraw
def getwidth(tree):
  try: 
      if tree.tb==None and tree.fb==None: return 1
      return getwidth(tree.tb)+getwidth(tree.fb)      
  except AttributeError:
      return 0
#  if tree.tb==None and tree.fb==None: return 1
#  return getwidth(tree.tb)+getwidth(tree.fb)
def strtofloat(dataset):
    datatmp=[]
    for data in dataset:
        result = map(eval, data)
        datatmp.append(result)
    return datatmp

def getdepth(tree):
  try: 
      if tree.tb==None and tree.fb==None: return 0
      return max(getdepth(tree.tb),getdepth(tree.fb))+1
  except AttributeError:
      return 0
#  if tree.tb==None and tree.fb==None: return 0
#  return max(getdepth(tree.tb),getdepth(tree.fb))+1

def drawtree(tree,jpeg='tree.jpg'):
  w=getwidth(tree)*100
  h=getdepth(tree)*100+120

  img=Image.new('RGB',(w,h),(255,255,255))
  draw=ImageDraw.Draw(img)

  drawnode(draw,tree,w/2,20)
  img.save(jpeg,'JPEG')
  
def drawnode(draw,tree,x,y):
  try:
      if tree.result==None:
        # Get the width of each branch
        w1=getwidth(tree.fb)*100
        w2=getwidth(tree.tb)*100
    
        # Determine the total space required by this node
        left=x-(w1+w2)/2
        right=x+(w1+w2)/2
    
        # Draw the condition string
        draw.text((x-20,y-10),str(tree.col)+':'+str(tree.value),(0,0,0))
    
        # Draw links to the branches
        draw.line((x,y,left+w1/2,y+100),fill=(255,0,0))
        draw.line((x,y,right-w2/2,y+100),fill=(255,0,0))
        
        # Draw the branch nodes
        drawnode(draw,tree.fb,left+w1/2,y+100)
        drawnode(draw,tree.tb,right-w2/2,y+100)
    
      else:
#        txt=' \n'.join(['%s:%d'%v for v in tree.result.items()])
        txt=' \n'.join(['result:%d'%tree.result])
        draw.text((x-20,y),txt,(0,0,0))
  except  AttributeError:
      print "There is no branch next"    
class Decisionnode:
    def __init__(self,col=-1,value=None,result=None,tb=None,fb=None,depth=-1):
        #col 待检验的判断条件
        #value：为了使结果为True所需要的条件
        #tb fb为左右节点
        #results保存当前分支的结果  
        self.col=col
        self.value=value 
        self.result=result 
        self.tb=tb
        self.fb=fb
        self.depth=depth 
        
def splitData(dataset,splitrate=0.3):
    '''split dataset to traindata : testdata=1-splitrate:splitrate'''
    row, colus = dataset.shape
    testlist=[random.randint(1,row-1) for i in xrange(int(row*splitrate))]
    trainlist=[i for i in xrange(1,row-1) if i not in testlist ]
    traindata=dataset.iloc[trainlist]
    testdata=dataset.iloc[testlist]
    print "Traindata's len is :{} \nTestdata's len is:{}".format(len(traindata.values.tolist()),
                               len(testdata.values.tolist()))
    return (traindata,testdata)

def counteachtype(rows,num):
    '''count each type in rows[num] and return a dic '''
    typelist={}
    for row in rows:
        if row[num] not in typelist:
            typelist[row[num]]=1
        else:
            typelist[row[num]] +=1
    return typelist

#类似于counteachtype
def uniquecount(rows,num=-1):
    results={}
    for row in rows:
        r=row[num]
        if r not in results:results[r]=0
        results[r]+=1
    return results
#如果最后结果有多个结果，返回数量最多的那一个
    
def major(dict_data):
    sorteddic=sorted(dict_data.items(),key=lambda x:x[1])
#    sorteddic=dic.sort(dic,lambda d:d[1])
    return sorteddic[-1][0]

def MSE(rows):
    datamean = float(np.mean(rows))
    R2 = sum([(x - datamean)**2 for x in rows])
    return R2

def divideSet(rows,column,value):
    '''split data base on  column and value，return (rightTreeData,leftTreeData)'''
    split_function=None
    #判断是否为数值型
    if isinstance(value,int) or isinstance(value,float):
    #lambda表达式 筛选数据
        split_function=lambda row: row[column] >= value
    else:
        split_function=lambda row: row[column] == value
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1,set2)

def divideSetLast(dataSet, featIndex, value):
    leftData, rightData = [], []
    for dt in dataSet:
        if dt[featIndex] < value:
            leftData.append(dt[-1])
        else:
            rightData.append(dt[-1])
    return leftData, rightData


def selectBestFeature(rows,scoref=MSE):
    best_gain = float('inf')
#    best_gain=0.0
    best_col=0
    best_iden=None
#    if len(rows[0])!=len(features):
#        print "No match !!!"
    for col in xrange(len(rows[0])-1):
        #统计每一列出现的特征
        eachtypenum=counteachtype(rows,col)
        #对于每一列出现的特征求基尼系数
        for iden in eachtypenum.keys():
            #依据特征划分数据
            (set1,set2)=divideSetLast(rows,col,iden)
            if len(set1)==0 and len(set2)!=0:
                gain=scoref(set2)
            if len(set1)!=0 and len(set2) ==0:
                gain=scoref(set1)
            if len(set1) !=0 and len(set2) !=0:
                gain=scoref(set1)+scoref(set2)
            #求总的概率
#            p=float(len(set1))/len(rows)
            #求基尼系数
#            nextscore =scoref(set1)*p+scoref(set2)*(1-p)
            #求增益
#            gain=currentscore-nextscore
            #找出最大的增益，记录最大增益，当前属性（哪一列），与当前特征
            if gain < best_gain and len(set1)>0 and len(set2)>0:
                best_gain=gain
                best_col=col
                best_iden=iden
    return (best_col,best_iden,best_gain)

def buildTree(rows,features,depth=0,scoref=MSE,maxdepth=8):    
    '''
    rows: dataset
    features: from bagging,we can use it to locate col
    depth: indicate tree depth
    scoref: loss.we can also use other loss
    maxdepth:control the tree's depth    
    '''
    if len(rows)==0: return Decisionnode()        
#    currentscore=scoref(rows)   
    resultend=np.mean([dt[-1] for dt in rows])
#    print resultend
    best_gain=0.0
    best_col=0
    best_iden=None
    (best_col, best_iden,best_gain)=selectBestFeature(rows,scoref=MSE)
    if depth <=maxdepth:
    #分裂子树
        nextdep=depth+1
        if best_gain>0:  
            #依据最大增益时的属性和列，划分数据            
            (setnext1,setnext2)=divideSet(rows,best_col,best_iden)
            if len(setnext1) ==0 or len(setnext2)==0:
#                result=np.mean([dt[-1] for dt in setnext2] + [dt[-1] for dt in setnext1])
                return Decisionnode(result=resultend,depth=nextdep)
            #生成右树
            trueBranch=buildTree(setnext1,features,depth=nextdep)
            #生成左树
            falseBranch=buildTree(setnext2,features,depth=nextdep)
            return Decisionnode(col=features[best_col],value=best_iden,
                                tb=trueBranch,fb=falseBranch,depth=nextdep)    
        else:
            return Decisionnode(result=resultend,depth=nextdep)
    else:
        return Decisionnode(result=resultend,depth=depth)
    
def classify(observation,tree):
    if tree.result !=None:
        return tree.result
    else:
        real_iden=observation[tree.col]
        branch=None
        if isinstance(real_iden,int) or isinstance(real_iden,float):
            if real_iden >= tree.value: branch=tree.tb
            else: branch=tree.fb
        else:
            if real_iden == tree.value: branch = tree.tb
            else: branch = tree.fb
        return classify(observation,branch)
    
def accuracy(testdata,tree,miu):
    TP=0.0
    for i in xrange(len(testdata)):       
        result=classify(testdata[i],tree)
#        print result," the true case is :",testdata[i][-1]
        if result-testdata[i][-1]<=miu:
            TP += 1
    accu=TP/len(testdata)
    return  accu


# 建立决策树
#def regressionTree(dataSet, features):
#    classList = [dt[-1] for dt in dataSet]
#    # label一样，全部分到一边
#    if classList.count(classList[0]) == len(classList):
#        return classList[0]
#    # 最后一个特征还不能把所有样本分到一边，则划分到平均值
#    if len(features) == 1:
#        return np.mean(classList)
#    bestFeatureIndex, bestSplitValue = chooseBestFeature(dataSet)
#    bestFeature = features[bestFeatureIndex]
#    # 删除root特征，生成新的去掉root特征的数据集
#    newFeatures, leftData, rightData = splitData(dataSet, bestFeatureIndex, features, bestSplitValue)
#
#    # 左右子树有一个为空，则返回该节点下样本均值
#    if len(leftData) == 0 or len(rightData) == 0:
#        return np.mean([dt[-1] for dt in leftData] + [dt[-1] for dt in rightData])
#    else:
#        # 左右子树不为空，则继续分裂
#        myTree = {bestFeature: {'<' + str(bestSplitValue): {}, '>' + str(bestSplitValue): {}}}
#        myTree[bestFeature]['<' + str(bestSplitValue)] = regressionTree(leftData, newFeatures)
#        myTree[bestFeature]['>' + str(bestSplitValue)] = regressionTree(rightData, newFeatures)
#    return myTree

def testHousing(df):
    df = pd.read_csv('housing.txt')
    return df

if __name__=="__main__":   
    df = pd.read_csv("housing.txt", header=None)[1:]   
#    dff=df.values.tolist()
    labels = df.columns.values.tolist()
    (traindata,testdata)=splitData(df,splitrate=0.3)
    trainlabel=traindata.columns.values.tolist()
    train=strtofloat(traindata.values.tolist())
    testlabel=testdata.columns.values.tolist()
    test=strtofloat(testdata.values.tolist())
    tree=buildTree(train,trainlabel,maxdepth=8)
    acu=accuracy(test,tree,miu=1)
    print acu
#    drawtree(tree,"tree3.jpg")
        
        
    
#    model=genRandomForest(traindata,treenum=10,rowrate=0.8,colrate=0.4,maxdepth=8)
#    print "the accuarcy is :{}".format(testRandomForest(testdata,model))
