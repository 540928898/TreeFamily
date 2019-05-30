# -*- coding: utf-8 -*-

'''
Author: GuPeng

Time:2019/2/23

Theme: Class DT ID3 C4.5 CART

'''
import pandas as pd 
#import numpy as np
#import copy
import random
#import math

#                                                                   #
#                                                                   #
#                                                                   #
#                              数据                                  
#                                                                   #
#                                                                   #
#                                                                   #  
def datesplit(dataname="wine.txt"):
    df = pd.read_csv(dataname, header=None)
#    df.sample(n=30)
#    labels = df.columns.values.tolist()    
    dftolist=df.values.tolist()
#    df = df[df[labels[-1]] != 3]
    m,n =df.shape
    testlist=[random.randint(0,m-1) for i in xrange(int(m*0.3))]
    trainlist=[i for i in xrange(m-1) if i not in testlist ]
    testdata=[dftolist[i] for i in testlist]    
    traindata=[dftolist[i] for i in trainlist]
    print "Train Vol:",len(traindata)
    print "Test Val: ",len(testdata)
    return (trainlist,testlist,traindata,testdata)

#                                                                   #
#                                                                   #
#                                                                   #
#                             构造决策树                                  
#                                                                   #
#                                                                   #
#                                                                   #  
class decisionnode:
    def __init__(self,col=-1,value=None,result=None,tb=None,fb=None):
        #col 待检验的判断条件
        #value：为了使结果为True所需要的条件
        #tb fb为左右节点
        #results保存当前分支的结果 
        self.col=col
        self.value=value
        self.result=result
        self.tb=tb
        self.fb=fb
#将每一列按照某一值得标准分类 
def divideset(rows,column,value):
    split_function=None
    if isinstance(value,int) or isinstance(value,float):
        split_function=lambda row: row[column] >= value
    else:
        split_function=lambda row: row[column] == value
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1,set2)
'''
divideset(my_data,2,"yes")

Result:
    
  ([['slashdot', 'USA',         'yes', 18, 'None'],
  ['google', 'France',          'yes', 23, 'Premium'],
  ['digg', 'USA',               'yes', 24, 'Basic'],
  ['kiwitobes', 'France',       'yes', 23, 'Basic'],
  ['slashdot', 'France',        'yes', 19, 'None'],
  ['digg', 'New Zealand',       'yes', 12, 'Basic'],
  ['google', 'UK',              'yes', 18, 'Basic'],
  ['kiwitobes', 'France',       'yes', 19, 'Basic']],
    
    
 [['google', 'UK',              'no', 21, 'Premium'],
  ['(direct)', 'New Zealand',   'no', 12, 'None'],
  ['(direct)', 'UK',            'no', 21, 'Basic'],
  ['google', 'USA',             'no', 24, 'Premium'],
  ['digg', 'USA',               'no', 18, 'None'],
  ['google', 'UK',              'no', 18, 'None'],
  ['kiwitobes', 'UK',           'no', 19, 'None'],
  ['slashdot', 'UK',            'no', 21, 'None']])

'''
#计数
def uniquecount(rows,num=-1):
    results={}
    for row in rows:
        r=row[num]
        if r not in results:results[r]=0
        results[r]+=1
    return results
'''
uniquecount(my_data)
{'Basic': 6, 'None': 7, 'Premium': 3}

'''  
#基尼系数
def ginicoefficient(rows) :
    total=len(rows)
    countresult=uniquecount(rows)
    gini=0
    for num in countresult:
        p1=float(countresult[num])/total
        gini += p1*(1-p1)        
    return gini 
 #基尼不纯度 == 基尼系数
def giniimpurity(rows):
    total=len(rows)
    countresult=uniquecount(rows)
    giniimp=0
    for num in countresult:
        p1=float(countresult[num])/total
        for num2 in countresult:
            if num==num2: continue
            p2=float(countresult[num2])/total
            giniimp += p1*p2
    return giniimp
#熵  
def entropy(rows):
    from math import log
    log2=lambda x:log(x)/log(2)
    entro=0
    total=len(rows)
    countresult = uniquecount(rows)
    for name in countresult:
        pi=float(countresult[name])/total
        entro =entro - pi*log2(pi)
    return entro


  
#统计每一列（每个属性）出现的特征与次数    
def counteachtype(rows,num):
    typelist={}
    for row in rows:
        if row[num] not in typelist:
            typelist[row[num]]=1
        else:
            typelist[row[num]] +=1
    return typelist
#构造决策树
def buildtree(rows,scoref=entropy):
    if len(rows)==0: return decisionnode()
    currentscore=scoref(rows)
    best_gain=0.0
    best_col=0
    best_iden=None
    for col in xrange(len(rows[0])-1):
        #统计每一列出现的特征
        eachtypenum=counteachtype(rows,col)
        #对于每一列出现的特征求基尼系数
        for iden in eachtypenum.keys():
            #依据特征划分数据
            (set1,set2)=divideset(rows,col,iden)
            #求总的概率
            p=float(len(set1))/len(rows)
            #求基尼系数
            nextscore =scoref(set1)*p+scoref(set2)*(1-p)
            #求增益
            gain=currentscore-nextscore
            #找出最大的增益，记录最大增益，当前属性（哪一列），与当前特征
            if gain > best_gain and len(set1)>0 and len(set2)>0:
                best_gain=gain
                best_col=col
                best_iden=iden
    #分裂子树
    if best_gain>0:  
        #依据最大增益时的属性和列，划分数据
        (setnext1,setnext2)=divideset(rows,best_col,best_iden)
        #生成右树
        trueBranch=buildtree(setnext1)
        #生成左树
        falseBranch=buildtree(setnext2)
        return decisionnode(col=best_col,value=best_iden,
                            tb=trueBranch,fb=falseBranch)    
    else:
        return decisionnode(result=uniquecount(rows))

 


#                                                                   #
#                                                                   #
#                                                                   #
#                              绘图                                  
#                                                                   #
#                                                                   #
#                                                                   #      
def printtree(tree,indent=''):
    if tree.result != None:
        print str(tree.result)
    else:
        #打印判断条件
        print str(tree.col)+':'+str(tree.value)+'?'
        print indent +'T->'
        printtree(tree.tb,indent=' ')
        print indent +'F->'
        printtree(tree.fb,indent=' ')
'''
            0:google?
            T->
            3:21?
             T->
            {'Premium': 3}
             F->
            2:yes?
             T->
            {'Basic': 1}
             F->
            {'None': 1}
            F->
            0:slashdot?
             T->
            {'None': 3}
             F->
            2:yes?
             T->
            {'Basic': 4}
             F->
            3:21?
             T->
            {'Basic': 1}
             F->
            {'None': 3}   
''' 
def getwidth(tree):
  try: 
      if tree.tb==None and tree.fb==None: return 1
      return getwidth(tree.tb)+getwidth(tree.fb)      
  except AttributeError:
      return 0
#  if tree.tb==None and tree.fb==None: return 1
#  return getwidth(tree.tb)+getwidth(tree.fb)

def getdepth(tree):
  try: 
      if tree.tb==None and tree.fb==None: return 0
      return max(getdepth(tree.tb),getdepth(tree.fb))+1
  except AttributeError:
      return 0
#  if tree.tb==None and tree.fb==None: return 0
#  return max(getdepth(tree.tb),getdepth(tree.fb))+1


from PIL import Image,ImageDraw

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
        txt=' \n'.join(['%s:%d'%v for v in tree.result.items()])
        draw.text((x-20,y),txt,(0,0,0))
  except  AttributeError:
      print "There is no branch next"

#                                                                   #
#                                                                   #
#                                                                   #
#                              测试                                  
#                                                                   #
#                                                                   #
#                                                                   #                  

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
def accuracy(testdata,tree):
    TP=0.0
    for i in xrange(len(testdata)):       
        result=classify(testdata[i],tree)
#        print result," the true case is :",testdata[i][-1]
        if int(result.keys()[0])==testdata[i][-1]:
            TP += 1
    accu=TP/len(testdata)
    return  accu
                      
        
    

#                                                                   #
#                                                                   #
#                                                                   #
#                              剪枝                                  
#                                                                   #
#                                                                   #
#                                                                   # 

def prune(tree,mingain,scoref=giniimpurity):
  # If the branches aren't leaves, then prune them
  if tree.tb.result==None:
    prune(tree.tb,mingain)
  if tree.fb.result==None:
    prune(tree.fb,mingain)
    
  # If both the subbranches are now leaves, see if they
  # should merged
  if tree.tb.result!=None and tree.fb.result!=None:
    # Build a combined dataset
    tb,fb=[],[]
    for v,c in tree.tb.result.items():
      tb+=[[v]]*c
    for v,c in tree.fb.result.items():
      fb+=[[v]]*c
    
    # Test the reduction in entropy
    delta=scoref(tb+fb)-(scoref(tb)+scoref(fb)/2)
    if delta<mingain:
      # Merge the branches
      tree.tb,tree.fb=None,None
      tree.result=uniquecount(tb+fb)    
    
    
if __name__=="__main__":    
    (trainlist,testlist,traindata,testdata)=datesplit()
    tree = buildtree(traindata) 
    accu=accuracy(testdata,tree)
    print "The accuarcy is : ",accu
    drawtree(tree,"beforeprune.jpg")
    prune(tree,0.25)
    drawtree(tree,"afterprune.jpg")
    accuprune=accuracy(testdata,tree)
    print "The accuarcy is :",accuprune
    
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
