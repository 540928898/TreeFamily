# -*- coding: utf-8 -*-
import pandas as pd
import random
from draw import drawtree
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
    testlist=[random.randint(0,row-1) for i in xrange(int(row*splitrate))]
    trainlist=[i for i in xrange(row-1) if i not in testlist ]
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

#将数据从dataframe改变为list
def bagging(dataset,rowrate=0.8,colrate=0.4):
    '''bagdata to create weaktree; return (list,list)'''
    row, colus = dataset.shape
    features = random.sample(dataset.columns.values[:-1], int(colus*colrate))
    features.append(dataset.columns.values[-1])
    rows = [random.randint(0, row-1) for i in range(int(row*rowrate))]
    bagdata=dataset.iloc[rows][features]
    m,n=bagdata.shape
#    print "Bagdata's len is :{} ".format(m)
#    print "Bagdate features' len:{}".format(n)    
    return (bagdata.values.tolist(),features)

#计算基尼不纯度 
def giniimpurity(rows):
    total=len(rows)
    countresult=uniquecount(rows)
    giniimp=0.0
    for num in countresult:
        p1=float(countresult[num])/total
        for num2 in countresult:
            if num==num2: continue
            p2=float(countresult[num2])/total
            giniimp += p1*p2
    return giniimp

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
def selectBestFeature(rows,scoref=giniimpurity):
    currentscore=scoref(rows)
    best_gain=0.0
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
            (set1,set2)=divideSet(rows,col,iden)
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
    return (best_col,best_iden,best_gain)


#features 是从bagging中传入，随机打乱了顺序，所以在最后生成树的时候需要知道是原数据中的哪一个col
def buildTree(rows,features,depth=0,scoref=giniimpurity,maxdepth=8):
    '''
    rows: dataset
    features: from bagging,we can use it to locate col
    depth: indicate tree depth
    scoref: loss.we can also use other loss
    maxdepth:control the tree's depth    
    '''
    if len(rows)==0: return Decisionnode()        
#    currentscore=scoref(rows)
    tmpresult=uniquecount(rows)    
    resultend=major(tmpresult)
    best_gain=0.0
    best_col=0
    best_iden=None
    (best_col, best_iden,best_gain)=selectBestFeature(rows,scoref=giniimpurity)
    if depth <=maxdepth:
    #分裂子树
        nextdep=depth+1
        if best_gain>0:  
            #依据最大增益时的属性和列，划分数据            
            (setnext1,setnext2)=divideSet(rows,best_col,best_iden)
            #生成右树
            trueBranch=buildTree(setnext1,features,depth=nextdep)
            #生成左树
            falseBranch=buildTree(setnext2,features,depth=nextdep)
            return Decisionnode(col=features[best_col],value=best_iden,
                                tb=trueBranch,fb=falseBranch,depth=nextdep)    
        else:
            return Decisionnode(result=resultend,depth=nextdep)
    else:
        return Decisionnode(result=resultend)
#剪枝算法不太对，剪枝后准确度下降，如何找到准确的准确度是一个问题。
#    todo：可以通过准确度剪枝吗？
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
    
    # Test the reduction in giniimpurity
    delta=giniimpurity(tb+fb)-(giniimpurity(tb)+giniimpurity(fb)/2)
    if delta<mingain:
      # Merge the branches
      tree.tb,tree.fb=None,None
      tree.result=uniquecount(tb+fb)        
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
    
def genRandomForest(dataset,treenum,selectfunc=bagging,rowrate=0.8,colrate=0.4,maxdepth=8):
    treenum=treenum
    treelist=[]
    for i in xrange(treenum):
        (datasetson,features)=bagging(dataset,rowrate,colrate)
        tree=buildTree(datasetson,features,maxdepth=8)
        treelist.append(tree)
    return treelist    

def testOneRandomForest(testdata,treelist,tesyfunc=classify):
    predictlabel=[]
    for tree in treelist:
        oneresult=classify(testdata,tree)
        predictlabel.append(oneresult)
    countdic={}
    for prelabel in predictlabel:
        if prelabel not in countdic:
            countdic[prelabel]=1
        else:
            countdic[prelabel]+=1
    #字典排序 .items() 为可迭代对象，d[0]为键排序，d[1]为值排序 返回一个数组
    sortlist = sorted(countdic.items(), key=lambda d: d[1])
    return sortlist[-1][0]
def classifyifyweakfilter(testdata,weakfilter,tesyfunc=classify):
    TP=0.0
    for row in xrange(len(testdata)):
        oneresult=classify(testdata,weakfilter)
        if oneresult == testdata[row][-1]:
            TP+=1
    result = TP/len(testdata)
#    print "weakfilter accu:",result
    return result
            
def testRandomForest(testdata,treelist):
    testdata=testdata.values.tolist()
    count=len(testdata)
    TP=0.0
    accu=0
    for i in xrange(count-1):
        testResult=testOneRandomForest(testdata[i],treelist)
        if testResult==testdata[i][-1]:
            TP+=1
    accu=TP/count
    return accu        
if __name__=="__main__":   
    df = pd.read_csv("wine.txt", header=None)    
    dff=df.values.tolist()
    labels = df.columns.values.tolist()
    (traindata,testdata)=splitData(df,splitrate=0.3)
    model=genRandomForest(traindata,treenum=10,rowrate=0.8,colrate=0.4,maxdepth=8)
    print "the accuarcy is :{}".format(testRandomForest(testdata,model))
'''
需要注意的点：
1.dataframe如何选择行和列： 
    1)traindata=dataset.iloc[trainlist]  
    2)df = df[df[labels[-1]] != 3] tip: df[]选择的是行

2.随机选择 features = random.sample(dataset.columns.values[:-1], int(colus*colrate))

3.lambda表达式 split_function=lambda row: row[column] >= value

4. 字典排序 sorted(countdic.items(), key=lambda d: d[1])

5.一开始树的生成总是会报错，原因就是对于末位节点的控制不到位。

'''
    
    