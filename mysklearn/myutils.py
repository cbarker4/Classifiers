from locale import normalize
from operator import index
from platform import node
import string
import numpy as np
import math
from prometheus_client import Counter
import copy
# some useful mysklearn package import statements and reloads
import importlib



# uncomment once you paste your myclassifiers.py into mysklearn package
import mysklearn.myclassifiers
importlib.reload(mysklearn.myclassifiers)
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier

import mysklearn.myevaluation
importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myevaluation

from mysklearn.mytree import tree_node
def disc(val):
    if val >=45:
        return 10
    elif 45 >= val >=37:
        return 9
    elif 37 > val >=31:
       return 8
    elif 31 > val >=27:
        return 7
    elif 27 > val >=24:
        return 6 
    elif 24 > val >=20:
        return 5 
    elif 20 > val >=17:
        return 4
    elif 17 > val >=15:
        return 3
    elif 15 > val >=14:
        return 2
    elif val<= 13:
        return 1

def compute_euclidean_distance(v1, v2):
    """ From Gina Sprint
    """
    for i,val in enumerate(v1):
        if type(val)==type("string"):
            if val == v2[i]:
                return 0 
            else:
                return 1
    
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))




def get_frequencies_str(col):
    temp = []
    for val in col:
        temp.append(str(val))
    col = temp
    col.sort() # inplace 
    # parallel lists
    values = []
    counts = []
    for value in col:
        if value in values: # seen it before
            counts[-1] += 1 # okay because sorted
        else: # haven't seen it before
            values.append(value)
            counts.append(1)

    return values, counts # we can return multiple values in python


def normalize(list,big=0):
    """Function normalizes a list 
        Args:
            list: a list 
    """
    #big = 0
    for val in list:
        if val >big:
            big = val
    temp = []

    for val in list:
        temp.append(val/big)
    return temp



def prod(temp):
    """ Takes the product of all values 
        Args:
            list: a list 
        Returns: single value
    """
    sum =1
    for val in temp:
        sum = val * sum
    return sum

def to_vals(arg,norm = False):
    change = {}
    count = 0
    for val in arg:
        if not val in change:
            change[val] = count
            count = count + 1
    temp = []
    for val in arg:
        temp.append(change[val])
    if norm == True:
       temp =  normalize(temp)
    arg = temp
    return temp 

def get_entropy(yes,no):
    if yes == 0 or no == 0 :
        return 0 
    E = -(yes * math.log(yes,2) + no * math.log(no,2))
    return E


def attribute_to_split_on(mt):

    train = []
    y_train = mt.get_column(-1)
    for val in y_train:
        if not val in train:
            train.append(val)

    attributes = len(mt.data[-1])
    attributes = attributes-1
  
    entropy =[]
    for trainval in train:
        for val in range(attributes):
            col = mt.get_column(val)
            values = []
            for var in col:
                if not var in values:
                    values.append(var)
            temp =[]
            for var in values:
               
                temp.append(mt.how_many_vals(val,var,last_column=trainval))
            entropy.append(temp)

    i=0
    size = 0
    for val in entropy:
        size = len(mt.data)
    list_entropy = []
    #attributes = attributes-
    while i < len(entropy)/2:
        j=0
        temp =[]
        
        while j < len(entropy[i]):

            yes = entropy[i][j]
         
            no = entropy[attributes][j]
         
            temp.append(get_entropy(yes/size,no/size))
            j= j+1
        list_entropy.append(temp)
        attributes = attributes + 1                                                                                    
        i = i +1
    start = [] 
    for val in list_entropy:
        start.append(sum(val)/len(val))   ## probally wrong but working 
    return start.index(min(start))



def tree_help (mt,node,atributes = []):
    
    if mt.data==[]:
        return
    split = attribute_to_split_on(mt)
  
    col  = mt.get_column(split)
    paths = []
    for val in col:
        if not val in paths :
            paths.append(val)
    if not split in atributes:
        atributes.append(split)
    for val in paths:
        
        nn = tree_node()
       
     
        # if len(node.children)<1:
        #     for temp3 in deleted: 
        #         if temp3 < split:
        #             temp2 = temp2+1
        #     nn.val = [val,temp2] 
        #     if not temp2 in deleted:
        #         deleted.append(temp2)
        # else:
        #     print("val",temp2)
        #     nn.val = [val,temp2] 
        nn.val = [val,split] 
        
       
        temp = copy.deepcopy(mt.get_rows_with_val(split,val))
        
        count = []
        counter =[]
        for row in temp.data:
            if not row[-1] in count:
                count.append(row[-1])
                counter.append(1)
            else:
                counter[count.index(row[-1])] = counter[count.index(row[-1])] + 1

        if len(count) == 1:
            hi =[]
            hi.append(count[0])
            child = tree_node()
            child.val = hi
            nn.children.append(child)
            node.children.append(nn)
        
        elif len(atributes)==len(temp.data[-1])+1:
          
            child = tree_node()
            if max(counter)!= min(counter):
                child.val = count[counter.index(max(counter))]
            else:
                count.sort()
                child.val = count[0]

            nn.children.append(child)
            node.children.append(nn) 
            
        
        else:
            #temp.drop_column(split)
            node.children.append(nn)
            tree_help(temp,nn,atributes=atributes) 
        



    
def to_list(node,list):
    for val in node.children:
        if val.val != None:
            list.append(val.val)
        to_list(val,list)



def get_prediction_for_tree(node,instance):
    if len(node.children)==1:
        return node.children[0].val
    for child in node.children:
        if child.val[0]== instance[child.val[1]]:
            return get_prediction_for_tree(child,instance)

def get_stmts(node,name,Atts=None,clas = None):
    if Atts is None:
        if len(node.children)==1:
            if clas== None:
                name = name +" IF " +"att"+ str(node.val[1]) +" == " + str(node.val[0]) + " THEN " + "Class = " + str(node.children[0].val[0])
            else:
                name = name +" IF " +"att"+ str(node.val[1]) +" == " + str(node.val[0]) + " THEN " + clas+" " + str(node.children[0].val[0])
        elif len(node.children)>1:     
            for child in node.children:
                name = name+" IF " + "att"+ str(node.val[1]) +" == "+ str(node.val[0]) + " AND "
                name = get_stmts(child,name) + "\n"
        else:
            name = name + Atts[node.val[1]] +"=="+ str(node.val[0])
    return name

def predictors(mt):
    

    y = mt.get_column(-1,delete_col= True)
    train,test = myevaluation.stratified_kfold_cross_validation(mt.data,y,n_splits = 5)
    bayacc = []
    knaccc =[]
    dumAcc = []
    treAcc = []

    X = mt
    for i,vals in enumerate(train):
        x_test = []
        x_train = []
        y_train = []
        y_act = []
        for j in vals:
            x_train.append(X.data[j])
            y_train.append(y[j])
        for j in test[i]:
            x_test.append(X.data[j])
            y_act.append(y[j])
        knn = MyKNeighborsClassifier(n_neighbors=5)
        dumb = MyDummyClassifier()
        bay = MyNaiveBayesClassifier()
        tree = MyDecisionTreeClassifier()
        dumb.fit(x_train,y_train)
        bay.fit(x_train,y_train)
        knn.fit(x_train,y_train)
        tree.fit(x_train,y_train)
        d =dumb.predict(x_test)
        b = bay.predict(x_test)
        k =knn.predict(x_test,randomNeibor = True)
        t = tree.predict(x_test)
        t2 =[]
        for val in t:
            try:
                t2.append(val[0])
            except:
                t2.append('H')
            
        treAcc.append(myevaluation.accuracy_score(t2,y_act))
        dumAcc.append(myevaluation.accuracy_score(d,y_act))
        bayacc.append(myevaluation.accuracy_score(b,y_act))
        knaccc.append(myevaluation.accuracy_score(k,y_act))
        

        treAcc = sum (treAcc)/len(treAcc)
        knaccc = sum(knaccc)/len(knaccc)
        dumAcc = sum(dumAcc)/len(dumAcc)
        bayacc = sum(bayacc)/len(bayacc)

        print("10-Fold Cross Validation")
        print("KNN clasifier: accuracy = ",knaccc,", error rate =",1-knaccc)
        print("Dummy Classifier: accuracy = ",dumAcc,", error rate =",1-dumAcc)
        print("Bays Classifier: accuracy = ",bayacc,", error rate =",1-bayacc)
        print("Decision Tree: accuracy = ",treAcc,", error rate =",1-treAcc)
        