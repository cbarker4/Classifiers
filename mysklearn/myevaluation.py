from mysklearn import myutils

from ast import Try
from cProfile import label
from mysklearn import mypytable, myutils
import random as rd
import numpy.random as srd
from mysklearn import mypytable as mt 
import copy

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!
    Returns:
         (list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_tests = []
    y_test = []
    if type(test_size)==type(0.0):
        How_many = len(X)
        How_many = int(How_many*test_size)+1
    else :
        How_many = test_size
    myT = mt.MyPyTable()
    myT.data = copy.deepcopy(X)
    if random_state != None:
        rd.seed(random_state)
    else :
        rd.seed(0)#len(X))
    
    y_copy = copy.deepcopy(y)
    if shuffle == True:
        myT.add_column(y_copy,"y-vals")
        myT.shuffle()
        y_copy = myT.get_column(-1)
        myT.drop_column(-1)

    
    i =1 
   
    while How_many > 0:
        loc = -1

        X_tests.insert(0,myT.get_row(loc, delete=True))
        y_test.insert(0,y_copy[loc])
        y_copy.pop(loc)

        How_many = How_many -1 
        i = i + 1

    return myT.data, X_tests, y_copy, y_test

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold
    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if shuffle==True:
        if random_state == None:
            rd.seed(56)
        else:
            rd.seed(random_state)

    test=[]
    train=[]
    extra =  len(X)%n_splits
    size = int((len(X)//n_splits)+1)
    if shuffle == True:
        temp = []
        for round in range(n_splits):
            i=0
            temp =[]
            if round == extra:
                size = size-1
            while i< size:
                num = rd.randint(0,len(X)-1)
                put = 1
                
                for val in test:
                    if num in val or num in temp:
                        put = 0
                if put ==1:
                    temp.append(num)
                    
                #extra = extra -1 
                i = i +1
                if put== 0:
                    i = i-1
                if not i < size:
                    test.append(temp)

        for val in test:
            i =0 
            other = []
            while i < len(X):
                if not i in val:
                    other.append(i)
                i = i + 1
            train.append(other)
    else:
        for round in range(n_splits):    
            i=0
            other = []
            if round == extra:
                size = size-1
            try:
                low = test[round-1][-1]
            except:
                low =-1
            while i < len(X):
                temp = []
                while low<i<=low+size:
                    if i<len(X):
                        temp.append(i)
                    i = i + 1              
                    if not low<i<=low+size:
                        test.append(temp)
                if i <len(X):
                    other.append(i)
                i = i +1
            train.append(other)

        


    return train, test # TODO: fix this

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.
    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    if random_state!= None:
        rd.seed(random_state)
    
    index = []
    test=[]
    train=[]
    for i in range(len(X)):
        index.append(i)
    myT = mt.MyPyTable()
    myT.data = copy.deepcopy(X)
    myT.add_column(index)
    myT.add_column(y)
    myT.sort_on_column(-1)
    ysort = myT.get_column(-1,delete_col=True)
    index = myT.get_column(-1,delete_col=True)
    temp = ysort[0]
    change =0
    for i in range(len(ysort)):
        if temp != ysort[i]:
            change = i
            break 

    extra =  len(X)%n_splits
    size = int((len(X)//n_splits)+1)

    temp = []
    up =0 
    down = 1
    switch =0
    num =0
    for round in range(n_splits):
        i=0
        temp =[]
        if round == extra:
            size = size-1
        while i< size:
            if switch == 0:
                if shuffle == True:
                    num = rd.randint(0,change) 
                else:
                    num = up
                    up = up +1
            else:
                if shuffle == True:
                    num = rd.randint(change,len(X)-1) 
                else:
                    num = len(X)-down
                    down = down+1 


            put = 1
            
            for val in test:
                if index[num] in val or index[num] in temp:
                    put = 0
            if put ==1:
                temp.append(index[num])
                 
            i = i +1
            if put== 0:
                i = i-1
            if not i < size:
                test.append(temp)
            if switch == 0:
                switch =1
            else:
                switch = 0
    i =0 

    for val in test:
        i =0 
        other = []
        while i < len(X):
            if not i in val:
                other.append(i)
            i = i + 1
        train.append(other)




    return train, test # TODO: fix this

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.
    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    x_sample =[]
    y_sample =[]
    x_bag = []
    y_bag = []
    if y==None:
        y_sample = None
        y_bag =None
  
    if random_state != None:
        srd.seed(random_state)

    if n_samples != None:
        temp = n_samples
    else:
        temp = len(X)
    for i in range(temp):
        i = srd.random_integers(0,len(X)-1)
        x_sample.append(X[i])
        if y!= None:
            y_sample.append(y[i])
    for i in range(len(X)):
        if not X[i] in x_sample and not X[i]in x_bag :
            if y!= None:
                y_bag.append(y[i])
            x_bag.append(X[i])
    return x_sample, x_bag, y_sample, y_bag 

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    for val in labels:
        temp =[]
        for i in range(len(labels)):
            temp.append(0)
        matrix.append(temp)
  
    for i,val in enumerate(y_true):
        try:
            val = str(val)            
            matrix[labels.index(val)][labels.index(y_pred[i])] = matrix[labels.index(val)][labels.index(y_pred[i])] + 1
        except:
            break
        

    return matrix # TODO: fix this

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).
    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    accuracy =0
    for i,val in enumerate(y_true):
        temp = str(y_pred[i])
        if val == temp:
            accuracy = accuracy + 1 
    if normalize == True:
        accuracy = accuracy/len(y_pred)
    return accuracy 

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        precision(float): Precision of the positive class
    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if pos_label == None:
        if label != None:
            pos_label = label[0]
        else:
            pos_label = y_true[0]
    num = 0
    accuracy = 0
    for i,val in enumerate(y_true):
        if y_pred[i] == pos_label:
            if val == pos_label:
                accuracy = accuracy+1
            else:
                num = num + 1 
    if num + accuracy != 0:
        #print ("A:",accuracy," N: ",num)
        accuracy = accuracy/(num+accuracy) 

    return accuracy

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        recall(float): Recall of the positive class
    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """

    if pos_label == None:
        if label != None:
            pos_label = label[0]
        else:
            pos_label = y_true[0]
    num = 0
    accuracy = 0
    for i,val in enumerate(y_true):
        if y_pred[i] == pos_label and val == pos_label: 
            accuracy = accuracy+1
        if y_pred[i] != pos_label and val == pos_label:
            num = num + 1 
    if num + accuracy != 0:
        #print ("A:",accuracy," N: ",num)
        accuracy = accuracy/(num+accuracy) 
    #print(accuracy)
    return accuracy
   
def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        f1(float): F1 score of the positive class
    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """

    recall = binary_recall_score(y_true,y_pred,labels=labels,pos_label=pos_label)
    precision = binary_precision_score(y_true,y_pred,labels=labels,pos_label=pos_label)
    if precision + recall == 0:
        return 0.0
    return  2 * (precision * recall) / (precision + recall)