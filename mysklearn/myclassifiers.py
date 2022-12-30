

from mysklearn import myutils
from mysklearn.mytree import myTree

import copy

from mysklearn import mypytable as MYT
from mysklearn import myutils
import random

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).
    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data
    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.
        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """

        temp = []
        for val in y_train:
            temp.append(self.discretizer(val))
        y_train = temp
        if self.regressor == None:
            self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)

        
        

        pass 

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        temp  = self.regressor.predict(X_test)
        return temp 

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test,All_neibors = False):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
            All_neibors(Boolean): 
        Returns:
            distances(list of list of list of type float): the first set of lists 
                is the closest neibors for that location in X_test in that list are there
                distances and the last number is the index of the neibor in x_train
        """
        distance = []
        for val in X_test:
            i = 0
            temp = []
            for neibor in self.X_train:
                temp.append([myutils.compute_euclidean_distance(val,neibor),i])
                i = i+1
            distance.append(copy.deepcopy(temp))
        i =0 
        for list in distance:
            distance[i] = sorted(list)
            i = i + 1
        limited = []
        if All_neibors == True:
            return distance 
        for val in distance:
            temp = []
            i =0 
            for close in val:
                if i < self.n_neighbors:
                    temp.append(close)
                i = i + 1
            limited.append(temp)
        return limited
            
                
        
        

    def predict(self, X_test,randomNeibor = False):
        
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        answer = []
        
        big_array = self.kneighbors(X_test)
        
       
        for val in big_array:
            neighbor_results = []
            for neighbors in val:
                neighbor_results.append(self.y_train[neighbors[-1]])
           
            values, col = myutils.get_frequencies_str(neighbor_results)
            large =0
            ind = 0
            i =0 
            for val in col:
                if val > large:
                    large = val 
                    ind = i
                i = i + 1
            if randomNeibor == True:
                answer.append(values[random.randint(0,ind)])
            else:
                answer.append(values[ind])
    


        return answer

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        
        values, col = myutils.get_frequencies_str(y_train)
        large =0
        ind = 0
        i =0 
        for val in col:
            if val > large:
                large = val 
                ind = i
            i = i + 1

        self.most_common_label = values[ind]

        pass 

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        out = []
        for i in X_test:
            out.append(self.most_common_label)
        return out

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = {}
        self.posteriors = {}
        self.outer_keys=[]
    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        
        outter_keys = []
        keys_data = []
        for val in y_train:
            if not val in outter_keys:
                outter_keys.append(val)
        for val in outter_keys:
            keys_data.append(y_train.count(val))
        for i,val in enumerate(outter_keys):
            self.priors[val] = keys_data[i]/len(y_train)
        inner = {}
        for val in outter_keys:
            self.posteriors[val]= None
        mt = MYT.MyPyTable()
        mt.data = copy.deepcopy(X_train)
        #mt.add_column(y_train)
        
        #{y_train{atrribute{val}}}
        attribute = {}
        middle_keys = []
        for atri in range(len(mt.data[0])):
            if not atri in attribute:
                attribute[atri] = None
                middle_keys.append(atri)
            
        for key in outter_keys:
            self.posteriors[key] = copy.deepcopy(attribute)
        self.outer_keys = outter_keys
        for outer in outter_keys:
            mt.data = copy.deepcopy(X_train)
            for mid in middle_keys:
                inner = {}
                in_key = []          
                col = mt.get_column(0,delete_col=True)
                for i,val in enumerate(col):
                    if outer == y_train[i]:
                        if not val in inner:
                            inner[val] = 1
                            in_key.append(val)
                        else :
                            inner[val] = inner[val] + 1 
                    if i == len(col)-1:                   
                        for other in in_key:
                            inner[other] = inner[other]/(self.priors[outer]*len(y_train))
                self.posteriors[outer][mid] = inner




        pass

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        anser = []
        
        for x in X_test:
            odds = []
            for key in self.outer_keys:
                other = []
                for i,val in enumerate(x):
                    try:
                        other.append(self.posteriors[key][i][val])
                    except:
                        other.append(0)
                odds.append(myutils.prod(other))
            anser.append(odds)

      
        
        real_answer = []
        for val in anser:
           real_answer.append(self.outer_keys[val.index(max(val))])
        return real_answer

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = myTree()

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        train = []
        for val in y_train:
            if not val in train:
                train.append(val)
        mt = MYT.MyPyTable()
        mt.data = X_train
        mt.add_column(y_train)
        
        myutils.tree_help(mt,self.tree.root)
   
           
            
        


        pass # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        ans =[]
        for instance in X_test:
            
            ans.append(myutils.get_prediction_for_tree(self.tree.root,instance))
        return ans # TODO: fix this

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        for child in self.tree.root.children:
            name = ""
            print(myutils.get_stmts(child,name,attribute_names))
        pass # TODO: fix this

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
       
        pass # TODO: (BONUS) fix this

    def tree_list(self):
        list = []
        myutils.to_list(self.tree.root,list)
        return list
    
