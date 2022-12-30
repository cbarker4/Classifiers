import numpy as np

from mysklearn.myclassifiers import MyDecisionTreeClassifier

# TODO: copy your test_myclassifiers.py solution from PA4-6 here



from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.mytree import myTree
ipone = [[2,2,"fair"],[1,1,"excellent"]]
b_tes = [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]
header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
X_train_degrees = [
    ['A', 'B', 'A', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'A'],
    ['A', 'A', 'A', 'B', 'B'],
    ['B', 'A', 'A', 'B', 'B'],
    ['A', 'A', 'B', 'B', 'A'],
    ['B', 'A', 'A', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B'],
    ['A', 'A', 'A', 'A', 'A'],
    ['B', 'A', 'A', 'B', 'B'],
    ['B', 'A', 'A', 'B', 'B'],
    ['A', 'B', 'B', 'A', 'B'],
    ['B', 'B', 'B', 'B', 'A'],
    ['A', 'A', 'B', 'A', 'B'],
    ['B', 'B', 'B', 'B', 'A'],
    ['A', 'A', 'B', 'B', 'B'],
    ['B', 'B', 'B', 'B', 'B'],
    ['A', 'A', 'B', 'A', 'A'],
    ['B', 'B', 'B', 'A', 'A'],
    ['B', 'B', 'A', 'A', 'B'],
    ['B', 'B', 'B', 'B', 'A'],
    ['B', 'A', 'B', 'A', 'B'],
    ['A', 'B', 'B', 'B', 'A'],
    ['A', 'B', 'A', 'B', 'B'],
    ['B', 'A', 'B', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B']
]
y_train_degrees = ['SECOND', 'FIRST', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                   'SECOND', 'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND',
                   'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND', 'FIRST',
                   'SECOND', 'SECOND', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                   'SECOND', 'SECOND']
X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]



header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]


def test_decision_tree_classifier_fit():
    mT = MyDecisionTreeClassifier()
    mT.fit(X_train,y_train)
   
    print("out",mT.tree_list())
    assert mT.tree_list() == [['Senior', 0], ['no', 2], 'False', ['yes', 2], 'True', ['Mid', 0], 'True', ['Junior', 0], ['no', 3], 'True', ['yes', 3], 'False']
    mT = MyDecisionTreeClassifier()
    mT.fit(X_train_degrees,y_train_degrees)
    print("Here",mT.tree_list())
    assert mT.tree_list() == [['A', 0], ['B', 4], ['B', 3], 'SECOND', ['A', 3], ['B', 1], 'SECOND', ['A', 1], 'FIRST', ['A', 4], 'FIRST', ['B', 0], 'SECOND']

    mT = MyDecisionTreeClassifier()
    mT.fit(X_train_iphone,y_train_iphone)
    assert mT.tree_list() == [[3, 1], 'yes', [2, 1], 'yes', [1, 1], 'yes']



def test_decision_tree_classifier_predict():
    mT = MyDecisionTreeClassifier()
    mT.fit(X_train,y_train)
    assert mT.predict([["Junior", "Java", "yes", "yes"],["Junior", "Java", "yes", "no"]]) == [['False'], ['True']]
    mT = MyDecisionTreeClassifier()
    mT.fit(X_train_degrees,y_train_degrees)
    assert mT.predict(b_tes) == [['SECOND'], ['FIRST'], ['FIRST']]
    mT = MyDecisionTreeClassifier()
    mT.fit(X_train_iphone,y_train_iphone)
    assert mT.predict(ipone) == ['yes', 'yes']




test_decision_tree_classifier_predict()