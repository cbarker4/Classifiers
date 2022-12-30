class tree_node:
    def __init__(self):
        self.children = []
        self.parent =None
        self.val = ""
    def addNode(self,node):
        self.children.append(node)
    def __str__(self):
        for child in self.children:
            print(child.val)
    

class myTree:
    def __init__(self):
        self.root = tree_node()
   




    
        
        
