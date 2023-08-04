from graphviz import Digraph

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def plot_tree(node, graph=None):
    if graph is None:
        graph = Digraph()
        graph.node(name=str(id(node)), label=str(node.value))
    
    if node.left:
        graph.node(name=str(id(node.left)), label=str(node.left.value))
        graph.edge(str(id(node)), str(id(node.left)))
        plot_tree(node.left, graph=graph)
        
    if node.right:
        graph.node(name=str(id(node.right)), label=str(node.right.value))
        graph.edge(str(id(node)), str(id(node.right)))
        plot_tree(node.right, graph=graph)
    
    return graph

# Construct the binary tree corresponding to the logical expression
root = Node('AND',
            Node('AND',
                 Node('AND',
                      Node('AND',
                           Node('OR',
                                Node('AND',
                                     Node('OR',
                                          Node('==', Node('r'), Node('c')),
                                          Node('==', Node('r'), Node('b'))),
                                     Node('>=', Node('r'), Node('b'))),
                                Node('==', Node('r'), Node('a'))),
                           Node('>=', Node('r'), Node('b'))),
                      Node('AND',
                           Node('OR',
                                Node('==', Node('r'), Node('d')),
                                Node('>=', Node('r'), Node('d'))),
                           Node('>=', Node('r'), Node('d')))),
                 Node('AND',
                      Node('>=', Node('r'), Node('b')),
                      Node('==', Node('r'), Node('d')))),
            Node('AND',
                 Node('>=', Node('r'), Node('b')),
                 Node('AND',
                      Node('>=', Node('r'), Node('a')),
                      Node('>=', Node('r'), Node('c')))))

# Plot and save the tree
tree = plot_tree(root)
tree.format = 'png'
tree.render('binary_tree')
