import torch
import torch_geometric
import numpy as np


def generate_graph(data, k):
    graph1 = []
    for i in range(len(data)):
        idx1 = torch_geometric.nn.knn_graph(torch.FloatTensor(data[i]), k, None,True,'target_to_source')
        g1 = torch_geometric.utils.to_dense_adj(idx1)
        g1 = ((g1 + g1.transpose(2,1)) > 0).float()
        graph1.append(g1)
    graph1 = torch.cat(graph1,0).numpy()
    return graph1

if __name__=='__main__':
    #data = np.load('./sydney/sydney_1.npz')
    data = np.load('./modelnet/modelnet40_core2.npz')

    graph1 = []
    graph2 = []

    train = data['x_train']
    test  = data['x_test']
    '''
    for i in range(len(train)):
        print('train data %03d'%i)
        idx1 = torch_geometric.nn.knn_graph(torch.FloatTensor(train[i]),20  ,None,True,'target_to_source')
        #idx1 = torch_geometric.nn.radius_graph(torch.FloatTensor(train[i]),0.25,batch=None,loop=True,flow='target_to_source')
        print(len(idx1[0]))
        g1 = torch_geometric.utils.to_dense_adj(idx1)
        g1 = ((g1 + g1.transpose(2,1)) > 0).float()
        graph1.append(g1)
    graph1 = torch.cat(graph1,0).numpy()
    print(graph1.shape)
    '''
    for j in range(len(test)):
        print('test data %03d'%j)
        idx2 = torch_geometric.nn.knn_graph(torch.FloatTensor(test[j]) ,10,None,True,'target_to_source')
        #idx2 = torch_geometric.nn.radius_graph(torch.FloatTensor(test[j]),0.25,batch=None,loop=True,flow='target_to_source')
        g2 = torch_geometric.utils.to_dense_adj(idx2)
        g2 = ((g2 + g2.transpose(2,1)) > 0).float() 
        graph2.append(g2)
    graph2 = torch.cat(graph2,0).numpy()
    print(graph2.shape)
    #np.savez('./modelnet/modelnet40_graph.npz', graph1=graph1,graph2=graph2)
    np.savez('./modelnet/modelnet40_graph2.npz', graph2=graph2)