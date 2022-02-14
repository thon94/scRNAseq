import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from coarsening import lmax_L, rescale_L, laplacian
from utils import sparse_mx_to_torch_sparse_tensor


class my_sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """

    @staticmethod
    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x.cuda())             # torch.mm is matrix mult
        return y

    @staticmethod
    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input )
        return grad_input_dL_dW, grad_input_dL_dx



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, n_gene=1000, poolsize=8, nn_embed=64):
        """Dense version of GAT."""
        """
            nfeat: in_features (F)
            nhid: out_features (F')
        """
        super(GAT, self).__init__()
        self.GCNembed = nfeat

        self.cl = nn.Linear(1, nfeat)
        self.poolsize = poolsize

        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        embed_size = 256   #
        self.fc1 = nn.Linear(nhid * nheads * n_gene // poolsize, embed_size)
        self.fc2 = nn.Linear(embed_size + nn_embed, nclass)

        self.decoder = nn.Linear(embed_size, n_gene)

        self.nnfc1 = nn.Linear(n_gene, 512)
        self.nnfc2 = nn.Linear(512, nn_embed)


    def forward(self, x, adj, conv_degree=3):
        L = [laplacian(adj, normalized=True)]
        x_nn = deepcopy(x)
        x = x.unsqueeze(2)
        x = self.graph_conv_cheby(x, self.cl, L[0], self.GCNembed, conv_degree)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(x)
        x = self.graph_max_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        # extract embedded features for decoding --> reconstruction
        x_enc = x
        x_rec = self.decoder(x_enc)

        # extract embedding features using nnfc layers
        x_nn = F.relu(self.nnfc1(x_nn))
        x_nn = F.relu(self.nnfc2(x_nn))

        # concat nnfc and gcn-gat-pooling embeddings
        x = torch.cat((x, x_nn), 1)

        # class prediction
        x = F.relu(self.fc2(x))
        y_preds = F.log_softmax(x, dim=1)

        return x_rec, y_preds


    def loss_func(self, x_rec, x_in, y_preds, y_labels):
        rec_loss = nn.MSELoss()(x_rec, x_in)
        class_loss = nn.NLLLoss()(y_preds, y_labels)
        return 1 * rec_loss + 1 * class_loss


    def graph_conv_cheby(self, x, cl, L, Fout, K):
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size
        B, V, Fin = x.size(); B, V, Fin = int(B), int(V), int(Fin)

        # rescale Laplacian
        lmax = lmax_L(L)
        L = rescale_L(L, lmax)

        # convert scipy sparse matric L to pytorch
        L = sparse_mx_to_torch_sparse_tensor(L)
        if torch.cuda.is_available():
            L = L.cuda()

        # transform to Chebyshev basis
        x0 = x.permute(1,2,0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin*B])            # V x Fin*B
        x0 = x0.cuda()
        x = x0.unsqueeze(0)                 # 1 x V x Fin*B   1 x 1000 x 64

        if K > 1:
            x1 = my_sparse_mm().apply(L, x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)),0)    # 2 x V x Fin*B
        for _ in range(2, K):
            x2 = 2 * my_sparse_mm().apply(L, x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)),0)  # M x Fin*B --> K x V x Fin*B
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])           # K x V x Fin x B
        x = x.permute(3,1,2,0).contiguous()  # B x V x Fin x K
        x = x.view([B*V, Fin*K])             # B*V x Fin*K

        # Compose linearly Fin features to get Fout features
        x = cl(x)                            # B*V x Fout
        x = x.view([B, V, Fout])             # B x V x Fout
        return x

    
    def graph_max_pool(self, x):
        if self.poolsize > 1:
            x = x.permute(0, 2, 1).contiguous()
            x = nn.MaxPool1d(self.poolsize)(x)
            x = x.permute(0, 2, 1).contiguous()
            return x
        else:
            return x


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

