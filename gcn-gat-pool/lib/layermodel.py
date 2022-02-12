import torch
#from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
from copy import deepcopy
import random
sys.path.insert(0, 'lib/')


if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

from coarsening import lmax_L
from coarsening import rescale_L
from utilsdata import sparse_mx_to_torch_sparse_tensor

class my_sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """

    @staticmethod
    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)             # torch.mm is matrix mult
        return y

    @staticmethod
    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input )
        return grad_input_dL_dW, grad_input_dL_dx



#########################################################################################################
class Graph_GCN(nn.Module):

    def __init__(self, net_parameters):

        print('Graph ConvNet: GCN')

        super(Graph_GCN, self).__init__()
    
        # parameters
        D_g, CL1_F, CL1_K, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim = net_parameters
        CNN1_F, CNN1_K = 32, 5
        CL2_F, CL2_K = 10, 10
        D_nn = D_g
        self.in_dim = D_g
        self.out_dim = out_dim
        self.FC2_F = FC2_F
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gene = D_nn
        self.initScale = initScale = 6
        self.poolsize = 8
        FC1Fin = CL1_F*(D_g//self.poolsize)
        self.FC1Fin = FC1Fin
        self.CL1_K = CL1_K; self.CL1_F = CL1_F; 
        
        # Feature_H, Feature_W = (Input_Height - filter_H + 2P)/S + 1, (Input_Width - filter_W + 2P)/S + 1
        height = int(np.ceil(np.sqrt(int(D_nn))))
        FC2Fin = int(CNN1_F * (height//2) ** 2)
        self.FC2Fin = FC2Fin;
        
        # graph CL1
        self.cl1 = nn.Linear(CL1_K, CL1_F)
#        # graph CL2
#        self.cl2 = nn.Linear(CL2_K*CL1_F, CL2_F)
#        #FC gcnpure
#        self.fc_gcnpure = nn.Linear(FC1Fin, self.out_dim)
        # FC 1
        self.fc1 = nn.Linear(FC1Fin, FC1_F)  # from (7*125) to 
        # FC 2
        if self.FC2_F == 0:
            FC2_F = self.num_gene

        self.fc2 = nn.Linear(FC1_F, FC2_F)
        # FC 3
        self.fc3 = nn.Linear(FC2_F, D_g)
        # CNN_FC1
        self.cnn_fc1 = nn.Linear(FC2Fin, FC1_F)
        #FC_concat with CNN
        Fin = FC1Fin + FC2Fin; Fout = self.out_dim;
        self.FC_concat = nn.Linear(Fin, self.out_dim)             
        #FC_sum2 with NN
        Fin = FC1_F + NN_FC2; Fout = self.out_dim;
        self.FC_sum2 = nn.Linear(Fin, Fout)                  
        #FC_sum1 with CNN
        Fin = FC1_F + FC1_F; Fout = self.out_dim;
        self.FC_sum1 = nn.Linear(Fin, Fout)             
        # NN_FC1
        self.nn_fc1 = nn.Linear(self.in_dim, NN_FC1)
        # NN_FC2
        self.nn_fc2 = nn.Linear(NN_FC1, NN_FC2)
        # NN_FC3_decode
        self.nn_fc3 = nn.Linear(NN_FC2, NN_FC1)
        # NN_FC4_decode
        Fin = NN_FC2; Fout = self.in_dim;
        self.nn_fc4 = nn.Linear(Fin, Fout)        

        
        # nb of parameters
        nb_param = CL1_K* CL1_F + CL1_F          # CL1
#        nb_param += CL2_K* CL1_F* CL2_F + CL2_F  # CL2
        nb_param += FC1Fin* FC1_F + FC1_F        # FC1
#        nb_param += FC1_F* FC2_F + FC2_F         # FC2
        print('nb of parameters=',nb_param,'\n')


    def init_weights(self, W, Fin, Fout):

        scale = np.sqrt( self.initScale / (Fin+Fout) )
        W.uniform_(-scale, scale)
        
        return W


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
        x = x0.unsqueeze(0)                 # 1 x V x Fin*B   1 x 1000 x 64

        if K > 1:
            x1 = my_sparse_mm().apply(L, x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)),0)    # 2 x V x Fin*B
        for k in range(2, K):
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


    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x


    def forward(self, x_in, dropout, L):
        # x_in = gene expression values (64 x 1000 = B x num_genes)
        x = x_in        #[:,:self.num_gene]       # for GCN
        x_nn = x_in     #[:,self.num_gene:]    # for NN

        x = x.unsqueeze(2) # B x V x Fin=1
        x = self.graph_conv_cheby(x, self.cl1, L[0], self.CL1_F, self.CL1_K)   # x shape = [B x V x 7]

        x = F.relu(x)
        x = self.graph_max_pool(x, self.poolsize)   # 64 x 125 x 7

        # flatten()
        x = x.view(-1, self.FC1Fin)  # [64 , 7*125]

        
        ##############################################
        ##                  GAE_re                  ##
        ##############################################
        x_reAdj = 0 #torch.stack([F.sigmoid(torch.mm(z_i, z_i.t())) for z_i in x_reAdj])
        
        ##############################################
        ##                  GAE                     ##
        ##############################################
        x = self.fc1(x)
        x = F.relu(x)
        x_hidden_gae = x    # 64 x 32
        # print('hidden gae', x_hidden_gae.shape)
        # exit()

        x_decode_gae = self.fc2(x_hidden_gae)
        if self.FC2_F != 0:                
            x_decode_gae = F.relu(x_decode_gae)
            x_decode_gae  = nn.Dropout(dropout)(x_decode_gae)            
            x_decode_gae = self.fc3(x_decode_gae)


        ##############################################
        ##                  GCN//NN                 ##
        ##############################################
        # NN
        x_nn = self.nn_fc1(x_nn) # B x V
        x_nn = F.relu(x_nn)
        x_nn = self.nn_fc2(x_nn)
        x_nn = F.relu(x_nn)     

        # concatenate layer  
        x = torch.cat((x_hidden_gae, x_nn), 1)    # B x 64
        x = self.FC_sum2(x)                  # B x 10
        x = F.log_softmax(x)
        
        return x_decode_gae, x_hidden_gae, x, x_reAdj  # final X is the prediction probability vector


    def loss(self, y1, y_target1, y2, y_target2, l2_regularization):
        loss1 = nn.MSELoss()(y1, y_target1)
        loss2 = nn.NLLLoss()(y2, y_target2)            
        loss = 1 * loss1 + 1 * loss2 

        l2_loss = 0.0
        for param in self.parameters():
            data = param* param
            l2_loss += data.sum()

        loss += 0.2* l2_regularization* l2_loss
        return loss



#########################################################################################################
class Double_GCN(nn.Module):

    def __init__(self, net_parameters):
        super(Double_GCN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # parameters
        D_g, CL1_F, CL1_K, FC1_F, FC2_F, NN_FC1, NN_FC2, CL4_F, CL4_K, FC1_GCN4, FC2_GCN4, out_dim = net_parameters
        CNN1_F, CNN1_K = 32, 5
        CL2_F, CL2_K = 10, 10
        self.in_dim = D_g
        self.out_dim = out_dim
        self.FC2_F = FC2_F
        self.num_gene = D_g
        self.num_class = out_dim
        self.initScale = initScale = 6
        self.poolsize = 8
        self.pool_GCN4 = 3   # best 3
        FC1Fin = CL1_F*(D_g//self.poolsize)
        self.GCN4_outdim = CL4_F * (self.num_class // self.pool_GCN4)   # 15
        self.FC1Fin = FC1Fin
        self.CL1_K = CL1_K; self.CL1_F = CL1_F; 
        self.CL4_K = CL4_K; self.CL4_F = CL4_F; 
        self.FC1_GCN4 = FC1_GCN4; self.FC2_GCN4 = FC2_GCN4
        
        # graph CL1
        self.cl1 = nn.Linear(CL1_K, CL1_F).to(self.device)
        self.fc1 = nn.Linear(FC1Fin, FC1_F).to(self.device)  # from (5*125, after pooling) to 32

        # graph CL4
        self.cl4 = nn.Linear(CL4_K*64, CL4_F).to(self.device)
        self.fc1_GCN4 = nn.Linear(self.GCN4_outdim, FC1_GCN4).to(self.device)
        self.fc2_GCN4 = nn.Linear(FC1_GCN4, FC2_GCN4)

        # FC 2 of the GCN 1
        if self.FC2_F == 0:
            FC2_F = self.num_gene

        self.fc2 = nn.Linear(FC1_F, FC2_F).to(self.device)
        # FC 3
        self.fc3 = nn.Linear(FC2_F, D_g).to(self.device)

        # FC
        Fin = FC1Fin
        Fout = self.out_dim
        
        inter = 256   # best 256
        final = 64    # best 64 

        #FC_sum2 is the final layer before softmax
        # Fin = FC1_F + NN_FC2 + FC1_GCN4
        Fin = FC1_F + NN_FC2 + FC2_GCN4 + final
        # Fin = FC1_F + NN_FC2 + self.GCN4_outdim
        self.FC_sum2 = nn.Linear(Fin, Fout).to(self.device)
          
        # NN_FC1
        self.nn_fc1 = nn.Linear(self.in_dim, NN_FC1).to(self.device)
        # NN_FC2
        self.nn_fc2 = nn.Linear(NN_FC1, NN_FC2).to(self.device)
        # NN_FC3_decode
        # self.nn_fc3 = nn.Linear(NN_FC2, NN_FC1).to(self.device)
        # NN_FC4_decode
        # Fin = NN_FC2; Fout = self.in_dim;
        # self.nn_fc4 = nn.Linear(Fin, Fout).to(self.device)

        ## FCN of the second model
        self.NN_14 = nn.Linear(640, inter).to(self.device)    # best: 640, 32
        self.NN_24 = nn.Linear(inter, final).to(self.device)     # best 32 12


    def init_weights(self, W, Fin, Fout):

        scale = np.sqrt( self.initScale / (Fin+Fout) )
        W.uniform_(-scale, scale)
        
        return W


    def graph_conv_cheby(self, x, cl, L, Fout, K):
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size
        B, V, Fin = x.size()
        B, V, Fin = int(B), int(V), int(Fin)

        # rescale Laplacian
        lmax = lmax_L(L)
        L = rescale_L(L, lmax)

        # convert scipy sparse matric L to pytorch
        L = sparse_mx_to_torch_sparse_tensor(L)
        if torch.cuda.is_available():
            L = L.cuda()

        # transform to Chebyshev basis
        x0 = x.permute(1,2,0).contiguous()  # V x Fin x B   (10 x 64 x 64)
        x0 = x0.view([V, Fin*B])            # V x Fin*B  (10 x 64*64)
        x = x0.unsqueeze(0)                 # 1 x V x Fin*B   1 x 10 x 64*64

        if K > 1:
            x1 = my_sparse_mm().apply(L, x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)),0)    # 2 x V x Fin*B
        for k in range(2, K):
            x2 = 2 * my_sparse_mm().apply(L, x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)),0)  # M x Fin*B --> K x V x Fin*B
            x0, x1 = x1, x2
        # print('x', x.shape)                  # K x 10 x 64*64

        x = x.view([K, V, Fin, B])           # K x V x Fin x B = 4 x 10 x 64 x 64
        x = x.permute(3,1,2,0).contiguous()  # B x V x Fin x K = 64 x 10 x 64 x 4
        x = x.view([B*V, Fin*K])             # B*V x Fin*K = 640 x 256

        # Compose linearly Fin features to get Fout features
        x = cl(x)                            # B*V x Fout = 640 x 4
        x = x.view([B, V, Fout])             # B x V x Fout
        return x


    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x


    def forward(self, x_in, x_embed, dropout, L, L_mg):
        # x_in = gene expression values (64 x 1000 = B x num_genes)
        x = x_in     #[:,:self.num_gene]       # for GCN
        x_nn = x_in  #[:,self.num_gene:]       # for NN       64 x 1000          
        batch_size = x.shape[0]
        x_embed = x_embed.unsqueeze(0)        # class + data embedding for 10 vertices, shape = 10 x 64
        x_embed = torch.broadcast_to(x_embed, (batch_size, x_embed.shape[1], x_embed.shape[2]))   # 64 x 10 x Fin=64
        x_GCN4 = x_embed
        x_nn_GCN4 = x_embed


        '''
        forward GCN 1
        '''
        x = x.unsqueeze(2) # B x V x Fin=1
        x = self.graph_conv_cheby(x, self.cl1, L[0], self.CL1_F, self.CL1_K)   # x shape = [B x V x self.CL1_F]

        x = F.relu(x)
        x = self.graph_max_pool(x, self.poolsize)   # Batch x V/self.poolsize x self.CL1_F

        # flatten()
        x = x.view(-1, self.FC1Fin)  # [batch x V/8*self.CL1_F] = [64 x 125*5]
        
        
        ##############################################
        ##                  GAE                     ##
        ##############################################
        x = self.fc1(x)
        x = F.relu(x)
        x_hidden_gae = x    # 64 x 32

        x_decode_gae = self.fc2(x_hidden_gae)
        if self.FC2_F != 0:                
            x_decode_gae = F.relu(x_decode_gae)
            x_decode_gae  = nn.Dropout(dropout)(x_decode_gae)            
            x_decode_gae = self.fc3(x_decode_gae)


        '''
        forward NN 1
        '''
        x_nn = self.nn_fc1(x_nn) # B x V
        x_nn = F.relu(x_nn)     # 64 x 256
        x_nn = self.nn_fc2(x_nn)  
        x_nn = F.relu(x_nn)     # 64 x 32
        # print('after NN', x_nn.shape)

#        x_hidden_ae = x_nn
#        x_decode_ae = self.nn_fc3(x_hidden_ae)
#        x_decode_ae = F.relu(x_decode_ae)
#        x_decode_ae = self.nn_fc4(x_hidden_ae)
#        x_decode_ae = F.relu(x_decode_ae)       

        # concatenate layer between GCN1
        x = torch.cat((x_hidden_gae, x_nn), 1)    # B x 64
        # print('before concat with GCN4')
        # print(x.shape)
        # print(x[0])


        '''
        forward GCN 4
        '''
        x_GCN4 = self.graph_conv_cheby(x_GCN4, self.cl4, L_mg[0], self.CL4_F, self.CL4_K)   # 1 x 10 x CL4_F
        x_GCN4 = F.relu(x_GCN4)
        x_GCN4 = self.graph_max_pool(x_GCN4, self.pool_GCN4)  # 1 x 10/pool_GCN4 x CLF_4
        x_GCN4 = x_GCN4.view(-1, self.GCN4_outdim)       # 1 x 10/pool_GCN4*CLF_4
        x_GCN4 = F.relu(self.fc1_GCN4(x_GCN4))
        x_GCN4 = F.relu(self.fc2_GCN4(x_GCN4))
        # x_GCN4 = torch.broadcast_to(x_GCN4, (batch_size, x_GCN4.shape[1]))


        # connect GCN1 + NN + GCN4
        x = torch.cat((x, x_GCN4), 1)
        # print('x[0] of GCN4 = ', x_GCN4[0])
        # print('after concat with GCN4')
        # print(x.shape)
        # print(x[0])


        '''
        forward NN of GCN4
        '''
        x_nn_GCN4 = x_nn_GCN4.view(batch_size, -1)
        x_nn_GCN4 = F.relu(self.NN_14(x_nn_GCN4))
        x_nn_GCN4 = F.relu(self.NN_24(x_nn_GCN4))

        x = torch.cat((x, x_nn_GCN4), 1)


        # final hidden layer
        data_emb = x
        

        # final fully connected layer
        x = self.FC_sum2(x)
        x = F.log_softmax(x)

        x_reAdj = 0
        return x_decode_gae, x_hidden_gae, x, x_reAdj  # final X is the prediction probability vector


    def loss(self, y1, y_target1, y2, y_target2, l2_regularization):

        # print('----------------')
        # print(y1.shape, y_target1.shape, y2.shape, y_target2.shape)
    
        loss1 = nn.MSELoss()(y1, y_target1)
        # print('loss recon', loss1) 

        class_weights = torch.tensor([1., 1., 1., 5, 1.75, 1.5, 1., 1., 1., 1.]).to(self.device)   # best 5, 1.75, 1.5  with acc = 0.9275
        # class_weights = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]).to(self.device)
        loss2 = nn.NLLLoss(weight=class_weights)(y2, y_target2)
        # loss2 = nn.CrossEntropyLoss(weight=class_weights)(y2, y_target2)
        # loss2 = nn.NLLLoss()(y2, y_target2)
        loss = 1 * loss1 + 1 * loss2 

        l2_loss = 0.0
        for param in self.parameters():
            data = param * param
            l2_loss += data.sum()
        # print('loss reg', l2_loss) 

        loss += 0.2 * l2_regularization * l2_loss

        return loss