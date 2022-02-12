import time
import csv
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import statistics

import numpy as np
import sys
sys.path.insert(0, 'lib/')
import utilsdata


def calculation(pred_test, test_labels, method='GCN'):
    test_acc = metrics.accuracy_score(pred_test, test_labels)
    test_f1_macro = metrics.f1_score(pred_test, test_labels, average='macro')
    test_f1_micro = metrics.f1_score(pred_test, test_labels, average='micro')
    f1 = metrics.f1_score(pred_test, test_labels, average=None)
    precision = metrics.precision_score(test_labels, pred_test, average=None)
    recall = metrics.recall_score(test_labels, pred_test, average=None)
    print(f'method: {method}')
    print(f'test accuracy: {test_acc}')
    print(f'test f1 macro: {test_f1_macro}')
    print(f'test f1 micro: {test_f1_micro}')
    print(f'test precision all classes: {precision}')
    print(f'test recall all classes: {recall}')
    print(f'test f1 all classes: {f1}')
    print(f'test f1 median = {statistics.median(f1)}')
        

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            m.bias.data.fill_(0.0)

def test_model(net, loader, L, args):
    t_start_test = time.time()
    
    net.eval()
    test_acc = 0
    count = 0
    confusionGCN = np.zeros([args.nclass, args.nclass])
    predictions = pd.DataFrame()
    y_true = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    for batch_x, batch_y in loader:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        out_gae, out_hidden, pred, out_adj = net(batch_x, args.dropout, L)
        
        test_acc += utilsdata.accuracy(pred, batch_y).item() * len(batch_y)
        count += 1
        y_true = batch_y.detach().cpu().numpy()
        y_predProbs = pred.detach().cpu().numpy()
        
    predictions = pd.DataFrame(y_predProbs)            
    for i in range(len(y_true)):
        confusionGCN[y_true[i], np.argmax(y_predProbs[i,:])] += 1
    
    t_total_test = time.time() - t_start_test
    preds_labels = np.argmax(np.asarray(predictions), 1)
    test_acc = test_acc/len(loader.dataset)
    predictions.insert(0, 'trueLabels', y_true)
    
    return test_acc, confusionGCN, predictions, preds_labels, t_total_test


def train_model(useModel, train_loader, val_loader, L, args, seed):   

    torch.manual_seed(seed) 

    # network parameters
    D_g = args.num_gene
    CL1_F = 5
    CL1_K = 5
    FC1_F = 32
    FC2_F = 0
    NN_FC1 = 256
    NN_FC2 = 32
    out_dim = args.nclass  
    net_parameters = [D_g, CL1_F, CL1_K, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim]

    # learning parameters
    dropout_value = 0.2
    l2_regularization = 5e-4
    batch_size = args.batchsize
    num_epochs = args.epochs_1
    

    nb_iter = int(num_epochs * args.train_size) // batch_size
    print('num_epochs=',num_epochs,', train_size=',args.train_size,', nb_iter=',nb_iter)
    
    # Optimizer
    global_lr = args.lr
    global_step = 0
    decay = 0.95
    decay_steps = args.train_size
        
        
   # instantiate the object net of the class
    net = useModel(net_parameters)
    net.apply(weight_init)
    
    if torch.cuda.is_available():
        net.cuda()
        
    # print(net)
            
    # optimizer = optim.Adam(net.parameters(),lr= args.lr, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), momentum=0.9, lr= args.lr)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    ## Train   
    net.train()
    losses_train = []
    acc_train = []
    
    t_total_train = time.time()

    def adjust_learning_rate(optimizer, epoch, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #    lr = args.lr * (0.1 ** (epoch // 30))
        lr = lr * pow( decay , float(global_step// decay_steps) )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
    
        # update learning rate
        cur_lr = adjust_learning_rate(optimizer, epoch, args.lr)
        
        # reset time
        t_start = time.time()
    
        # extract batches
        epoch_loss = 0.0
        epoch_acc = 0.0
        count = 0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
            optimizer.zero_grad()   
            out_gae, out_hidden, output, out_adj = net(batch_x, dropout_value, L)
            # print('++++++++++++++')
            # print(output.shape)

            loss_batch = net.loss(out_gae, batch_x, output, batch_y, l2_regularization)
          
            acc_batch = utilsdata.accuracy(output, batch_y).item()
            
            loss_batch.backward()
            optimizer.step()
            
            count += 1
            epoch_loss += loss_batch.item()
            epoch_acc += acc_batch
            global_step += args.batchsize 
            
            # print
            if count % 1000 == 0: # print every x mini-batches
                print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (epoch + 1, count, loss_batch.item(), acc_batch))
    
    
        epoch_loss /= count
        epoch_acc /= count
        losses_train.append(epoch_loss) # Calculating the loss
        acc_train.append(epoch_acc) # Calculating the acc
        # print
        t_stop = time.time() - t_start


        # with open('train_baseline.csv', 'a') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow([epoch+1, epoch_loss, epoch_acc])
        
        if (epoch+1) % 10 == 0 and epoch != 0:
            with torch.no_grad():
                val_acc = 0  
                count = 0
                for b_x, b_y in val_loader:
                    b_x, b_y = b_x.to(device), b_y.to(device)          
                    _, _, val_pred, _ = net(b_x, args.dropout, L)                    
                    val_acc += utilsdata.accuracy(val_pred, b_y).item() * len(b_y)
                    count += 1
                    
                val_acc = val_acc/len(val_loader.dataset)
                
            print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
                  (epoch + 1, epoch_loss, epoch_acc, t_stop, cur_lr))
            print('----accuracy(val)= ', val_acc)
            # print('training_time:',t_stop)
        else:
            print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
                  (epoch + 1, epoch_loss, epoch_acc, t_stop, cur_lr))
            # print('training_time:',t_stop)
        
    
    t_total_train = time.time() - t_total_train  
    
    return net, t_total_train




def build_adj_matrix(cfm, normalized=True):
    for i in range(len(cfm)):
        cfm[i][i] = 0

    if normalized:
        for i in range(len(cfm)):
            sum_row = sum(cfm[i])
            # print(sum_row)
            try:
                cfm[i] = [i/sum_row for i in cfm[i]]
            except ZeroDivisionError:
                pass

    return cfm




# def train_model(useModel, train_loader, val_loader, L, args):    

#     # network parameters
#     D_g = args.num_gene
#     CL1_F = 5
#     CL1_K = 5
#     FC1_F = 32
#     FC2_F = 0
#     NN_FC1 = 256
#     NN_FC2 = 32
#     out_dim = args.nclass  
#     net_parameters = [D_g, CL1_F, CL1_K, FC1_F,FC2_F,NN_FC1, NN_FC2, out_dim]

#     # learning parameters
#     dropout_value = 0.2
#     l2_regularization = 5e-4
#     batch_size = args.batchsize
#     num_epochs = args.epochs
    

#     nb_iter = int(num_epochs * args.train_size) // batch_size
#     print('num_epochs=',num_epochs,', train_size=',args.train_size,', nb_iter=',nb_iter)
    
#     # Optimizer
#     global_lr = args.lr
#     global_step = 0
#     decay = 0.95
#     decay_steps = args.train_size
        
        
#    # instantiate the object net of the class
#     net = useModel(net_parameters)
#     net.apply(weight_init)
    
#     if torch.cuda.is_available():
#         net.cuda()
        
#     print(net)
            
#     #optimizer = optim.Adam(net.parameters(),lr= args.lr, weight_decay=5e-4)
#     optimizer = optim.SGD(net.parameters(), momentum=0.9, lr= args.lr)
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
#     ## Train   
#     net.train()
#     losses_train = []
#     acc_train = []
    
#     t_total_train = time.time()

#     def adjust_learning_rate(optimizer, epoch, lr):
#         """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     #    lr = args.lr * (0.1 ** (epoch // 30))
#         lr = lr * pow( decay , float(global_step// decay_steps) )
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         return lr
    
#     for epoch in range(num_epochs):  # loop over the dataset multiple times
    
#         # update learning rate
#         cur_lr = adjust_learning_rate(optimizer,epoch, args.lr)
        
#         # reset time
#         t_start = time.time()
    
#         # extract batches
#         epoch_loss = 0.0
#         epoch_acc = 0.0
#         count = 0
#         for i, (batch_x, batch_y) in enumerate(train_loader):
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
#             optimizer.zero_grad()   
#             out_gae, out_hidden, output, out_adj = net(batch_x, dropout_value, L)

#             loss_batch = net.loss(out_gae, batch_x, output, batch_y, l2_regularization)
          
#             acc_batch = utilsdata.accuracy(output, batch_y).item()
            
#             loss_batch.backward()
#             optimizer.step()
            
#             count += 1
#             epoch_loss += loss_batch.item()
#             epoch_acc += acc_batch
#             global_step += args.batchsize 
            
#             # print
#             if count % 1000 == 0: # print every x mini-batches
#                 print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (epoch + 1, count, loss_batch.item(), acc_batch))
    
    
#         epoch_loss /= count
#         epoch_acc /= count
#         losses_train.append(epoch_loss) # Calculating the loss
#         acc_train.append(epoch_acc) # Calculating the acc
#         # print
#         t_stop = time.time() - t_start
        
#         if epoch % 10 == 0 and epoch != 0:
#             with torch.no_grad():
#                 val_acc = 0  
#                 count = 0
#                 for b_x, b_y in val_loader:
#                     b_x, b_y = b_x.to(device), b_y.to(device)          
#                     _, _, val_pred, _ = net(b_x, args.dropout, L)                    
#                     val_acc += utilsdata.accuracy(val_pred, b_y).item() * len(b_y)
#                     count += 1
                    
#                 val_acc = val_acc/len(val_loader.dataset)
                
#             print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
#                   (epoch + 1, epoch_loss, epoch_acc, t_stop, cur_lr))
#             print('----accuracy(val)= ', val_acc)
#             print('training_time:',t_stop)
#         else:
#             print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
#                   (epoch + 1, epoch_loss, epoch_acc, t_stop, cur_lr))
#             print('training_time:',t_stop)
        
    
#     t_total_train = time.time() - t_total_train  
    
#     return net, t_total_train