import os
import glob
import logging
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import lib.utilsdata as utilsdata
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Zhengsorted', help='Name of dataset')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.') 
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--embed', type=int, default=10, help='Number of hidden units.')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

parser.add_argument('--num_gene', type=int, default=1000, help='Number of filtered genes')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
print('---loading dataset')
adjall, alldata, labels, shuffle_index = utilsdata.load_largesc(path='../', dirAdj='../'+args.dataset+'/', dataset=args.dataset, net='String')

print('---shuffling samples')
if shuffle_index.all():
    shuffle_index = shuffle_index.astype(np.int32)
else:
    shuffle_index = np.random.permutation(alldata.shape[0])
    np.savetxt('../' + args.dataset +'/shuffle_index_'+ args.dataset + '.txt', X=shuffle_index)
    
print('---filtering genes')
train_all_data, adj = utilsdata.down_genes(alldata, adjall, args.num_gene)


# Split the dataset into train, val, test dataset. Use a fixed shuffle index to fix the sample order for comparison.
print('---split dataset')
train_data, val_data, test_data, train_labels, val_labels, test_labels = utilsdata.spilt_dataset(train_all_data, labels, shuffle_index)
nclass = len(np.unique(labels))

train_loader, val_loader, test_loader, _ = utilsdata.generate_loader(train_data, val_data, test_data, train_labels, val_labels, test_labels, args.batch_size)


# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=args.embed, 
                nhid=args.hidden,
                nclass=nclass,
                dropout=args.dropout,
                nheads=args.nb_heads,
                n_gene=args.num_gene,
                poolsize=8,
                alpha=args.alpha)
else:
    model = GAT(nfeat=args.embed,
                nhid=args.hidden,
                nclass=nclass,
                dropout=args.dropout,
                nheads=args.nb_heads,
                n_gene=args.num_gene,
                poolsize=8,
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

# trainable params
print('--> trainable parameters')
for n, p in model.named_parameters():
    print('\t',n)

if args.cuda:
    model.cuda()


def train(train_loader, val_loader, adj):
    t = time.time()
    model.train()

    loss_values = []
    for epoch in range(args.epochs):
        epoch_loss = 0.
        epoch_acc = 0.
        count = 0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            if args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            optimizer.zero_grad()
            output = model.forward(batch_x, adj)
            loss_batch = F.nll_loss(output, batch_y)
            acc_batch = utilsdata.accuracy(output, batch_y).item()
            loss_batch.backward()

            # check back propagation gradients
            # for n, p in model.named_parameters():
            #     print(n)
            #     print(p)
            #     print(p.grad)
            # exit()

            optimizer.step()

            count += 1
            epoch_loss += loss_batch.item()
            epoch_acc += acc_batch
            
            # if count % 1000 == 0:
            #     print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (epoch + 1, count, loss_batch.item(), acc_batch))

        epoch_loss /= count
        epoch_acc /= count

        # if (epoch+1) % 10 ==0 and epoch != 0:
        with torch.no_grad():
            val_acc = 0.
            val_loss = 0.
            count = 0
            for b_x, b_y in val_loader:
                if args.cuda: b_x, b_y = b_x.cuda(), b_y.cuda()
                val_pred = model(b_x, adj)
                val_loss += F.nll_loss(val_pred, b_y)
                val_acc += utilsdata.accuracy(val_pred, b_y).item()
                count += 1
            
            val_acc /= count
            val_loss /= count

        loss_values.append(val_loss)

        print('epoch= {}, loss(train)= {:.3f}, acc(train)= {:.3f}'.format(epoch+1, epoch_loss, epoch_acc))
        print('--> loss(val)= {:.3f}, acc(val)= {:.3f}'.format(val_loss, val_acc))
        # else:
        #     print(f'epoch= {epoch+1}, loss(train)= {epoch_loss:%.3f}, acc(train)= {epoch_acc:%.3f}')

    return loss_values



def compute_test():
    model.eval()
    test_acc = 0.
    test_loss = 0.
    count = 0
    for b_x, b_y in test_loader:
        if args.cuda: b_x, b_y = b_x.cuda(), b_y.cuda()
        test_pred = model(b_x, adj)
        test_loss += F.nll_loss(test_pred, b_y)
        test_acc += utilsdata.accuracy(test_pred, b_y)
        count += 1
    print("Test set results:",
          "loss= {:.4f}".format(test_loss/count),
          "accuracy= {:.4f}".format(test_acc/count))

# Train model
t_total = time.time()
val_loss = train(train_loader, val_loader, adj)
# loss_values = []
# bad_counter = 0
# best = args.epochs + 1
# best_epoch = 0
# for epoch in range(args.epochs):
#     loss_values.append(train(epoch, train_loader, val_loader, adj))

#     torch.save(model.state_dict(), '{}.pkl'.format(epoch))
#     if loss_values[-1] < best:
#         best = loss_values[-1]
#         best_epoch = epoch
#         bad_counter = 0
#     else:
#         bad_counter += 1

#     if bad_counter == args.patience:
#         break

#     files = glob.glob('*.pkl')
#     for file in files:
#         epoch_nb = int(file.split('.')[0])
#         if epoch_nb < best_epoch:
#             os.remove(file)

# files = glob.glob('*.pkl')
# for file in files:
#     epoch_nb = int(file.split('.')[0])
#     if epoch_nb > best_epoch:
#         os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
# print('Loading {}th epoch'.format(best_epoch))
# model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()
