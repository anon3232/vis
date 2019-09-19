from __future__ import print_function

import argparse
import time

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import cVAE
from data_utils import get_dataset_loaders

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--nref-tr', type=int, default=5, metavar='N')
parser.add_argument('--nref-te', type=int, default=10, metavar='N')
parser.add_argument('--nref-cl', type=int, default=10, metavar='N')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N')
parser.add_argument('--lr-inn', type=float, default=5e-4, metavar='N')  # prev 0.0001
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#train_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=True, download=True,
#                   transform=transforms.ToTensor()),
#    batch_size=args.batch_size, shuffle=True, **kwargs)
#test_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#    batch_size=args.batch_size, shuffle=True, **kwargs)
#class_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#    batch_size=1, shuffle=True, **kwargs)


train_loader, test_loader,class_loader = get_dataset_loaders(
    'mnist', args.batch_size, **kwargs)

num_classes = 10


model = cVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
#optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(model.parameters())

# Reconstruction + KL divergence losses summed over all elements and batch


def loss_function(recon_x, z, x, y, mu, logvar, model, N_ref):

    BCE = -torch.sum(x.view(-1, 784)*torch.log(recon_x) +
                     (1-x.view(-1, 784))*torch.log(1-recon_x))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # z = mu + eps*sigma
    # eps = (z - mu)/sigma
    J = mu.shape[1]
    prior = -J/2*np.log(2*np.pi) - 0.5*torch.sum(mu.pow(2) + logvar.exp())
    ent = -J/2*np.log(2*np.pi) - 0.5*torch.sum(1 + logvar)

    lr_inn = args.lr_inn
    #print(lr_inn)

    # 0.001 for 482
    BCE_ref = BCE
    for i in range(N_ref):
        l_g = torch.autograd.grad(
            BCE_ref, z, create_graph=True)[0]
        # it worsens a little in all logliks than the version not regularized with p(z)
        z = z - lr_inn*(l_g + z)
        new_x = model.decode(z, y)
        BCE_ref = -torch.sum(x.view(-1, 784)*torch.log(new_x) +
                             (1-x.view(-1, 784))*torch.log(1-new_x))

    #print(model.lr_inn)
    return BCE_ref - prior + ent

def loss_function_approx(recon_x, x, y, model, z, N_ref=0):

    S = 5  # with 10, gpu suffers and and time goes from 10 to 15. Better use 5
    _, mu, logvar, _ = model(x, y)
    q = torch.distributions.Normal(mu, torch.exp(0.5*logvar))
    prior = torch.distributions.Normal(
        torch.zeros_like(mu), torch.ones_like(logvar))

    zs = q.rsample(torch.tensor([S]))

    #zs = z
    zs_orig = zs
    #print(y.repeat(5, 1, 1).shape)
    #print(zs.shape)
    new_x = model.decode(zs, y.repeat(S, 1, 1))
    #print(new_x.shape)
    BCE = -torch.sum(x.view(-1, 784)*torch.log(new_x + 1e-7) +
                     (1-x.view(-1, 784))*torch.log(1-new_x + 1e-7))

    #lr_inn = args.lr  # *0.1
    # print(lr_inn)

    BCE_ref = BCE
    # print(BCE_ref)
    for i in range(N_ref):
        l_g = torch.autograd.grad(
            BCE_ref, zs, create_graph=True)[0]
        # it worsens a little in all logliks than the version not regularized with p(z)
        zs = zs - model.lr_inn*(l_g + zs)

        # print(lr_inn*(l_g).sum())
        new_x = model.decode(zs, y.repeat(S, 1, 1))

        BCE_ref = -torch.sum(x.view(-1, 784)*torch.log(new_x + 1e-7) +
                             (1-x.view(-1, 784))*torch.log(1-new_x + 1e-7))

    new_x = model.decode(zs, y.repeat(S, 1, 1))
    BCE_ref = -torch.sum(x.view(-1, 784)*torch.log(new_x + 1e-7) +
                         (1-x.view(-1, 784))*torch.log(1-new_x + 1e-7))
    # tmp = -torch.sum(x.view(-1, 784)*torch.log(new_x  + 1e-7 ) + \
    #                         (1-x.view(-1, 784))*torch.log(1-new_x  + 1e-7))+ \
    #    prior.log_prob(zs).sum(-1) - q.log_prob(zs).sum(-1)
    #tmp = tmp.mean(0)

    # print( p.log_prob(x.view(-1, 784)).sum() )
    #J = mu.shape[1]
    #prior_ana = -J/2*np.log(2*np.pi) - 0.5*torch.sum(mu.pow(2) + logvar.exp())
    #ent_ana = -J/2*np.log(2*np.pi) - 0.5*torch.sum(1 + logvar)

    # print(prior.log_prob(zs).shape)
    # print(BCE_ref.shape)
    # return BCE_ref - prior_ana + ent_ana
    #print(q.log_prob(zs_orig).sum())

    # fix metrics for the non delta case. Sell HMM/DLM experiments as ablations
    qq = torch.distributions.Normal(zs, torch.exp(0.5*logvar))
    #return (BCE_ref - prior.log_prob(zs).sum() + qq.log_prob(zs).sum() )/S
    return (BCE_ref - prior.log_prob(zs).sum() + qq.log_prob(zs).sum() )/S

    #p = torch.distributions.Bernoulli(model.decode(zs))
    # tmp = p.log_prob(x.view(-1, 784)).sum(-1) + \
    #    prior.log_prob(zs).sum(-1) - q.log_prob(zs).sum(-1)
    #tmp = tmp.logsumexp(0) - torch.log(torch.tensor(float(S)))
    # return -tmp.sum()


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(
        y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def train(epoch):
    model.to('cuda')
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        y = to_one_hot(label, 10).to(device)
        recon_batch, z, mu, logvar = model(data, y)
        #loss = loss_function(recon_batch, z, data, y, mu, logvar, model, N_ref=args.nref_tr)
        loss = loss_function_approx(recon_batch, data, y, model, z, N_ref=args.nref_tr)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    # model.eval()
    if args.nref_te < 1000 or epoch > args.epochs - 1:
        test_loss = 0
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            y = to_one_hot(label, 10).to(device)
            recon_batch, z, mu, logvar = model(data, y)
            test_loss += loss_function_approx(recon_batch, data, y, model, z, N_ref=args.nref_te).item()
            #test_loss += loss_function(recon_batch, z,
            #                        data, y, mu, logvar, model, N_ref=args.nref_te).item()
            # if i == 0:
            #    n = min(data.size(0), 8)
            #    comparison = torch.cat([data[:n],
            #                            recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #    save_image(comparison.cpu(),
            #               '../results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


def classify(epoch):
    
    if epoch > args.epochs - 1:
        device = 'cpu'
        model.to(device)
        # model.eval()
        test_loss = 0.
        acc = 0.
        for i, (data, label) in enumerate(class_loader):
            if i < 1000:
                #print(i)
                test_loss = torch.zeros([data.shape[0], num_classes])
                data = data.to(device)
                #yy = torch.zeros()
                for c in range(num_classes):
                    #label = label.to(device)
                    y = Variable(c*torch.ones(data.shape[0]))
                    y = to_one_hot(y, 10).to(device)

                    #print(data.shape)
                    #print(y.shape)
                    recon_batch, z, mu, logvar = model(data, y)
                    #test_loss[c] = loss_function(recon_batch, z,
                    #                            data, y, mu, logvar, model, N_ref=args.nref_te).item()
                    
                    test_loss[:, c] = loss_function_approx(recon_batch, data, y, model, z, N_ref=args.nref_te).item()
                    # if i == 0:
                    #    n = min(data.size(0), 8)
                    #    comparison = torch.cat([data[:n],
                    #                            recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                    #    save_image(comparison.cpu(),
                    #
                    #             '../results/reconstruction_' + str(epoch) + '.png', nrow=n)
                #print(test_loss.shape)
                #print('arg', np.argmin(test_loss, axis=1).shape)
                #print(label.shape)
                #print(np.argmin(test_loss.detach().numpy(), axis=1).astype(float))
                #print(label.detach().numpy().astype(float))
                #print(np.argmin(test_loss.detach().numpy(), axis=1).astype(float) == label.detach().numpy().astype(float))
                acc += np.sum(np.argmin(test_loss.detach().numpy(), axis=1).astype(float) == label.detach().numpy().astype(float))
                #print(acc)
        #print(acc/len(class_loader.dataset))
        print(acc/1000.)

        print(epoch)
        #test_loss /= len(class_loader.dataset)
        #print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):

        start = time.time()
        train(epoch)
        end = time. time()
        print('Time: ', end-start)
        test(epoch)
        classify(epoch)

        # classify(epoch)
        # with torch.no_grad():
        #    sample = torch.randn(64, 20).to(device)
        #    sample = model.decode(sample).cpu()
        #    save_image(sample.view(64, 1, 28, 28),
        #               '../results/sample_' + str(epoch) + '.png')

# 0.9197, 5 for train and 10 for test, 0.9165 with no refs
# 95.6, lr=0.01, bs=1024, N_ref=0 both.
