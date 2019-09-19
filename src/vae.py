import argparse
import torch
import torch.utils.data
import time
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np

from models import VAE
from data_utils import get_dataset_loaders

# fMNIST: python vae.py --epochs 15 --nref-tr 0 --nref-te 0 --log 1000 --batch-size 128


parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--S', type=int, default=10)
parser.add_argument('--nref-tr', type=int, default=0)
parser.add_argument('--nref-te', type=int, default=0)

parser.add_argument('--binarize-stat', action='store_true', default=True,
                    help='statically binarize dataset')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader, test_loader,_ = get_dataset_loaders(
    'mnist', args.batch_size, **kwargs)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
sched = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, z, x, mu, logvar, model, N_ref):

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    BCE = -torch.sum(x.view(-1, 784)*torch.log(recon_x + 1e-7) +
                     (1-x.view(-1, 784))*torch.log(1-recon_x + 1e-7))
    
    J = mu.shape[1]
    prior = -J/2*np.log(2*np.pi) - 0.5*torch.sum(mu.pow(2) + logvar.exp())
    ent = -J/2*np.log(2*np.pi) - 0.5*torch.sum(1 + logvar)

    lr_inn = args.lr/1.

    BCE_ref = BCE
    for i in range(N_ref):
        l_g = torch.autograd.grad(
            BCE_ref, z, create_graph=True)[0]
        z = z - lr_inn*(l_g + z)
        new_x = model.decode(z)
        BCE_ref = -torch.sum(x.view(-1, 784)*torch.log(new_x + 1e-7) +
                             (1-x.view(-1, 784))*torch.log(1-new_x + 1e-7))
    return BCE_ref - prior + ent


def loss_function_approx(recon_x, x, model, z, N_ref=0):

    S = 5  # with 10, gpu suffers and and time goes from 10 to 15. Better use 5
    _, mu, logvar, _ = model(x)
    q = torch.distributions.Normal(mu, torch.exp(0.5*logvar))
    prior = torch.distributions.Normal(
        torch.zeros_like(mu), torch.ones_like(logvar))

    zs = q.rsample(torch.tensor([S]))

    #zs = z
    zs_orig = zs
    new_x = model.decode(zs)
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
        new_x = model.decode(zs)

        BCE_ref = -torch.sum(x.view(-1, 784)*torch.log(new_x + 1e-7) +
                             (1-x.view(-1, 784))*torch.log(1-new_x + 1e-7))

    new_x = model.decode(zs)
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
    return (BCE_ref - prior.log_prob(zs).sum() + q.log_prob(zs_orig).sum() )/S

    #p = torch.distributions.Bernoulli(model.decode(zs))
    # tmp = p.log_prob(x.view(-1, 784)).sum(-1) + \
    #    prior.log_prob(zs).sum(-1) - q.log_prob(zs).sum(-1)
    #tmp = tmp.logsumexp(0) - torch.log(torch.tensor(float(S)))
    # return -tmp.sum()


def compute_ll(model, x, z, S=args.S, N_ref=args.nref_te):

    recon_x, mu, logvar, _ = model(x)
    q = torch.distributions.Normal(mu, torch.exp(0.5*logvar))
    prior = torch.distributions.Normal(
        torch.zeros_like(mu), torch.ones_like(logvar))

    S = 10
    zs = q.rsample(torch.tensor([S]))
    zs.requires_grad_(True)
    zs_orig = zs
    #zs = zs.clone().detach()
    # Compute langevin here...

    new_x = model.decode(zs)
    BCE = -torch.sum(x.view(-1, 784)*torch.log(new_x + 1e-7) +
                     (1-x.view(-1, 784))*torch.log(1-new_x + 1e-7))

    #lr_inn = args.lr/1.
    # print(lr_inn)

    BCE_ref = BCE
    # print(BCE_ref)
    for i in range(N_ref):
        l_g = torch.autograd.grad(
            BCE_ref, zs, create_graph=False)[0]
        # it worsens a little in all logliks than the version not regularized with p(z)
        zs = zs - model.lr_inn*(l_g + zs)
        # print(lr_inn*(l_g).sum())
        new_x = model.decode(zs)

        BCE_ref = -torch.sum(x.view(-1, 784)*torch.log(new_x + 1e-7) +
                             (1-x.view(-1, 784))*torch.log(1-new_x + 1e-7))

    p = torch.distributions.Bernoulli(model.decode(zs))

    qq = torch.distributions.Normal(zs, torch.exp(0.5*logvar))

    #tmp = p.log_prob(x.view(-1, 784)).sum(-1) + \
    #    prior.log_prob(zs).sum(-1) - q.log_prob(zs_orig).sum(-1)
    tmp = p.log_prob(x.view(-1, 784)).sum(-1) + \
        prior.log_prob(zs).sum(-1) - qq.log_prob(zs).sum(-1)
    tmp = tmp.logsumexp(0) - torch.log(torch.tensor(float(S)))
    return tmp.sum(), model.decode(zs[0])


def train(epoch):
    model.train()
    train_loss = 0
    sched.step()

    for batch_idx, (data, _) in enumerate(train_loader):

        data = data.to(device)
        if args.binarize_stat:
            data[data > 0.5] = 1.
            data[data <= 0.5] = 0.

        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        loss = loss_function_approx(recon_batch, data, model, z, N_ref=args.nref_tr)

        loss.backward()

        #print(model.lr_inn.grad)
        train_loss += loss.item()

        optimizer.step()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    # model.eval()
    test_loss = 0
    ll_approx = 0
    ll_approx_more = 0
    # with torch.no_grad():
    for i, (data, _) in enumerate(test_loader):
        data = data.to(device)
        if args.binarize_stat:
            data[data > 0.5] = 1.
            data[data <= 0.5] = 0.
        recon_batch, mu, logvar, z = model(data)
        # test_loss += loss_function(recon_batch, z,
        #                           data, mu, logvar, model, N_ref=args.nref_te).item()
        #test_loss += loss_function_approx(recon_batch, data, model, z, N_ref=args.nref_te).item()
        ll, refined = compute_ll(model, data, z, S=args.S, N_ref=args.nref_te)
        ll_approx += ll
        #ll_approx_more += compute_ll(model, data, z, S=2, N_ref=args.nref_te)
        if i == 0:
            n = min(data.size(0), 10)
            comparison = torch.cat([data[:n],
                                    refined.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.cpu(),
                    'results/reconstruction_mnist_gauss' + str(epoch) + '_' +str(args.nref_tr) + '_' + str(args.nref_te) + '_' + str(args.seed) +'_.png', nrow=n)

        
    test_loss /= len(test_loader.dataset)
    ll_approx /= len(test_loader.dataset)
    ll_approx_more /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Test set -LL 10: {:.4f}'.format(-ll_approx))
    print('====> Test set -LL 100: {:.4f}'.format(-ll_approx_more))


if __name__ == "__main__":
    orig_sample = torch.randn(args.batch_size, 10).to(device)
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        train(epoch)
        print('dt_train ', time.time() - t)
        t = time.time()
        test(epoch)
        print('dt_test ', time.time() - t)
        # with torch.no_grad():
        print('Sampling...')
        
        sample = model.decode(orig_sample).cpu()

        #recon_batch, mu, logvar, z = model(data)
        #ll, recon = compute_ll(model, data, z, S=args.S, N_ref=args.nref_te)
        save_image(sample.view(args.batch_size, 1, 28, 28),
                  'results/sample_mnist_gauss' + str(epoch) +  '_' +str(args.nref_tr) + '_' + str(args.nref_te) + '_' + str(args.seed) + '.png')
