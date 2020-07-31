import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import Iterable
from sklearn.preprocessing import LabelEncoder

def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization."
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


def embedding(ni,nf):
    "Create an embedding layer."
    emb = nn.Embedding(ni, nf)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad(): trunc_normal_(emb.weight, std=0.01)
    return emb


def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None: p=[]
    elif isinstance(p, str):          p = [p]
    elif not isinstance(p, Iterable): p = [p]
    #Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try: a = len(p)
        except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


def ifnone(a, b):
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def label_encoder(X, cols): 
    X_encoded = X.copy(deep=True)

    for col in cols:
        X_encoded.loc[:, col] = LabelEncoder().fit_transform(X_encoded.loc[:, col])
    return X_encoded

class FeedFowardNNet(nn.Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs, cont, categ, out_sz, layers, ps=None,
                 emb_drop=0.1, y_range=None, use_bn=True, bn_final=False, nonlin=nn.ReLU):
        super(FeedFowardNNet, self).__init__()
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        n_cont = len(cont)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        self.categ, self.cont= categ, cont
        sizes = self.get_sizes(layers, out_sz)
        actns = [nonlin(inplace=True) for _ in range(len(sizes)-2)] + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, X, **kwargs): # X, **kwargs Add scikit learn compat
        if self.n_emb != 0:
            x_cat = X[:, self.categ].to(torch.int64)
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = X[:, self.cont].to(torch.float32)
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        return x