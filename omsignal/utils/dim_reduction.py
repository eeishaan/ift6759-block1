#!/usr/bin/env python3
'''
Module containing dimension reduction techniques
'''

import torch

from sklearn.manifold import TSNE


class SVDTransform(object):
    '''
    SVD transformer
    '''

    def __init__(self, out_dim=2):
        self.out_dim = out_dim
        self.name = 'SVD Transformer'

    def __call__(self, X):
        U, _, _ = torch.svd(X.t())
        return U[:, :self.out_dim]
    

class TSNETransform(object):
    '''
    t-SNE transformer
    '''

    def __init__(self, out_dim=2):
        self.out_dim = out_dim
        self.name = 't-SNE Transformer'
    
    def __call__(self, X):
        return TSNE(n_components=self.out_dim).fit_transform(X)
