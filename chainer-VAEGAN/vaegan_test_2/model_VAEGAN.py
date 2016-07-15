# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import pandas as pd
import math

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.vae import gaussian_kl_divergence

import six


# Encoder
class Encoder(chainer.Chain):
    def __init__(self, n_latent=1000, input_size=96, input_ch=3, output_ch=256):
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.input_size = input_size
        self.out_size = input_size/(2**4)
        super(Encoder, self).__init__(
            ec0 = L.Convolution2D(self.input_ch, self.output_ch/8, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.input_ch)),
            ec1 = L.Convolution2D(self.output_ch/8, self.output_ch/4, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch/8)),
            ec2 = L.Convolution2D(self.output_ch/4, self.output_ch/2, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch/4)),
            ec3 = L.Convolution2D(self.output_ch/2, self.output_ch, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch/2)),
            l4_mu = L.Linear(self.out_size*self.out_size*self.output_ch, n_latent, wscale=0.02*math.sqrt(self.out_size*self.out_size*self.output_ch)),
            l4_var = L.Linear(self.out_size*self.out_size*self.output_ch, n_latent, wscale=0.02*math.sqrt(self.out_size*self.out_size*self.output_ch)),
            bne0 = L.BatchNormalization(self.output_ch/8),
            bne1 = L.BatchNormalization(self.output_ch/4),
            bne2 = L.BatchNormalization(self.output_ch/2),
            bne3 = L.BatchNormalization(self.output_ch),
            )
        
    def __call__(self, x, test=False):
        h = F.relu(self.bne0(self.ec0(x), test=test))
        h = F.relu(self.bne1(self.ec1(h), test=test))
        h = F.relu(self.bne2(self.ec2(h), test=test))
        h = F.relu(self.bne3(self.ec3(h), test=test))
        mu = F.relu(self.l4_mu(h))
        var = F.relu(self.l4_var(h))
        return mu, var

# Decoder
class Decoder(chainer.Chain):
    def __init__(self, n_latent=1000, input_size=96, input_ch=3, output_ch=256):
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.input_size = input_size
        self.out_size = input_size/(2**4)
        super(Decoder, self).__init__(
            # decoder
            l0z = L.Linear(n_latent, self.out_size*self.out_size*self.output_ch, wscale=0.02*math.sqrt(n_latent)),
            dc1 = L.Deconvolution2D(self.output_ch, self.output_ch/2, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch)),
            dc2 = L.Deconvolution2D(self.output_ch/2, self.output_ch/4, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch/2)),
            dc3 = L.Deconvolution2D(self.output_ch/4, self.output_ch/8, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch/4)),
            dc4 = L.Deconvolution2D(self.output_ch/8, self.input_ch, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch/8)),
            bnd0l = L.BatchNormalization(self.out_size*self.out_size*self.output_ch),
            bnd0 = L.BatchNormalization(self.output_ch),
            bnd1 = L.BatchNormalization(self.output_ch/2),
            bnd2 = L.BatchNormalization(self.output_ch/4),
            bnd3 = L.BatchNormalization(self.output_ch/8),
            )

    def __call__(self, z, sigmoid=True, test=False):
        h = F.reshape(F.relu(self.bnd0l(self.l0z(z), test=test)), (z.data.shape[0], self.output_ch, self.out_size, self.out_size))
        h = F.relu(self.bnd1(self.dc1(h), test=test))
        h = F.relu(self.bnd2(self.dc2(h), test=test))
        h = F.relu(self.bnd3(self.dc3(h), test=test))
        x = (self.dc4(h))
        if sigmoid:
            return F.sigmoid(x)
        else:
            return x

# Discriminator
class Discriminator(chainer.Chain):
    def __init__(self, n_latent=1000, input_size=96, input_ch=3, output_ch=256):
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.input_size = input_size
        self.out_size = input_size/(2**4)
        super(Discriminator, self).__init__(
            # discriminator
            gc0 = L.Convolution2D(self.input_ch, self.output_ch/8, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.input_ch)),
            gc1 = L.Convolution2D(self.output_ch/8, self.output_ch/4, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch/8)),
            gc2 = L.Convolution2D(self.output_ch/4, self.output_ch/2, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch/4)),
            gc3 = L.Convolution2D(self.output_ch/2, self.output_ch, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch/2)),
            gl = L.Linear(self.out_size*self.out_size*self.output_ch, 2, wscale=0.02*math.sqrt(self.out_size*self.out_size*self.output_ch)),
            gn0 = L.BatchNormalization(self.output_ch/8),
            gn1 = L.BatchNormalization(self.output_ch/4),
            gn2 = L.BatchNormalization(self.output_ch/2),
            gn3 = L.BatchNormalization(self.output_ch),
            )

    def __call__(self, rec, test=False):
        h0 = F.relu(self.gn0(self.gc0(rec), test=test))
        h1 = F.relu(self.gn1(self.gc1(h0), test=test))
        h2 = F.relu(self.gn2(self.gc2(h1), test=test))
        h3 = F.relu(self.gn3(self.gc3(h2), test=test))
        d = self.gl(h3)
        hidden_out = [h0, h1, h2, h3]
        return d, hidden_out

