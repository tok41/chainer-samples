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


class EncodeDecode():
    def __init__(self, model_file, n_latent=1000, input_size=96, input_ch=3, output_ch=512):
        '''
        Arguments:
          model_file : 学習ずみモデルファイルのパス付きファイル名 [String]
          n_latent : 中間表現ベクトルzの次元数 [Integer]
          input_size : 入力画像のサイズ(正方形画像を想定) [Integer]
          input_ch : 入力画像のカラーチャネル [Integer]
          output_ch : Convolutionネットワークの最大チャネルサイズ [Integer]
        '''
        self.n_latent = n_latent
        self.input_size = input_size
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.model_file = model_file
        self.model = VAE(n_latent=n_latent, input_size=input_size, input_ch=input_ch, output_ch=output_ch)
        serializers.load_hdf5(self.model_file, self.model)
        
    def getLatentVector(self, images):
        '''
        中間表現ベクトルを取得する
        arguments:
            images : 画像データ.
                サイズ : [N, ch, width, height]
                N=画像枚数, ch=チャネル(カラーの場合3)
                0~1に正規化されたデータ
            型 : chainer.Variable()
        return:
            latent_vector : n_latent次元のベクトル
        '''
        mu, sig = self.model.encode(images)
        self.mu = mu
        self.sig = sig
        return mu
    
    def getReconstructImage(self, z):
        y = self.model.decode(z)
        return y
    

class VAE(chainer.Chain):
    """AutoEncoder"""
    def __init__(self, n_latent=1000, input_size=128, input_ch=3, output_ch=512):
        '''
        Arguments:
          n_latent : 中間表現ベクトルzの次元数 [Integer]
          input_size : 入力画像のサイズ(正方形画像を想定) [Integer]
          input_ch : 入力画像のカラーチャネル [Integer]
          output_ch : Convolutionネットワークの最大チャネルサイズ [Integer]
        '''
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.input_size = input_size
        self.out_size = input_size/(2**4)
        super(VAE, self).__init__(
            ## ネットワーク構造の定義
            # encoder
            c0 = L.Convolution2D(self.input_ch, self.output_ch/8, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.input_ch)),
            c1 = L.Convolution2D(self.output_ch/8, self.output_ch/4, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch/8)),
            c2 = L.Convolution2D(self.output_ch/4, self.output_ch/2, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch/4)),
            c3 = L.Convolution2D(self.output_ch/2, self.output_ch, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*self.output_ch/2)),
            l4_mu = L.Linear(self.out_size*self.out_size*self.output_ch, n_latent, wscale=0.02*math.sqrt(self.out_size*self.out_size*self.output_ch)),
            l4_var = L.Linear(self.out_size*self.out_size*self.output_ch, n_latent, wscale=0.02*math.sqrt(self.out_size*self.out_size*self.output_ch)),
            bne0 = L.BatchNormalization(self.output_ch/8),
            bne1 = L.BatchNormalization(self.output_ch/4),
            bne2 = L.BatchNormalization(self.output_ch/2),
            bne3 = L.BatchNormalization(self.output_ch),
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

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        # 下記、encodeとdecodeの中身をこの中に書いても良いがencodeとｄｅｃｏｄｅは他でも使うので再利用性を高めるために
        return self.decode(self.encode(x)[0], sigmoid)
    
    def encode(self, x, test=False):
        # 推論モデル, 中間表現のベクトルqを学習
        h = F.relu(self.bne0(self.c0(x)))
        h = F.relu(self.bne1(self.c1(h), test=test))
        h = F.relu(self.bne2(self.c2(h), test=test))
        h = F.relu(self.bne3(self.c3(h), test=test))
        mu = (self.l4_mu(h))
        var = (self.l4_var(h))
        return mu, var

    def decode(self, z, sigmoid=True, test=False):
        # 中間表現ベクトルqを入力として(z), 画像を生成
        h = F.reshape(F.relu(self.bnd0l(self.l0z(z), test=test)), (z.data.shape[0], self.output_ch, self.out_size, self.out_size))
        h = F.relu(self.bnd1(self.dc1(h), test=test))
        h = F.relu(self.bnd2(self.dc2(h), test=test))
        h = F.relu(self.bnd3(self.dc3(h), test=test))
        x = (self.dc4(h))
        if sigmoid:
            return F.sigmoid(x)
        else:
            return x

    def get_loss_func(self, C=1.0, k=1, train=True):
        """Get loss function of VAE.
        Args:
        C (int): Usually this is 1.0. Can be changed to control the
        second term of ELBO bound, which works as regularization.
        k (int): Number of Monte Carlo samples used in encoded vector.
        train (bool): If true loss_function is used for training.
        """
        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)
                #rec_loss += F.mean_squared_error(x, self.decode(z)) / (k)
            self.rec_loss = rec_loss
            # reguralization
            self.loss = self.rec_loss + C * gaussian_kl_divergence(mu, ln_var) / batchsize
            return self.loss
        return lf
