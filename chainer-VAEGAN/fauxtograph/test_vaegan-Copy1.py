
# coding: utf-8

# # VAE+GAN
# githubにあがってたものを利用する

import sys, os
import numpy as np
import pandas as pd
import six
import math

from PIL import Image
from StringIO import StringIO
import matplotlib.pyplot as plt

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.vae import gaussian_kl_divergence

import chainer.optimizers as O
import tqdm
import time
from IPython.display import display
import json


# In[2]:

from vaegan import Encoder, Decoder, Discriminator, EncDec
from fauxtograph import get_paths, image_resize


# In[3]:

paths = get_paths('/home/tokita/projects/cinet/YouTubePriors_flv4/DividedImages/images_resize/sample_train_s/')
print len(paths)


class VAEGAN(object):
    def __init__(self, img_width=64, img_height=64, color_channels=3, encode_layers=[1000, 600, 300],
                 decode_layers=[300, 800, 1000], disc_layers=[1000, 600, 300],
                 kl_ratio=1.0, latent_width=500, flag_gpu=True, mode='convolution',
                 enc_adam_alpha=0.0002, enc_adam_beta1=0.5, 
                 dec_adam_alpha=0.0002, dec_adam_beta1=0.5,
                 disc_adam_alpha=0.0001, disc_adam_beta1=0.5,
                 rectifier='clipped_relu', dropout_ratio=0.5):
        self.img_width = img_width
        self.img_height = img_height
        self.color_channels = color_channels
        self.encode_layers = encode_layers
        self.decode_layers = decode_layers
        self.disc_layers = disc_layers
        self.kl_ratio = kl_ratio
        self.latent_width = latent_width
        self.flag_gpu = flag_gpu
        self.mode = mode
        self.enc_adam_alpha = enc_adam_alpha
        self.enc_adam_beta1 = enc_adam_beta1
        self.dec_adam_alpha = dec_adam_alpha
        self.dec_adam_beta1 = dec_adam_beta1
        self.disc_adam_alpha = disc_adam_alpha
        self.disc_adam_beta1 = disc_adam_beta1
        self.rectifier = rectifier
        self.dropout_ratio = dropout_ratio

        self.enc = Encoder(img_width=self.img_width, img_height=self.img_height, 
                           color_channels=self.color_channels, encode_layers=self.encode_layers,
                           latent_width=self.latent_width, mode=self.mode)
        self.dec = Decoder(img_width=self.img_width,img_height=self.img_height,
                           color_channels=self.color_channels, decode_layers=self.decode_layers,
                           latent_width=self.latent_width, mode=self.mode)
        self.disc = Discriminator(img_width=self.img_width, img_height=self.img_height,
                                  color_channels=self.color_channels, disc_layers=self.disc_layers,
                                  latent_width=self.latent_width, mode=self.mode)
        if self.flag_gpu:
            self.enc = self.enc.to_gpu()
            self.dec = self.dec.to_gpu()
            self.disc = self.disc.to_gpu()

        self.enc_opt = O.Adam(alpha=self.enc_adam_alpha, beta1=self.enc_adam_beta1)
        self.dec_opt = O.Adam(alpha=self.dec_adam_alpha, beta1=self.dec_adam_beta1)
        self.disc_opt = O.Adam(alpha=self.disc_adam_alpha, beta1=self.disc_adam_beta1)

    def _encode(self, data, test=False):
        x = self.enc(data, test=test)
        mean, ln_var = F.split_axis(x, 2, 1)
        samp = np.random.standard_normal(mean.data.shape).astype('float32')
        samp = Variable(samp)
        if self.flag_gpu:
            samp.to_gpu()
        z = samp * F.exp(0.5*ln_var) + mean

        return z, mean, ln_var

    def _decode(self, z, test=False):
        x = self.dec(z, test=test, rectifier=self.rectifier)

        return x

    def _forward(self, batch, test=False):

        # TrainingSetのEncodeとDecode
        encoded, means, ln_vars = self._encode(batch, test=test)
        rec = self._decode(encoded, test=test)
        normer = reduce(lambda x, y: x*y, means.data.shape) # データ数
        kl_loss = F.gaussian_kl_divergence(means, ln_vars)/normer
        #print 'means={}'.format(means.data.shape)
        #print 'ln_vars={}'.format(ln_vars.data.shape)
        #print 'kl_loss={}, normer={}'.format(kl_loss.data, normer)

        # zのサンプル
        samp_p = np.random.standard_normal(means.data.shape).astype('float32')
        z_p = chainer.Variable(samp_p)

        if self.flag_gpu:
            z_p.to_gpu()

        rec_p = self._decode(z_p)

        disc_rec, conv_layer_rec = self.disc(rec, test=test, dropout_ratio=self.dropout_ratio)

        disc_batch, conv_layer_batch = self.disc(batch, test=test, dropout_ratio=self.dropout_ratio)

        disc_x_p, conv_layer_x_p = self.disc(rec_p, test=test, dropout_ratio=self.dropout_ratio)

        dif_l = F.mean_squared_error(conv_layer_rec, conv_layer_batch)

        return kl_loss, dif_l, disc_rec, disc_batch, disc_x_p

    def transform(self, data, test=False):
        #make sure that data has the right shape.
        if not type(data) == Variable:
            if len(data.shape) < 4:
                data = data[np.newaxis]
            if len(data.shape) != 4:
                raise TypeError("Invalid dimensions for image data. Dim = %s.                     Must be 4d array." % str(data.shape))
            if data.shape[1] != self.color_channels:
                if data.shape[-1] == self.color_channels:
                    data = data.transpose(0, 3, 1, 2)
                else:
                    raise TypeError("Invalid dimensions for image data. Dim = %s"
                                    % str(data.shape))
            data = Variable(data)
        else:
            if len(data.data.shape) < 4:
                data.data = data.data[np.newaxis]
            if len(data.data.shape) != 4:
                raise TypeError("Invalid dimensions for image data. Dim = %s.                     Must be 4d array." % str(data.data.shape))
            if data.data.shape[1] != self.color_channels:
                if data.data.shape[-1] == self.color_channels:
                    data.data = data.data.transpose(0, 3, 1, 2)
                else:
                    raise TypeError("Invalid dimensions for image data. Dim = %s"
                                    % str(data.shape))

        # Actual transformation.
        if self.flag_gpu:
            data.to_gpu()
        z = self._encode(data, test=test)[0]

        z.to_cpu()

        return z.data

    def inverse_transform(self, data, test=False):
        if not type(data) == Variable:
            if len(data.shape) < 2:
                data = data[np.newaxis]
            if len(data.shape) != 2:
                raise TypeError("Invalid dimensions for latent data. Dim = %s.                     Must be a 2d array." % str(data.shape))
            data = Variable(data)

        else:
            if len(data.data.shape) < 2:
                data.data = data.data[np.newaxis]
            if len(data.data.shape) != 2:
                raise TypeError("Invalid dimensions for latent data. Dim = %s.                     Must be a 2d array." % str(data.data.shape))
        assert data.data.shape[-1] == self.latent_width,            "Latent shape %d != %d" % (data.data.shape[-1], self.latent_width)

        if self.flag_gpu:
            data.to_gpu()
        out = self._decode(data, test=test)

        out.to_cpu()

        if self.mode == 'linear':
            final = out.data
        else:
            final = out.data.transpose(0, 2, 3, 1)

        return final

    def load_images(self, filepaths):
        def read(fname):
            im = Image.open(fname)
            im = np.float32(im)
            return im/255.
        x_all = np.array([read(fname) for fname in tqdm.tqdm(filepaths)])
        x_all = x_all.astype('float32')
        if self.mode == 'convolution':
            x_all = x_all.transpose(0, 3, 1, 2)
        print("Image Files Loaded!")
        return x_all

    def fit(self, img_data, gamma=1.0, save_freq=-1, pic_freq=-1, n_epochs=100, batch_size=100,
            weight_decay=True,  model_path='./VAEGAN_training_model/', img_path='./VAEGAN_training_images/',
            img_out_width=10, mirroring=False):
        width = img_out_width
        self.enc_opt.setup(self.enc)
        self.dec_opt.setup(self.dec)
        self.disc_opt.setup(self.disc)

        if weight_decay:
            self.enc_opt.add_hook(chainer.optimizer.WeightDecay(0.00001))
            self.dec_opt.add_hook(chainer.optimizer.WeightDecay(0.00001))
            self.disc_opt.add_hook(chainer.optimizer.WeightDecay(0.00001))

        n_data = img_data.shape[0]

        batch_iter = list(range(0, n_data, batch_size))
        n_batches = len(batch_iter)

        c_samples = np.random.standard_normal((width, self.latent_width)).astype(np.float32)
        save_counter = 0

        df_col = ['epoch', 'enc_loss', 'dec_loss', 'dis_loss', 'GAN_loss', 'like_loss', 'prior_loss', 'L_base', 'L_rec', 'L_p']
        self.loss_buf = pd.DataFrame(columns=df_col)
        for epoch in range(1, n_epochs + 1):
            print('epoch: %i' % epoch)
            t1 = time.time()
            indexes = np.random.permutation(n_data)
            sum_l_enc = 0.
            sum_l_dec = 0.
            sum_l_disc = 0.

            sum_l_gan = 0.
            sum_l_like = 0.
            sum_l_prior = 0.
            
            sum_l_b_gan = 0.
            sum_l_r_gan = 0.
            sum_l_s_gan = 0.
            count = 0
            for i in tqdm.tqdm(batch_iter):
                x = img_data[indexes[i: i + batch_size]]
                size = x.shape[0]
                if mirroring:
                    for j in range(size):
                        if np.random.randint(2):
                            x[j, :, :, :] = x[j, :, :, ::-1]
                x_batch = Variable(x)
                zeros = Variable(np.zeros(size, dtype=np.int32))
                ones = Variable(np.ones(size, dtype=np.int32))

                if self.flag_gpu:
                    x_batch.to_gpu()
                    zeros.to_gpu()
                    ones.to_gpu()

                # kl_loss : VAE中間表現のKL正則化ロス
                # dif_l : Discriminatorの中間層出力のMSE(学習データセットと再構成画像の中間出力のMSE)
                # disc_{rec, batch, samp} : Discriminator出力(2次元)
                kl_loss, dif_l, disc_rec, disc_batch, disc_samp = self._forward(x_batch)

                # Discriminator出力のloss計算
                L_batch_GAN = F.softmax_cross_entropy(disc_batch, ones)
                L_rec_GAN = F.softmax_cross_entropy(disc_rec, zeros)
                L_samp_GAN = F.softmax_cross_entropy(disc_samp, zeros)

                l_gan = (L_batch_GAN + L_rec_GAN + L_samp_GAN)/3.
                l_like = dif_l
                l_prior = kl_loss

                enc_loss = self.kl_ratio*l_prior + l_like
                dec_loss = gamma*l_like - l_gan
                disc_loss = l_gan

                self.enc_opt.zero_grads()
                enc_loss.backward()
                self.enc_opt.update()

                self.dec_opt.zero_grads()
                dec_loss.backward()
                self.dec_opt.update()

                self.disc_opt.zero_grads()
                disc_loss.backward()
                self.disc_opt.update()

                sum_l_enc += enc_loss.data
                sum_l_dec += dec_loss.data
                sum_l_disc += disc_loss.data

                sum_l_gan += l_gan.data
                sum_l_like += l_like.data
                sum_l_prior += l_prior.data
                
                sum_l_b_gan += L_batch_GAN.data
                sum_l_r_gan += L_rec_GAN.data
                sum_l_s_gan += L_samp_GAN.data
                count += 1

                #plot_data = img_data[indexes[:width]]

            sum_l_enc /= n_batches
            sum_l_dec /= n_batches
            sum_l_disc /= n_batches
            sum_l_gan /= n_batches
            sum_l_like /= n_batches
            sum_l_prior /= n_batches
            sum_l_b_gan /= n_batches
            sum_l_r_gan /= n_batches
            sum_l_s_gan /= n_batches
            msg = "enc_loss = {0}, dec_loss = {1} , disc_loss = {2}"
            msg2 = "gan_loss = {0}, sim_loss = {1}, kl_loss = {2}"
            print(msg.format(sum_l_enc, sum_l_dec, sum_l_disc))
            print(msg2.format(sum_l_gan, sum_l_like, sum_l_prior))
            t_diff = time.time()-t1
            print("time: %f\n\n" % t_diff)
            df_tmp = pd.DataFrame([[epoch, 
                                    sum_l_enc, sum_l_dec, sum_l_disc, sum_l_gan, sum_l_like, sum_l_prior, 
                                    sum_l_b_gan, sum_l_r_gan, sum_l_s_gan]], columns=df_col)
            self.loss_buf = self.loss_buf.append(df_tmp, ignore_index=True)



# ########### モデルのインスタンス生成
vg = VAEGAN(img_width=96, img_height=96, flag_gpu=True)

# 画像ファイルのロード、正規化、transpose
x_all = vg.load_images(paths)
print 'image_data_shape = {}'.format(x_all.shape)

#vg.fit(x_all, n_epochs=10, mirroring=True)
m_path = './out/model/'
im_path = './out/images/'
vg.fit(x_all, save_freq=2, pic_freq=30, n_epochs=4, model_path = m_path, img_path=im_path, mirroring=True)


# loss系列の保存
vg.loss_buf.to_csv('./out/loss_vaegan_faux.csv')




