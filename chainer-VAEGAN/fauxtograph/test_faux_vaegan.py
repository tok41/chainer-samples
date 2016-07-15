
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


from vaegan import Encoder, Decoder, Discriminator, EncDec
from fauxtograph import VAEGAN, get_paths, image_resize


paths = get_paths('/home/tokita/projects/cinet/YouTubePriors_flv4/DividedImages/images_resize/sample_train/')
print len(paths)


# ########### モデルのインスタンス生成
vg = VAEGAN(img_width=96, img_height=96, flag_gpu=True)

# 画像ファイルのロード、正規化、transpose
x_all = vg.load_images(paths)
print 'image_data_shape = {}'.format(x_all.shape)

#vg.fit(x_all, n_epochs=10, mirroring=True)
m_path = '/home/tokita/workspace/projects/NTTD_CiNet/modelAutoEncoder/fauxtograph/out/model/'
im_path = '/home/tokita/workspace/projects/NTTD_CiNet/modelAutoEncoder/fauxtograph/out/images/'
vg.fit(x_all, save_freq=2, pic_freq=-1, n_epochs=4, model_path = m_path, img_path=im_path, mirroring=True)


# loss系列の保存
vg.loss_buf.to_csv('./out/loss_vaegan_faux.csv')




