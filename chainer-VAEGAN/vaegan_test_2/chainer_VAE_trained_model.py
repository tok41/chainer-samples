
# coding: utf-8

# # オートエンコーダー（一般画像）
# 学習済みのモデルファイルを入力して中間表現ベクトルと再構成画像を出力する
# * 中間層にKL正規化項を入れ、変分AEにしている
# * ネットワークはConvolution - Deconcolutionネットワークを使う

import sys, os
import numpy as np
import pandas as pd
from PIL import Image
from StringIO import StringIO
import math
import argparse
import six

import chainer
from chainer import Variable

import tables
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# AEモデルクラスのimport
from model_VAE import VAE, EncodeDecode

## 画像を描画して保存する関数
def draw_img_rgb(data, fig_path):
    n = data.shape[0]
    plt.figure(figsize=(n*2, 2))
    plt.clf()
    data /= data.max()
    cnt = 1
    for idx in np.arange(n):
        plt.subplot(1, n, cnt)
        tmp = data[idx,:,:,:].transpose(2,1,0)
        plt.imshow(tmp)
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        cnt+=1
    plt.savefig(fig_path)

parser = argparse.ArgumentParser(description='option')
parser.add_argument('--img_file', '-img', default='../DataSet/Stimuli.mat', type=str)
parser.add_argument('--model_file', default='out/out_models_vae_stim01/model_VAE_00500.h5', type=str)
args = parser.parse_args()

# ## データのロード
# load images
print("loading image_files : {}".format(args.img_file))
## 刺激画像
fileStimuli = args.img_file
dataStimuli = tables.openFile(fileStimuli)
imgStimuli = dataStimuli.get_node('/st')[:]
imgStimuliVd = dataStimuli.get_node('/sv')[:]
print 'DataShape [Stimuli] : {}'.format(imgStimuli.shape)
print 'DataShape [Stimuli for varidation] : {}'.format(imgStimuliVd.shape)

## 15fpsで映像を見せているので、15枚ずつ間引く
x_train = imgStimuli[np.arange(0, imgStimuli.shape[0], 15)]
x_train = x_train.astype(np.float32)
### テストデータ
x_test = imgStimuliVd[np.arange(0, imgStimuliVd.shape[0], 15)]
x_test = x_test.astype(np.float32)

## データの正規化
x_train = x_train / 255
x_test = x_test / 255
print 'x_train.shape={}'.format(x_train.shape)
print 'x_test.shape={}'.format(x_test.shape)

N_train = x_train.shape[0]
N_test = x_test.shape[0]
print('N_train={}, N_test={}'.format(N_train, N_test))


## モデルのロード
print 'model_file : {}'.format(args.model_file)
ae = EncodeDecode(model_file=args.model_file)

## 中間表現ベクトルの取得
test_ind = np.random.permutation(N_test)[:10]
test = chainer.Variable(np.asarray(x_test[test_ind]), volatile='on')
z = ae.getLatentVector(test)
print 'latent_vector : {}'.format(z.data.shape)

## 再構成画像の取得
y = ae.getReconstructImage(z)

## 画像の出力
draw_img_rgb(test.data, "input_images.png")
draw_img_rgb(y.data, "reconstruct_images.png")


