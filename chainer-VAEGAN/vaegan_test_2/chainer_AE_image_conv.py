
# coding: utf-8

# # オートエンコーダー（一般画像）
# * アニメ顔とかのオートエンコーダーを作ってみる
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
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.vae import gaussian_kl_divergence

import tables
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# AEモデルクラスのimport
from model_AE import AE


parser = argparse.ArgumentParser(description='option')
parser.add_argument('--gpu', '-g', default=0, type=int)
parser.add_argument('--label', '-l', default='test', type=str)
parser.add_argument('--img_dir', '-img', default='../DataSet/Stimuli.mat', type=str)
parser.add_argument('--img_size', default=128, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--epoch_num', default=10, type=int)
parser.add_argument('--latent_dimension', default=100, type=int)
parser.add_argument('--max_convolution_size', default=512, type=int)
parser.add_argument('--adam_alpha', default=0.001, type=float)
parser.add_argument('--adam_beta1', default=0.9, type=float)
parser.add_argument('--adam_beta2', default=0.999, type=float)
parser.add_argument('--out_interval', default=100, type=int)
args = parser.parse_args()

# ## GPU設定
gpu_flag = args.gpu
if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np


print "##### start : chainer_AE_image_conv.py"


# ## 学習パラメータの設定
batchsize = args.batch_size # ミニバッチのサイズ
n_epoch = args.epoch_num     # epoch数
n_latent = args.latent_dimension   # 潜在変数の次元(DCGANで言うところのプライヤーベクトルの次元)
conv_size = args.max_convolution_size  # convolution層の最大チャネルサイズ

# Optimizer(Adam)
al = args.adam_alpha
b1 = args.adam_beta1
b2 = args.adam_beta2

# 学習データセット
# img_size
size = args.img_size
# image path
image_dir = args.img_dir
# 訓練データの割合
train_rate = 0.7

# モデルの出力インターバル
model_interval = args.out_interval
# モデルファイルの出力先
out_model_dir = './out/out_models_%s'%args.label
try:
    os.mkdir(out_model_dir)
except:
    pass
## 学習パラメータの設定ここまで

# ## パラメータの確認
print('epoch_num={}'.format(n_epoch))
print('latent_dimension={}'.format(n_latent))
print('AdamParameter(alpha, b1, b2) = {}, {}, {}'.format(al, b1, b2))
print('output_directory={}'.format(out_model_dir))


# ## データのロード
# load images
print("now loading image_files : {}".format(image_dir))
## 刺激画像
fileStimuli = "../DataSet/Stimuli.mat"
dataStimuli = tables.openFile(fileStimuli)
imgStimuli = dataStimuli.get_node('/st')[:]
imgStimuliVd = dataStimuli.get_node('/sv')[:]
print 'DataShape [Stimuli] : {}'.format(imgStimuli.shape)
print 'DataShape [Stimuli for varidation] : {}'.format(imgStimuliVd.shape)

## 15fpsで映像を見せているので、とりあえず15枚ずつ間引く
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


## 画像を描画する関数
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


# ## Optimizerの設定
# モデルの設定
model = AE(input_size=size, n_latent=n_latent, output_ch=conv_size)
if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()
xp = np if gpu_flag < 0 else cuda.cupy

# Optimizerを定義する
optimizer = optimizers.Adam(alpha=al, beta1=b1, beta2=b2)
optimizer.setup(model)

sys.stdout.flush()

# ## 訓練の実行
loss_arr = []
print "epoch, train_mean_loss"
for epoch in six.moves.range(1, n_epoch + 1):
    # training
    ## 訓練データのsampler
    perm = np.random.permutation(N_train)
    ## lossのbuffer
    sum_loss = 0       # total loss
    sum_rec_loss = 0   # reconstruction loss
    ## バッチ学習
    for i in six.moves.range(0, N_train, batchsize):
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]])) # バッチ分のデータの抽出
        
        model.zerograds()
        loss = model.get_loss_func()(x)
        loss.backward()
        optimizer.update()
        
        sum_loss += float(model.loss.data) * len(x.data)

    print('{}, {}'.format(epoch, sum_loss / N_train))
    loss_arr.append(float(sum_loss)/N_train)
    
    # モデルの保存
    if epoch%model_interval==0:
        serializers.save_hdf5("%s/model_AE_%05d.h5"%(out_model_dir, epoch), model)
    sys.stdout.flush()
# モデルの保存(最終モデル)
if epoch%model_interval!=0:
    serializers.save_hdf5("%s/model_AE_%05d.h5"%(out_model_dir, epoch), model)


# ## 結果の可視化
## Lossの変化
plt.figure(figsize=(7, 4))
plt.clf()
plt.plot(range(len(loss_arr)), loss_arr, color="#0000FF", label="total_loss")
plt.legend()
plt.savefig('%s/learning_curv_%04d.png'%(out_model_dir, epoch))


## 描画テスト (Closed test)
test_ind = np.random.permutation(N_train)[:10]
print "Reconstruct Test [Closed Test]"
print test_ind
test = chainer.Variable(xp.asarray(x_train[test_ind]), volatile='on')
y = model(test)
draw_img_rgb(x_train[test_ind], ("%s/image_closed_input.png"%(out_model_dir)))
draw_img_rgb(y.data.get(), ("%s/image_closed_reconstruct.png"%(out_model_dir)))


## 描画テスト (Open test) 
test_ind = np.random.permutation(N_test)[:10]
print "Reconstruct Test [Open Test]"
print test_ind
test = chainer.Variable(xp.asarray(x_test[test_ind]), volatile='on')
y = model(test)
draw_img_rgb(x_test[test_ind], ("%s/image_open_input.png"%(out_model_dir)))
draw_img_rgb(y.data.get(), ("%s/image_open_reconstruct.png"%(out_model_dir)))


## 描画テスト (Open test, 固定画像) 
test_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
print "Reconstruct Test [Open-fixedIndex Test]"
print test_ind
test = chainer.Variable(xp.asarray(x_test[test_ind]), volatile='on')
y = model(test)
draw_img_rgb(x_test[test_ind], ("%s/image_fixed_input.png"%(out_model_dir)))
draw_img_rgb(y.data.get(), ("%s/image_fixed_reconstruct.png"%(out_model_dir)))


## draw images from randomly sampled z
z = chainer.Variable(xp.random.normal(0, 1, (10, n_latent)).astype(np.float32))
x = model.decode(z)
print "Reconstruct Test [random input]"
draw_img_rgb(x.data.get(), ("%s/image_random_reconstruct.png"%(out_model_dir)))




