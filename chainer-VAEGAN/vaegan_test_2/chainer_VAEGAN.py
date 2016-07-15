
# coding: utf-8

# # オートエンコーダー（一般画像）
# VAE+GAN

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

from model_VAEGAN import Encoder, Decoder, Discriminator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='option')
parser.add_argument('--gpu', '-g', default=2, type=int)
parser.add_argument('--label', '-l', default='test', type=str)
parser.add_argument('--train_dir', default='/home/tokita/projects/cinet/YouTubePriors_flv4/DividedImages/images_resize/sample_train', type=str)
parser.add_argument('--test_dir', default='/home/tokita/projects/cinet/YouTubePriors_flv4/DividedImages/images_resize/sample_test', type=str)
parser.add_argument('--img_size', default=96, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--epoch_num', default=10, type=int)
parser.add_argument('--latent_dimension', default=1000, type=int)
parser.add_argument('--max_convolution_size', default=512, type=int)
parser.add_argument('--adam_alpha', default=0.0001, type=float)
parser.add_argument('--adam_beta1', default=0.5, type=float)
parser.add_argument('--adam_beta2', default=0.999, type=float)
parser.add_argument('--kl_weight', default=1.0, type=float)
parser.add_argument('--gamma', default=1.0, type=float)
parser.add_argument('--out_interval', default=100, type=int)
args = parser.parse_args()


# ##### データのロード用のメソッド
def load_image(dir_name):
    fs = os.listdir(dir_name)
    data_set = []
    for fn in fs:
        f = open('%s/%s'%(dir_name, fn), 'rb')
        img_bin = f.read()
        img = np.asarray(Image.open(StringIO(img_bin)).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
        data_set.append(img)
        f.close()
    data_set = np.asarray(data_set)
    # 正規化(0~1に)
    data_set /= 255
    return data_set
# ##### 画像を描画する関数
def draw_img_rgb(data, fig_path):
    n = data.shape[0]
    plt.figure(figsize=(n*2, 2))
    plt.clf()
    data /= data.max()
    cnt = 1
    for idx in np.arange(n):
        plt.subplot(1, n, cnt)
        tmp = data[idx,:,:,:].transpose(1,2,0)
        plt.imshow(tmp)
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        cnt+=1
    plt.savefig(fig_path)


# ## GPU設定
gpu_flag = args.gpu
if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np


print "##### start : chainer_VAEGAN.py"


# ## 学習パラメータの設定
batchsize = args.batch_size # ミニバッチのサイズ
n_epoch = args.epoch_num     # epoch数
n_latent = args.latent_dimension   # 潜在変数の次元(DCGANで言うところのプライヤーベクトルの次元)
conv_size = args.max_convolution_size  # convolution層の最大チャネルサイズ
kl_weight = args.kl_weight   # KL正則化項の重み
gamma = args.gamma # VAEとGANの重みを変えるためのパラメータ

# Optimizer(Adam)
al = args.adam_alpha
b1 = args.adam_beta1
b2 = args.adam_beta2

# 学習データセット
# img_size
size = args.img_size
# image path
#image_dir = args.img_dir
dir_train = args.train_dir
dir_test = args.test_dir

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
print('KL_Weight={}'.format(kl_weight))
print('AdamParameter(alpha, b1, b2) = {}, {}, {}'.format(al, b1, b2))
print('output_directory={}'.format(out_model_dir))


# ## データのロード
x_train = load_image(dir_train)
x_test = load_image(dir_test)
print 'x_train.shape={}'.format(x_train.shape)
print 'x_test.shape={}'.format(x_test.shape)
N_train = x_train.shape[0]
N_test = x_test.shape[0]
print('N_train={}, N_test={}'.format(N_train, N_test))

sys.stdout.flush()

# ## モデルの定義
encode = Encoder(input_size=size, n_latent=n_latent, output_ch=conv_size)
decode = Decoder(input_size=size, n_latent=n_latent, output_ch=conv_size)
disc = Discriminator(input_size=size, n_latent=n_latent, output_ch=conv_size)

# GPU設定
if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    encode.to_gpu()
    decode.to_gpu()
    disc.to_gpu()
xp = np if gpu_flag < 0 else cuda.cupy

# Optimizerの設定
o_enc = optimizers.Adam(alpha=al, beta1=b1, beta2=b2)
o_dec = optimizers.Adam(alpha=al, beta1=b1, beta2=b2)
o_dis = optimizers.Adam(alpha=al, beta1=b1, beta2=b2)
o_enc.setup(encode)
o_dec.setup(decode)
o_dis.setup(disc)

# ## 訓練の実行
print 'epoch, loss_enc, loss_dec, loss_dis, gan_loss, like_loss, prior_loss'
df_col = ['epoch', 'enc_loss', 'dec_loss', 'dis_loss', 'GAN_loss', 'like_loss', 'prior_loss', 'L_base', 'L_rec', 'L_p']
loss_buf = pd.DataFrame(columns=df_col)
for epoch in six.moves.range(1, n_epoch + 1):
    # training
    ## 訓練データのsampler
    perm = np.random.permutation(N_train)
    ## lossのbuffer
    sum_enc_loss = 0.
    sum_dec_loss = 0.
    sum_dis_loss = 0.
    sum_gan_loss = 0.
    sum_like_loss = 0.
    sum_prior_loss = 0.
    sum_L_base = 0.
    sum_L_rec = 0.
    sum_L_p = 0.
    ## バッチ学習
    for i in six.moves.range(0, N_train, batchsize):
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]])) # バッチ分のデータの抽出
        ##### ForwardとLossの計算
        # KL距離
        mu, ln_var = encode(x, test=False)
        x_rec = decode(mu, sigmoid=True)
        batchsize = len(mu.data)
        kl_loss = gaussian_kl_divergence(mu, ln_var) / reduce(lambda x,y:x*y, mu.data.shape)

        # ランダムzの生成とランダムzでのdecode ## zはN(0, 1)から生成
        z_p = xp.random.standard_normal(mu.data.shape).astype('float32')
        z_p = chainer.Variable(z_p)
        x_p = decode(z_p)

        # Discriminatorの出力を得る
        d_x_rec, h_out_rec = disc(x_rec)
        d_x_base, h_out_base = disc(x)
        d_x_p, h_out_p = disc(x_p)
        # Discriminatorのsoftmax_cross_entropy
        L_rec = F.softmax_cross_entropy(d_x_rec, Variable(xp.zeros(batchsize, dtype=np.int32)))
        L_base = F.softmax_cross_entropy(d_x_base, Variable(xp.ones(batchsize, dtype=np.int32)))
        L_p = F.softmax_cross_entropy(d_x_p, Variable(xp.zeros(batchsize, dtype=np.int32)))

        # Reconstruction Errorを得る(Discriminatorの中間出力の誤差)
        rec_loss = (F.mean_squared_error(h_out_rec[0], h_out_base[0])
                    + F.mean_squared_error(h_out_rec[1], h_out_base[1])
                    + F.mean_squared_error(h_out_rec[2], h_out_base[2])
                    + F.mean_squared_error(h_out_rec[3], h_out_base[3]) ) / 4.0
        ##### Loss計算ここまで
        
        l_gan = (L_base + L_rec + L_p) / 3.0
        l_like = rec_loss
        l_prior = kl_loss

        enc_loss = kl_weight * l_prior + l_like
        dec_loss = gamma*l_like - l_gan
        dis_loss = l_gan

        ##### パラメータの更新
        # Encoder
        o_enc.zero_grads()
        enc_loss.backward()
        o_enc.update()
        # Decoder
        o_dec.zero_grads()
        dec_loss.backward()
        o_dec.update()
        #Discriminator
        o_dis.zero_grads()
        dis_loss.backward()
        o_dis.update()
        ##### パラメータの更新ここまで

        sum_enc_loss += enc_loss.data
        sum_dec_loss += dec_loss.data
        sum_dis_loss += dis_loss.data

        sum_gan_loss += l_gan.data
        sum_like_loss += l_like.data
        sum_prior_loss += l_prior.data

        sum_L_base += L_base.data
        sum_L_rec += L_rec.data
        sum_L_p += L_p.data
    print '{}, {}, {}, {}, {}, {}, {}'.format(epoch, sum_enc_loss, sum_dec_loss, sum_dis_loss, sum_gan_loss, sum_like_loss, sum_prior_loss)
    df_tmp = pd.DataFrame([[epoch, sum_enc_loss, sum_dec_loss, sum_dis_loss, sum_gan_loss, sum_like_loss, sum_prior_loss, sum_L_base, sum_L_rec, sum_L_p]], columns=df_col)
    loss_buf = loss_buf.append(df_tmp, ignore_index=True)
    # モデルの保存
    if epoch%model_interval==0:
        serializers.save_hdf5("%s/model_VAEGAN_Enc_%05d.h5"%(out_model_dir, epoch), encode)
        serializers.save_hdf5("%s/model_VAEGAN_Dec_%05d.h5"%(out_model_dir, epoch), decode)
        serializers.save_hdf5("%s/model_VAEGAN_Dis_%05d.h5"%(out_model_dir, epoch), disc)
    sys.stdout.flush()
# モデルの保存(最終モデル)
if epoch%model_interval!=0:
    serializers.save_hdf5("%s/model_VAEGAN_Enc_%05d.h5"%(out_model_dir, epoch), encode)
    serializers.save_hdf5("%s/model_VAEGAN_Dec_%05d.h5"%(out_model_dir, epoch), decode)
    serializers.save_hdf5("%s/model_VAEGAN_Dis_%05d.h5"%(out_model_dir, epoch), disc)

# lossの保存
loss_buf.to_csv("%s/loss_VAEGAN_%05d.csv"%(out_model_dir, epoch))

## 描画テスト (Closed test)
test_ind = np.random.permutation(N_train)[:10]
print "Reconstruct Test [Closed Test]"
print test_ind
x = chainer.Variable(xp.asarray(x_train[test_ind]), volatile='on')
mu, ln_var = encode(x, test=False)
x_rec = decode(mu, sigmoid=True)
draw_img_rgb(x_train[test_ind], ("%s/image_closed_input.png"%(out_model_dir)))
draw_img_rgb(x_rec.data.get(), ("%s/image_closed_reconstruct.png"%(out_model_dir)))

## 描画テスト (Open test) 
test_ind = np.random.permutation(N_test)[:10]
print "Reconstruct Test [Open Test]"
print test_ind
x = chainer.Variable(xp.asarray(x_test[test_ind]), volatile='on')
mu, ln_var = encode(x, test=False)
x_rec = decode(mu, sigmoid=True)
draw_img_rgb(x_train[test_ind], ("%s/image_open_input.png"%(out_model_dir)))
draw_img_rgb(x_rec.data.get(), ("%s/image_open_reconstruct.png"%(out_model_dir)))


