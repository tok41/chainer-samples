#!/bin/sh

PYCMD=/home/tokita/.pyenv/shims/python

TIME=`date '+%y/%m/%d %H:%M:%S'`
echo ${TIME} "vae_sp02"
${PYCMD} chainer_VAE_image_conv.py --gpu=1 --label=vae_sp02 --epoch_num=200 --latent_dimension=10000 --kl_weight=1.0 --adam_alpha=0.0001 --out_interval=100 > logs/vae_sp02-20160601.log

TIME=`date '+%y/%m/%d %H:%M:%S'`
echo ${TIME} "vae_sp03"
${PYCMD} chainer_VAE_image_conv.py --gpu=1 --label=vae_sp03 --epoch_num=200 --latent_dimension=10000 --kl_weight=10.0 --adam_alpha=0.0001 --out_interval=100 > logs/vae_sp03-20160601.log

TIME=`date '+%y/%m/%d %H:%M:%S'`
echo ${TIME} "vae_sp04"
${PYCMD} chainer_VAE_image_conv.py --gpu=1 --label=vae_sp04 --epoch_num=200 --latent_dimension=10000 --kl_weight=0.1 --adam_alpha=0.0001 --out_interval=100 > logs/vae_sp04-20160601.log

TIME=`date '+%y/%m/%d %H:%M:%S'`
echo ${TIME} "vae_sp05"
${PYCMD} chainer_VAE_image_conv.py --gpu=1 --label=vae_sp05 --epoch_num=200 --latent_dimension=50000 --kl_weight=1.0 --adam_alpha=0.0001 --out_interval=100 > logs/vae_sp05-20160601.log
