#!/bin/sh

PYCMD=/home/tokita/.pyenv/shims/python

TIME=`date '+%y/%m/%d %H:%M:%S'`
echo ${TIME} "ae_sp02"
${PYCMD} chainer_AE_image_conv.py --gpu=0 --label=ae_sp02 --epoch_num=200 --latent_dimension=10000 --adam_alpha=0.0001 --out_interval=100 > logs/ae_sp02-20160601.log

TIME=`date '+%y/%m/%d %H:%M:%S'`
echo ${TIME} "ae_sp03"
${PYCMD} chainer_AE_image_conv.py --gpu=0 --label=ae_sp03 --epoch_num=200 --latent_dimension=10000 --adam_alpha=0.001 --out_interval=100 > logs/ae_sp03-20160601.log

TIME=`date '+%y/%m/%d %H:%M:%S'`
echo ${TIME} "ae_sp04"
${PYCMD} chainer_AE_image_conv.py --gpu=0 --label=ae_sp04 --epoch_num=200 --latent_dimension=50000 --adam_alpha=0.0001 --out_interval=100 > logs/ae_sp04-20160601.log

TIME=`date '+%y/%m/%d %H:%M:%S'`
echo ${TIME} "ae_sp05"
${PYCMD} chainer_AE_image_conv.py --gpu=0 --label=ae_sp05 --epoch_num=200 --latent_dimension=10000 --adam_alpha=0.0001 --max_convolution_size=1024 --out_interval=100 > logs/ae_sp05-20160601.log


