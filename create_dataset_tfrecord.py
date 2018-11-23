#!/usr/bin/env python
# -*- coding: utf-8 -*-

########## create_dataset_tfrecord.py ##########
#
# LSGANのTFRecord形式データセット作成用ユーティリティ
#
# created 2018/10/15 Takeyuki Watadani @ UT Radiology
#
########################################

# import section
import config as cf

import random
import glob
import os.path

import tensorflow as tf
from PIL import Image

import numpy as np

##########

logger = cf.LOGGER

########## ここからメインプログラム ##########

# 乱数を初期化
random.seed()

# 教師用データセットを読み込む
logger.info('学習用データセットを作成します。')

logger.info('教師用画像サーチパス: ' + cf.TRAIN_DATA_PATH)
# 画像ファイルのリストアップ
globstr = os.path.join(cf.TRAIN_DATA_PATH, '**/*' + cf.TRAIN_DATA_EXT)
img_list = glob.glob(globstr, recursive=True)

logger.info(str(len(img_list)) + '個の画像を確認しました。画像順をシャッフルします。')
random.shuffle(img_list)

# データセットディレクトリのチェックと作成
if not os.path.exists(cf.DATASET_PATH):
    os.makedirs(cf.DATASET_PATH)

# データセットを書き込んだ回数
record_count = 0

tfrpath = ''

while record_count < cf.TRAIN_DATA_NUMBER:

    if record_count % cf.TFRECORD_CAPACITY == 0:
        n = record_count // cf.TFRECORD_CAPACITY
        tfrpath = os.path.join(cf.DATASET_PATH, cf.TRAIN_PREFIX + '-' + str(n) + '.tfrecord')
    
    with tf.python_io.TFRecordWriter(tfrpath) as writer:

        for i in range(cf.TFRECORD_CAPACITY):
            img = Image.open(img_list[i])
            if img.mode != 'L':
                grayimg = img.convert(mode='L')
            else:
                grayimg = img
            imgsize = grayimg.size
            imagesize = (cf.PIXELSIZE, cf.PIXELSIZE)
            if grayimg.size != imagesize:
                resized = grayimg.resize((cf.PIXELSIZE, cf.PIXELSIZE), resample=Image.BICUBIC)
            else:
                resized = grayimg
            imgnp = np.asarray(resized, dtype=np.float32)
            
            features = tf.train.Features(feature=
                { 
                    'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgnp.tobytes()]))
                }
            )

            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            record_count += 1

        logger.info(str(record_count) + '件のTFRecord書き込みが終了しました。')


logger.info('データセット作成を終了します。')
