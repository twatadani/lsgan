#! /usr/bin/env python
# -*- coding: utf-8 -*-

##########functions.py ##########
#
#
# LSGAN用のユーティリティ－関数群
#
# created 2018/10/16 Takeyuki Watadani @ UT Radiology
#
########################################

import config as cf

import os.path
import glob

import tensorflow as tf
import numpy as np

logger = cf.LOGGER

def print_shape(layer):
    '''loggerからtensorflowのlayer shapeを出力する'''
    logger.info(layer.name + ': ' + str(layer.shape))
    return

def apply_dropout(layer):
    '''layerに対してdropoutを設定する。返り値はdropout layer'''
    return tf.layers.dropout(inputs = layer,
                             rate = cf.DROPOUT_RATE,
                             name = 'dropout_' + layer.name)

def apply_batchnorm(layer):
    '''layerに対してbatch normalizationを設定する。返り値はbn layer'''
    return tf.layers.batch_normalization(inputs = layer,
                                         name = 'batchnorm_' + layer.name)

def apply_dobn(layer, basename):
    '''layerに対してdropout, batch_normalizationを設定する。
    返り値は出力となるlayer'''

    do = tf.layers.dropout(inputs = layer,
                           rate = cf.DROPOUT_RATE,
                           name = 'dropout_' + str(basename))
    bn = tf.layers.batch_normalization(inputs = do,
                                       name = 'batchnorm_' + str(basename))
    return bn

def obtain_minibatch_op():
    '''データセットのTFRecordからミニバッチを取得するオペレーションを定義する'''
    filestr = cf.TRAIN_PREFIX + '-*.tfrecord'
    globstr = os.path.join(cf.DATASET_PATH, filestr)
    tfrecords = glob.glob(globstr)
    dataset_shuffle_repeat = tf.data.TFRecordDataset(tfrecords).apply(
        tf.contrib.data.shuffle_and_repeat(buffer_size = cf.MINIBATCHSIZE * 2))
    dataset_map_batch = dataset_shuffle_repeat.apply(tf.contrib.data.map_and_batch(
        map_func=record_parse,
        batch_size=cf.MINIBATCHSIZE,
        num_parallel_calls = cf.DATASET_PARALLEL_CALLS))

    prefetch = dataset_map_batch.prefetch(buffer_size = 10)
    iterator = prefetch.make_one_shot_iterator()
    return iterator.get_next()

def record_parse(example):
    '''データセットの各Exampleをパースする関数'''
    features = tf.parse_single_example(example,
                                       features = {
                                           'img': tf.FixedLenFeature([], dtype=tf.string)
                                       }
    )
    imgbytes = features['img']
    parsed = tf.io.decode_raw(imgbytes, out_type=tf.float32)
    return parsed
