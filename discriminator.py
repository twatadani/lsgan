# -*- coding: utf-8 -*-

########## discriminator.py ##########
#
# LSGAN Discriminator
# 
#
# created 2018/10/16 Takeyuki Watadani @ UT Radiology
#
########################################

import config as cf
import functions as f

import tensorflow as tf
import numpy as np

import os.path
import io
import math
import random

logger = cf.LOGGER

class Discriminator:
    '''DCGANのDiscriminatorを記述するクラス'''


    def __init__(self, global_step):
        self.global_step_tensor = global_step

        self.from_dataset = None # データセットからの入力画像
        self.from_generator = None # generatorからの入力画像
        self.output = None # 出力の確率
        self.optimizer = None
        self.train_op = None

    def define_forward(self, input, vreuse = None):
        '''判定する計算を返す'''
        
        with tf.variable_scope('D_network', reuse=vreuse):

            norm_factor = tf.constant(255.0, dtype=tf.float32,
                                      shape = (cf.MINIBATCHSIZE, cf.PIXELSIZE, cf.PIXELSIZE, 1))
            inreshaped = tf.divide(tf.reshape(input,
                                              shape = (-1, cf.PIXELSIZE, cf.PIXELSIZE, 1),
                                              name = 'D_inreshaped'), norm_factor)

            c1 = 'D_conv1'
            conv1 = f.apply_dobn(tf.layers.conv2d(inputs = inreshaped,
                                                  filters = 64,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  kernel_initializer = tf.keras.initializers.he_normal(),
                                                  activation = tf.nn.leaky_relu,
                                                  name = c1),
                                 c1)
            f.print_shape(conv1)

            c2 = 'D_conv2'
            conv2 = f.apply_dobn(tf.layers.conv2d(inputs = conv1,
                                                  filters = 128,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  kernel_initializer = tf.keras.initializers.he_normal(),
                                                  activation = tf.nn.leaky_relu,
                                                  name = c2),
                                 c2)
            f.print_shape(conv2)

            c3 = 'D_conv3'
            conv3 = f.apply_dobn(tf.layers.conv2d(inputs = conv2,
                                                  filters = 256,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  kernel_initializer = tf.keras.initializers.he_normal(),
                                                  activation = tf.nn.leaky_relu,
                                                  name = c3),
                                 c3)
            f.print_shape(conv3)

            c4 = 'D_conv4'
            conv4 = f.apply_dobn(tf.layers.conv2d(inputs = conv3,
                                                  filters = 512,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  kernel_initializer = tf.keras.initializers.he_normal(),
                                                  activation = tf.nn.leaky_relu,
                                                  name = c4),
                                 c4)
            f.print_shape(conv4)

            flatten = tf.layers.flatten(conv4,
                                        name = 'D_flatten')
            f.print_shape(flatten)

            fc = 'D_fully_connected'
            fully_connected = f.apply_dobn(tf.layers.dense(inputs = flatten,
                                                           units = 1,
                                                           kernel_initializer = tf.keras.initializers.he_normal(),
                                                           activation = tf.nn.sigmoid,
                                                           name = fc),
                                           fc)
            f.print_shape(fully_connected)
            return fully_connected

    def define_graph(self):
        '''discriminatorの計算グラフを定義する'''

        with tf.variable_scope('D_network'):

            self.from_dataset = f.obtain_minibatch_op()
            print(str(self.from_dataset))
        
            #zerosrandom = tf.random_normal(shape=(cf.MINIBATCHSIZE, 1),
            #                         mean=0.015,
            #                         stddev=0.015,
            #                         dtype=tf.float32,
            #                         name='D_zeros')

            #onesrandom = tf.random_normal(shape=(cf.MINIBATCHSIZE, 1),
            #                              mean = 0.985,
            #                              stddev = 0.015,
            #                              dtype = tf.float32,
            #                              name='onesrandom')

            #half = tf.constant(0.5,
            #                   dtype = tf.float32,
            #                   shape = (cf.MINIBATCHSIZE, 1),
            #                   name = 'half')



            #(shape = (cf.MINIBATCHSIZE, 1),
            #               dtype = tf.float32,
            #               name = 'ones')

            #twos = tf.constant(2.0,
            #                   dtype = tf.float32,
            #                   shape = (cf.MINIBATCHSIZE, 1),
            #                   name = 'twos')

            self.p_fake = self.define_forward(self.from_generator, vreuse=tf.AUTO_REUSE)
            self.p_real = self.define_forward(self.from_dataset, vreuse=tf.AUTO_REUSE)

            ones = tf.ones_like(self.p_real)

            self.loss = tf.reduce_mean(tf.add(
                tf.nn.l2_loss(tf.subtract(self.p_real, ones)),
                tf.nn.l2_loss(self.p_fake)))

            #self.loss = tf.reduce_mean(
            #    tf.add(tf.multiply(tf.pow(tf.subtract(self.p_real, ones), twos), half),
            #           tf.multiply(tf.pow(self.p_fake, twos), half)))
            

            #G_crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=zeros, logits=self.p_fake)
            #D_crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=onesrandom, logits=self.p_real)

            
            #self.loss = tf.reduce_mean(tf.add(D_crossentropy, G_crossentropy), name='D_loss')

            tf.summary.scalar(name = 'Discriminator loss', tensor = self.loss)
            D_vars = [x for x in tf.trainable_variables() if 'D_' in x.name]
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate = cf.D_LEARNING_RATE,
                                                    beta1 = cf.BETA_1,
                                                    name = 'D_optimizer')
            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=self.global_step_tensor,
                                                    var_list=D_vars,
                                                    name='D_train_op')

    def set_input_from_generator(self, generator):
        with tf.variable_scope('D_network'):
            self.from_generator = generator.output
        return

    def create_minibatch(self, session):
        '''データセットからミニバッチを作成する'''
        #minibatch_tf = self.from_dataset#f.obtain_minibatch()
        minibatch_np = session.run(self.from_dataset)
        return minibatch_np
