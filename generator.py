# -*- coding: utf-8 -*-

########## generator.py ##########
#
# LSGAN Generator
# 
#
# created 2018/10/16 Takeyuki Watadani @ UT Radiology
#
########################################

import config as cf
import functions as f

import tensorflow as tf

logger = cf.LOGGER

class Generator:
    '''DCGANのGeneratorを記述するクラス'''

    def __init__(self, discriminator, global_step):
        self.D = discriminator # 対となるDiscriminator


        self.global_step_tensor = global_step

        self.latent = None # 入力のlatent vector
        self.output = None # 出力のピクセルデータ
        self.loss = None # 損失関数
        self.optimizer = None # オプティマイザ
        self.train_op = None # 学習オペレーション

    def define_graph(self):
        '''generatorのネットワークを記述する'''

        self.projectedunits = 4 * 4 * 1024 #論文通りの数
        # カーネル初期値最適化用の変数
        self.inival = 2.0

        with tf.variable_scope('G_network', reuse=tf.AUTO_REUSE):

            # Ver. 5でplaceholderからTF内での生成に変更
            self.latent = tf.random_uniform(shape = (cf.MINIBATCHSIZE, cf.LATENT_VECTOR_SIZE),
                                            minval = -1.0,
                                            maxval = 1.0,
                                            dtype = tf.float32,
                                            seed = cf.SEED,
                                            name = 'G_latent_vector')
            f.print_shape(self.latent)

            pjname = 'G_projected'
            projected = f.apply_dobn(tf.layers.dense(inputs = self.latent,
                                                     units = self.projectedunits,
                                                     kernel_initializer = tf.initializers.random_uniform(minval = 0, maxval = self.inival),
                                                     name=pjname),
                                     pjname)
            f.print_shape(projected)
            self.initializer_value = tf.placeholder(dtype = tf.float32,
                                                    shape = projected.shape,
                                                    name = 'G_initializer_value')

            preshaped = tf.reshape(projected,
                                   shape=(-1, 4, 4, 1024),
                                   name='G_reshaped')
            f.print_shape(preshaped)

            tc1 = 'G_tconv1'
            tconv1 = f.apply_dobn(tf.layers.conv2d_transpose(inputs = preshaped,
                                                             filters = 512,
                                                             kernel_size = (4, 4),
                                                             strides = (2, 2),
                                                             padding = 'same',
                                                             trainable = True,
                                                             activation = tf.nn.relu,
                                                             kernel_initializer = tf.keras.initializers.he_uniform(),
                                                             name = tc1),
                                  tc1)
            f.print_shape(tconv1)
            tf.summary.tensor_summary(name='G_tconv1_kernel',
                                      tensor=tf.get_variable('G_tconv1/kernel'))
            
            tc2 = 'G_tconv2'
            tconv2 = f.apply_dobn(tf.layers.conv2d_transpose(inputs = tconv1,
                                                             filters = 256,
                                                             kernel_size = (5, 5),
                                                             strides = (2, 2),
                                                             padding = 'same',
                                                             trainable = True,
                                                             activation = tf.nn.relu,
                                                             kernel_initializer = tf.keras.initializers.he_uniform(),
                                                             name = tc2),
                                  tc2)
            f.print_shape(tconv2)
            
            tc3 = 'G_tconv3'
            tconv3 = f.apply_dobn(tf.layers.conv2d_transpose(inputs = tconv2,
                                                             filters = 128,
                                                             kernel_size = (5, 5),
                                                             strides = (2, 2),
                                                             padding = 'same',
                                                             activation = tf.nn.relu,
                                                             kernel_initializer = tf.keras.initializers.he_uniform(),
                                                             name = tc3),
                                  tc3)
            f.print_shape(tconv3)
            
            tc4 = 'G_tconv4'
            tconv4 = f.apply_dobn(tf.layers.conv2d_transpose(inputs = tconv3,
                                                             filters = 1,
                                                             kernel_size = (5, 5),
                                                             strides = (2, 2),
                                                             padding = 'same',
                                                             trainable = True,
                                                             activation = tf.nn.tanh,
                                                             kernel_initializer = tf.keras.initializers.he_uniform(),
                                                             name = tc4),
                                  tc4)
            f.print_shape(tconv4)
            self.kernel4 = tf.get_variable(name = 'G_tconv4/kernel',
                                           shape = (5, 5, 1, 128))

        
            mulcons = tf.constant(255.0 / 2.0,
                                  dtype = tf.float32,
                                  shape = (cf.MINIBATCHSIZE, cf.PIXELSIZE, cf.PIXELSIZE, 1))
            addcons = tf.constant(1.0,
                                  dtype = tf.float32,
                                  shape = (cf.MINIBATCHSIZE, cf.PIXELSIZE, cf.PIXELSIZE, 1))
            
            #0-255の値域にする
            self.output = tf.multiply(mulcons, tf.add(addcons, tconv4),
                                      name = 'G_output')
            f.print_shape(self.output)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=cf.G_LEARNING_RATE,
                                                    beta1 = cf.BETA_1,
                                                    name='G_optimizer')

            self.D.set_input_from_generator(self)
        return


    def define_graph_postD(self):

        onesrandom = tf.random_normal(shape=(cf.MINIBATCHSIZE, 1),
                                      mean = 1.0,
                                      stddev = 0.015,
                                      dtype = tf.float32,
                                      name='G_onesrandom')
        
        G_crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=onesrandom, logits=self.D.p_fake)

        self.loss = tf.reduce_mean(G_crossentropy, name='G_loss')
        G_vars = [x for x in tf.trainable_variables() if 'G_' in x.name]
        logger.info('G_vars: ' + str(len(G_vars)))
        for v in G_vars:
            logger.info(str(v))
            
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor, var_list=G_vars, name='G_train_op')
        tf.summary.scalar(name = 'Generator loss', tensor = self.loss)
                              
        self.mean_D_score = tf.reduce_mean(self.D.p_fake)
        tf.summary.scalar(name = 'D score', tensor = self.mean_D_score)

        self.gradients = self.optimizer.compute_gradients(
            self.loss,
            var_list = G_vars)

        return
