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

        self.projectedunits = 4 * 4 * 1024
        # カーネル初期値最適化用の変数
        self.inival = 1.0

        with tf.variable_scope('G_network', reuse=tf.AUTO_REUSE):

            # Ver. 5でplaceholderからTF内での生成に変更
            self.latent = tf.random_normal(shape = (cf.MINIBATCHSIZE, cf.LATENT_VECTOR_SIZE),
                                            #minval = -1.0,
                                            #maxval = 1.0,
                                           mean = 0.0,
                                           stddev = 0.5,
                                           dtype = tf.float32,
                                           seed = cf.SEED,
                                           name = 'G_latent_vector')
            f.print_shape(self.latent)

            pjname = 'G_projected'
            projected = f.apply_dobn(
                tf.layers.dense(inputs = self.latent,
                                units = self.projectedunits,
                                kernel_initializer = tf.initializers.ones,
                                activation = tf.nn.relu,
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

            tcs2_1 = 'G_tconvs2_1'
            tconvs2_1 = f.apply_dobn(
                tf.layers.conv2d_transpose(inputs = preshaped,
                                           filters = 256,
                                           kernel_size = (3, 3),
                                           strides = (2, 2),
                                           padding = 'same',
                                           trainable = True,
                                           activation = tf.nn.relu,
                                           kernel_initializer = tf.keras.initializers.he_normal(),
                                           name = tcs2_1),
                tcs2_1)
            f.print_shape(tconvs2_1)
            tf.summary.tensor_summary(name='G_tconvs2_1_kernel',
                                      tensor=tf.get_variable('G_tconvs2_1/kernel'))
            

            tcs1_1 = 'G_tconvs1_1'
            tconvs1_1 = f.apply_dobn(
                tf.layers.conv2d_transpose(inputs = tconvs2_1,
                                           filters = 256,
                                           kernel_size = (3, 3),
                                           strides = (1, 1),
                                           padding = 'same',
                                           activation = tf.nn.relu,
                                           kernel_initializer = tf.keras.initializers.he_normal(),
                                           name = tcs1_1),
            tcs1_1)
            f.print_shape(tconvs1_1)

            tcs2_2 = 'G_tconvs2_2'
            tconvs2_2 = f.apply_dobn(
                tf.layers.conv2d_transpose(inputs = tconvs1_1,
                                           filters = 256,
                                           kernel_size = (3, 3),
                                           strides = (2, 2),
                                           padding = 'same',
                                           activation = tf.nn.relu,
                                           kernel_initializer = tf.keras.initializers.he_normal(),
                                           name = tcs2_2),
                tcs2_2)
            f.print_shape(tconvs2_2)

            tcs1_2 = 'G_tconvs1_2'
            tconvs1_2 = f.apply_dobn(
                tf.layers.conv2d_transpose(inputs = tconvs2_2,
                                           filters = 256,
                                           kernel_size = (3, 3),
                                           strides = (1, 1),
                                           padding = 'same',
                                           activation = tf.nn.relu,
                                           kernel_initializer = tf.keras.initializers.he_normal(),
                                           name = tcs1_2),
                tcs1_2)
            f.print_shape(tconvs1_2)
            
            tcs2_3 = 'G_tconvs2_3'
            tconvs2_3 = f.apply_dobn(
                tf.layers.conv2d_transpose(inputs = tconvs1_2,
                                           filters = 128,
                                           kernel_size = (3, 3),
                                           strides = (2, 2),
                                           padding = 'same',
                                           activation = tf.nn.relu,
                                           kernel_initializer = tf.keras.initializers.he_normal(),
                                           name = tcs2_3),
                tcs2_3)
            f.print_shape(tconvs2_3)
            
            tcs2_4 = 'G_tconvs2_4'
            tconvs2_4 = f.apply_dobn(
                tf.layers.conv2d_transpose(inputs = tconvs2_3,
                                           filters = 64,
                                           kernel_size = (3, 3),
                                           strides = (2, 2),
                                           padding = 'same',
                                           activation = tf.nn.relu,
                                           kernel_initializer = tf.keras.initializers.he_normal(),
                                           name = tcs2_4),
                tcs2_4)
            f.print_shape(tconvs2_4)
            self.kernel4 = tf.get_variable(name = 'G_tconvs2_4/kernel',
                                           shape = (3, 3, 64, 128))

            tcs1_3 = 'G_tconvs1_3'
            tconvs1_3 = f.apply_dobn(
                tf.layers.conv2d_transpose(inputs = tconvs2_4,
                                           filters = 1,
                                           kernel_size = (3, 3),
                                           strides = (1, 1),
                                           padding = 'same',
                                           activation = tf.nn.tanh,
                                           name = tcs1_3),
                tcs1_3)
        
            mulcons = tf.constant(255.0 / 2.0,
                                  dtype = tf.float32,
                                  shape = (cf.MINIBATCHSIZE, cf.PIXELSIZE, cf.PIXELSIZE, 1))
            addcons = tf.constant(1.0,
                                  dtype = tf.float32,
                                  shape = (cf.MINIBATCHSIZE, cf.PIXELSIZE, cf.PIXELSIZE, 1))
            
            #0-255の値域にする
            self.output = tf.multiply(mulcons, tf.add(addcons, tconvs1_3),
                                      name = 'G_output')
            f.print_shape(self.output)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=cf.G_LEARNING_RATE,
                                                    beta1 = cf.BETA_1,
                                                    name='G_optimizer')

            self.D.set_input_from_generator(self)
        return


    def define_graph_postD(self):

        #half = tf.constant(0.5,
        #                   dtype = tf.float32,
        #                   shape = (cf.MINIBATCHSIZE, 1))

        ones = tf.ones_like(self.D.p_fake)

        #twos = tf.constant(2.0,
        #                   dtype = tf.float32,
        #                   shape = (cf.MINIBATCHSIZE, 1))

        self.loss = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(self.D.p_fake, ones)))

        #self.loss = tf.reduce_mean(tf.multiply(tf.pow(tf.subtract(self.D.p_fake, ones), twos), half))
                    

        #onesrandom = tf.random_normal(shape=(cf.MINIBATCHSIZE, 1),
        #                              mean = 1.0,
        #                              stddev = 0.015,
        #                              dtype = tf.float32,
        #                              name='G_onesrandom')
        
        #G_crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=onesrandom, logits=self.D.p_fake)

        #self.loss = tf.reduce_mean(G_crossentropy, name='G_loss')
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
