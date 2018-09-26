# the discriminator class

import tensorflow as tf
import config


class Discriminator():
    def __init__(self, n_node, node_emd_init, node_features):
        self.n_node = n_node
        self.node_emd_init = node_emd_init
        self.node_features = node_features

        with tf.variable_scope('discriminator'):
            self.node_embed = tf.get_variable(name="node_embed", shape=self.node_emd_init.shape,
                                              initializer=tf.constant_initializer(self.node_emd_init), trainable=True)

            self.node_b = tf.Variable(tf.zeros([self.n_node]))

            self.node_features = tf.get_variable(name="node_features", shape=self.node_features.shape,
                                              initializer=tf.constant_initializer(self.node_features), dtype=tf.bool, trainable=False)

            self.node_features_w = tf.Variable( initial_value=tf.random_normal([config.n_features,1]),trainable=True)
        self.node_features_w *= self.node_features_w #W^2 chnage made here
            

        self.q_node = tf.placeholder(tf.int32)
        self.rel_node = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)
        self.q_embedding = tf.nn.embedding_lookup(self.node_embed, self.q_node)
        self.rel_embedding = tf.nn.embedding_lookup(self.node_embed, self.rel_node)

        self.q_features = tf.nn.embedding_lookup(self.node_features, self.q_node)
        self.rel_features = tf.nn.embedding_lookup(self.node_features, self.rel_node)


        self.i_bias = tf.gather(self.node_b, self.rel_node)
        self.score = tf.reduce_sum(tf.multiply(self.q_embedding, self.rel_embedding), 1) + self.i_bias

        self.w_bias = tf.gather(self.node_features_w, self.rel_node)
        self.feat = tf.reduce_sum( tf.matmul(  tf.cast(x = tf.logical_and(x=self.q_features,y=self.rel_features), dtype=tf.float32) , self.node_features_w ) , 1)

        


        self.score += self.feat
        ##################l2
        
        # prediction loss
        self.pre_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) \
                        + config.lambda_dis * (tf.nn.l2_loss(self.rel_embedding) + tf.nn.l2_loss(self.q_embedding) + tf.nn.l2_loss(self.i_bias) + tf.nn.l2_loss(self.node_features_w) )

        
        ##############l1 True###################################################
        '''
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.8, scope=None)
        #weights = tf.trainable_variables() # all vars of your graph
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer,weights_list=[self.node_features_w])     

 

        # prediction loss
        self.pre_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) \
                        + config.lambda_dis * (tf.nn.l2_loss(self.rel_embedding) + tf.nn.l2_loss(self.q_embedding) + tf.nn.l2_loss(self.i_bias)) + config.lambda_dis_*regularization_penalty

        
        '''
        ########################################

        '''Ignore
        #l1 Regulazrization
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.8, scope=None)
        weights = tf.trainable_variables() # all vars of your graph
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        self.pre_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) + config.lambda_dis*regularization_penalty
        d_opt = tf.train.AdamOptimizer(config.lr_dis)
        '''
        d_opt = tf.train.AdamOptimizer(config.lr_dis)
        self.d_updates = d_opt.minimize(self.pre_loss)
        # self.reward = config.reward_factor * (tf.sigmoid(self.score) - 0.5)
        self.score = tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.log(1 + tf.exp(self.score))
