"""the generator class
use the model to initialize and add some other structures different from the discriminator
"""
import tensorflow as tf
import config


class Generator(object):
    def __init__(self, n_node, node_emd_init, node_features):#provided in the start
        self.n_node = n_node
        self.node_emd_init = node_emd_init
        self.node_features = node_features


        with tf.variable_scope('Generator'):
            self.node_embed =  tf.get_variable(name="node_embed", shape=self.node_emd_init.shape,
                                               initializer=tf.constant_initializer(self.node_emd_init), trainable=True)
            self.node_b = tf.Variable(tf.zeros([self.n_node]))

            self.node_features = tf.get_variable(name="node_features", shape=self.node_features.shape,
                                              initializer=tf.constant_initializer(self.node_features), dtype=tf.bool, trainable=False)

            self.node_features_w = tf.Variable( initial_value=tf.random_normal([config.n_features,1]),trainable=True )



        self.node_features_w *= self.node_features_w

        self.all_score = tf.matmul(self.node_embed, self.node_embed, transpose_b=True) + self.node_b 
        # placeholder
        self.q_node = tf.placeholder(tf.int32, shape=[None])#during runtime provided
        self.rel_node = tf.placeholder(tf.int32, shape=[None])
        self.reward = tf.placeholder(tf.float32, shape=[None])

        self.q_features = tf.nn.embedding_lookup(self.node_features, self.q_node)
        self.rel_features = tf.nn.embedding_lookup(self.node_features, self.rel_node)

        self.w_bias = tf.gather(self.node_features_w, self.rel_node)
        self.feat = tf.reduce_sum( tf.matmul(  tf.cast(x = tf.logical_and(x=self.q_features,y=self.rel_features), dtype=tf.float32) , self.node_features_w ) , 1)



        # look up embeddings
        self.q_embedding= tf.nn.embedding_lookup(self.node_embed, self.q_node)  # batch_size*n_embed
        self.rel_embedding = tf.nn.embedding_lookup(self.node_embed, self.rel_node)  # batch_size*n_embed
        self.i_bias = tf.gather(self.node_b, self.rel_node)
        score = tf.reduce_sum(self.q_embedding*self.rel_embedding, axis=1) + self.i_bias
        


        #for sampling this is used
        self.feat_custom = tf.cast(x = self.node_features , dtype=tf.float32) * tf.transpose( self.node_features_w )
        self.feat_custom = tf.matmul(self.feat_custom, self.feat_custom, transpose_b=True) 
        self.all_score  += self.feat_custom


        score += self.feat
        


        ###############l2########################

        i_prob = tf.nn.sigmoid(score)
        # clip value
        self.i_prob = tf.clip_by_value(i_prob, 1e-5, 1)
        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) \
                        + config.lambda_gen* (tf.nn.l2_loss(self.rel_embedding) + tf.nn.l2_loss(self.q_embedding) + tf.nn.l2_loss(self.i_bias) + tf.nn.l2_loss(self.node_features_w) )
        g_opt = tf.train.AdamOptimizer(config.lr_gen)
        self.gan_updates = g_opt.minimize(self.gan_loss)
        

        
        ###############l1########################
        '''
        #l1 regularization on weights of feature only
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.8, scope=None)
        #weights = tf.trainable_variables() # all vars of your graph
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer,weights_list=[self.node_features_w])



     
        i_prob = tf.nn.sigmoid(score)
        # clip value
        self.i_prob = tf.clip_by_value(i_prob, 1e-5, 1)#reduce_mean gives expected value
        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) \
                        + config.lambda_gen* (tf.nn.l2_loss(self.rel_embedding) + tf.nn.l2_loss(self.q_embedding) + tf.nn.l2_loss(self.i_bias)) + config.lambda_gen_*regularization_penalty
        g_opt = tf.train.AdamOptimizer(config.lr_gen)
        self.gan_updates = g_opt.minimize(self.gan_loss)
        '''


 # NxM (4039x1293)
 # W   (1293x1)



 # for each N:
 #    1x193
 #    1x193 

 # NxM . MxN

 # NxN (4039x4039)
