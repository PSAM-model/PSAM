import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import numpy as np
from tensorflow.contrib import rnn
seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
def gradients(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[]):
    gradients = opt.compute_gradients(loss, vars)
    if max_gradient_norm is not None:
        to_clip = [(g, v) for g, v in gradients if v.name not in dont_clip]
        not_clipped = [(g, v) for g, v in gradients if v.name in dont_clip]
        gradients, variables = zip(*to_clip)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(
            gradients,
            max_gradient_norm
        )
        gradients = list(zip(clipped_gradients, variables)) + not_clipped

    # Add histograms for variables, gradients and gradient norms
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is None:
            print('warning: missing gradient: {}'.format(variable.name))
        if grad_values is not None:
            tf.compat.v1.summary.histogram(variable.name, variable)
            tf.compat.v1.summary.histogram(variable.name + '/gradients', grad_values)
            tf.compat.v1.summary.histogram(
                variable.name + '/gradient_norm',
                clip_ops.global_norm([grad_values])
            )

    return opt.apply_gradients(gradients, global_step=step)


def Loss_F(useremb, queryemb, product_pos, product_neg, k):
    # useremb -> [batch_size, emb_size]
    # product_neg -> [batch_size, k, emb_size] 
    u_plus_q = useremb + queryemb
    dis_pos = tf.multiply(tf.norm(u_plus_q - product_pos, ord=2, axis = 1), -1.0)
    log_u_plus_q_minus_i_pos = tf.math.log(tf.sigmoid(dis_pos))
    
    expand_u_plus_q = tf.tile(tf.expand_dims(u_plus_q, axis=1),[1,k,1])
    dis_neg = tf.norm(expand_u_plus_q - product_neg, ord=2, axis = 2)
    log_u_plus_q_minus_i_neg = tf.reduce_sum(tf.log(tf.sigmoid(dis_neg)),axis=1)
    batch_loss = -1 * (log_u_plus_q_minus_i_neg + log_u_plus_q_minus_i_pos)
    batch_loss = tf.reduce_mean(batch_loss)
    return batch_loss, tf.reduce_mean(-dis_pos), tf.reduce_mean(dis_neg)

def Pairwise_Loss_F(useremb, queryemb, product_pos, product_neg, k):
    # useremb -> [batch_size, emb_size]
    # product_neg -> [batch_size, k, emb_size] 
    u_plus_q = useremb + queryemb
    #dis_pos = tf.multiply(tf.norm(u_plus_q - product_pos, ord=2, axis=1), tf.cast(k, tf.float32))
    dis_pos = tf.norm(u_plus_q - product_pos, ord=2, axis=1)
    dis_pos = tf.reshape(dis_pos, [-1, 1])
    #dis_pos = tf.tile(dis_pos, [1, k])

    expand_u_plus_q = tf.tile(tf.expand_dims(u_plus_q, axis=1),[1, k, 1])
    dis_neg = tf.norm(expand_u_plus_q - product_neg, ord=2, axis = 2)
    
    batch_loss = tf.multiply(tf.reduce_sum(tf.log(tf.sigmoid(dis_neg - k * dis_pos) + 1e-6), axis=1), -1.0)
    batch_loss = tf.reduce_mean(batch_loss)
    return batch_loss, tf.reduce_mean(dis_pos), tf.reduce_mean(dis_neg)

def Inner_product(useremb, queryemb, product_pos, product_neg, k):
    # u_plus_q = user+query
    u_plus_q = useremb + queryemb
    
    #uq=u_plus_q.unsqueeze(2)  # skip
    #itp = item_pos.unsqueeze(1) # skip
    #pos_skip = torch.bmm(itp, uq) # skip
    #transpose_product_pos = tf.transpose(product_pos)

    #dis_pos = tf.matmul(u_plus_q, product_pos, transpose_b=True)
    dis_pos = tf.reduce_sum(tf.multiply(u_plus_q, product_pos), axis=1) 
    #dis_pos = -1.0 * tf.matmul(u_plus_q, product_pos, transpose_b=True)
    #dis_pos = tf.multiply(u_plus_q, product_pos)
    #loss_pos = pos_skip.sigmoid().log().mean()
    loss_pos = tf.reduce_mean(tf.math.log(tf.sigmoid(dis_pos)))
    #dis_pos = tf.reshape(dis_pos, [-1, 1])
    
    #itn = items_neg.unsqueeze(2) # skip
    #batch_size, neg_num, em_dim = items_neg.shape
        #neg_skip = torch.empty(batch_size, neg_num, 1)
    #for i in range(self.batch_size):
        #neg_skip[i] = torch.matmul(itn[i],uq[i]).squeeze(2)
    expand_u_plus_q = tf.tile(tf.expand_dims(u_plus_q, axis=1),[1,k,1])
    #transpose_product_neg = tf.transpose(product_neg,perm=[0, 2, 1])
    
    #dis_neg = tf.matmul(expand_u_plus_q, transpose_product_neg)
    dis_neg = tf.reduce_sum(tf.multiply(expand_u_plus_q, product_neg), axis=2) 
    #dis_neg = tf.multiply(expand_u_plus_q, product_neg)
    # loss_neg = neg_skip.mul(-1.0).sigmoid().log().sum(dim=1).mean()
    loss_neg = tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.sigmoid(tf.multiply(dis_neg, -1.0))),axis=1))
    #loss_neg = tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.sigmoid(dis_neg)),axis=1))
    #dis_neg = tf.reshape(dis_neg, [-1, 1])
    batch_loss = -1.0*(loss_pos + loss_neg)
    
    #return tf.reduce_mean(batch_loss), tf.reduce_mean(dis_pos), tf.reduce_mean(dis_neg)
    return batch_loss, tf.reduce_mean(dis_pos), tf.reduce_mean(dis_neg)
    
    #loss_neg = neg_skip.mul(-1.0).sigmoid().log().sum(dim=1).mean()
    #batch_loss = -1.0*(loss_pos + loss_neg)
    #return batch_loss, pos_skip.mean(), neg_skip.mean()

class PSAM(object):
    def __init__(self, Embed, params):
        self.UserID = tf.compat.v1.placeholder(tf.int32, shape=(None), name = 'uid')
        self.current_product_id = tf.compat.v1.placeholder(tf.int32, shape=(None), name='current_product_id')
        self.current_product_neg_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='current_product_neg_id')
        self.short_term_query_id = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='short_term_query_id')
        self.short_term_query_len = tf.compat.v1.placeholder(tf.int32, shape=(None), name='short_term_query_len')
        self.short_term_query_len_mask = tf.compat.v1.placeholder(tf.float32, shape=[None,None], name='short_term_query_len_mask')
        self.long_term_query_id = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None], name='long_term_query_id')
        self.long_term_query_len =  tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='long_term_query_len')
        self.long_term_query_len_mask =  tf.compat.v1.placeholder(tf.float32, shape=[None, None, None], name='long_term_query_len_mask')
        
        self.short_term_before_product_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='short_term_before_product_id')
        self.short_term_before_product_len = tf.compat.v1.placeholder(tf.int32, shape=(None), name='short_term_before_product_len')
        self.long_term_before_product_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='long_term_before_product_id')
        self.long_term_before_product_len = tf.compat.v1.placeholder(tf.int32, shape=(None), name='long_term_before_product_len')
        self.num_units = params.num_units
        self.long_term_size = params.long_term_size
        
        #batch_size = tf.shape(self.UserID)[0]
        self.productemb = Embed.GetAllProductEmbedding()
        # emb
        self.user_ID_emb = Embed.GetUserEmbedding(self.UserID)

        # current query emb
        self.queryemb = Embed.GetQueryEmbedding(self.short_term_query_id, self.short_term_query_len, self.short_term_query_len_mask)

        self.beforequeryemb = Embed.GetQueryEmbedding(self.long_term_query_id, self.long_term_query_len, self.long_term_query_len_mask)
        
        # positive emb
        self.product_pos_emb = Embed.GetProductEmbedding(self.current_product_id)
        self.product_neg_emb = Embed.GetProductEmbedding(self.current_product_neg_id)
        
        self.short_term_before_productemb = Embed.GetProductEmbedding(self.short_term_before_product_id)
        self.long_term_before_productemb = Embed.GetProductEmbedding(self.long_term_before_product_id)
        

        #self.long_short_rate = params.long_short_rate
        #self.long_short_rate = tf.expand_dims(tf.sigmoid(tf.cast(self.long_term_before_product_len - params.short_term_size, dtype=tf.float32)),-1)

        # # q*I
        # self.q_plus_before_productemb = tf.multiply(tf.tile(tf.expand_dims(self.queryemb, axis=1),[1,params.window_size,1]),self.before_productemb)


        # #Define User Emb
        # self.lstm_layer = rnn.BasicLSTMCell(self.num_units, forget_bias=1)
        # #self.outputs,_ = tf.nn.dynamic_rnn(self.lstm_layer, self.before_productemb, dtype="float32")
        # self.outputs,_ = tf.nn.dynamic_rnn(self.lstm_layer, self.q_plus_before_productemb, dtype="float32")
        # self.outputs = tf.transpose(self.outputs, [1, 0, 2])
        
        # # u = u*I (bad performance )
        # #self.useremb = tf.multiply(self.useremb, self.outputs[-1])

        # # u = FC(concat(u,I)) (good performance)
        # combine_user_item_emb =  tf.concat([self.useremb, self.outputs[-1]], 1)
        
        # Define Query-Based Attention LSTM for User's short-term inference

        self.short_term_lstm_layer = rnn.BasicLSTMCell(self.num_units, forget_bias=1)
        self.short_term_lstm_outputs,_ = tf.nn.dynamic_rnn(self.short_term_lstm_layer, self.short_term_before_productemb, dtype="float32")
        short_term_expand_q = tf.tile(tf.expand_dims(self.queryemb, axis=1),[1,params.short_term_size,1])
        self.short_term_attention = tf.nn.softmax(tf.multiply(self.short_term_lstm_outputs, short_term_expand_q))
        
        self.user_short_term_emb = tf.reduce_sum(tf.multiply(self.short_term_attention, self.short_term_lstm_outputs),axis=1)
        short_term_combine_user_item_emb =  tf.concat([self.user_ID_emb, self.user_short_term_emb], 1)
        
        self.short_term_combine_weights=tf.Variable(tf.random_normal([2 * params.embed_size, params.embed_size]))
        self.short_term_combine_bias=tf.Variable(tf.random_normal([params.embed_size]))
        
        
        self.short_term_useremb = tf.tanh(tf.matmul(short_term_combine_user_item_emb, self.short_term_combine_weights) + self.short_term_combine_bias)
        #self.useremb = tf.layers.dense(inputs=combine_user_item_emb, units=params.embed_size, activation=None)
        

        # # Define Memory Network for User's long-term inference
        # long_term_expand_user_emb = tf.tile(tf.expand_dims(self.user_ID_emb, axis=1),[1, self.long_term_size, 1])
        # long_term_expand_query_emb = tf.tile(tf.expand_dims(self.queryemb, axis=1),[1, self.long_term_size, 1])
        # long_term_scores = tf.multiply(long_term_expand_user_emb, long_term_expand_query_emb) +  tf.multiply(self.long_term_before_productemb, long_term_expand_query_emb)
        
        # long_term_memory_mask = tf.tile(tf.expand_dims(tf.cast(tf.not_equal(self.long_term_before_product_id,0), dtype=tf.float32),axis=-1),[1,1,params.embed_size])
        # self.long_term_scores = tf.reduce_sum(tf.multiply(long_term_scores, long_term_memory_mask),axis=-1)
        
        
        # self.long_term_attention = tf.nn.softmax(self.long_term_scores, name='LongTermAttention')

        # # [batch_Size, max_pro_length] => [batch_Size, 1, max_pro_length]
        # probs_temp = tf.expand_dims(self.long_term_attention, 1, name='TransformLongTermAttention')

        # #  output_memory: [batch size, max_pro_length, emb_size]
        # #  Transpose: [batch_Size, emb_size, max_pro_length]
        # c_temp = tf.transpose(self.beforequeryemb, [0, 2, 1], name='TransformOutputMemory')

        # # Apply a weighted scalar or attention to the external memory
        # # [batch size, 1, <max length>] * [batch size, embedding size, <max length>]
        # neighborhood = tf.multiply(c_temp, probs_temp, name='WeightedNeighborhood')
        # # Sum the weighted memories together
        # # Input:  [batch Size, embedding size, <max length>]
        # # Output: [Batch Size, Embedding Size]
        # # Weighted output vector
        # self.long_term_useremb = tf.reduce_sum(neighborhood, axis=2, name='OutputNeighborhood')

        # second
        
        # query-Attention
        # long_term_expand_query_emb:[batch_size, long_term_size, emb_size]
        long_term_expand_query_emb = tf.tile(tf.expand_dims(self.queryemb, axis=1),[1, self.long_term_size, 1])
        normalize_beforequeryemb = tf.nn.l2_normalize(self.beforequeryemb, axis=2)
        normalize_currentqueryemb = tf.nn.l2_normalize(long_term_expand_query_emb, axis=2)
        
        current_query_similarity = tf.reduce_sum(tf.multiply(normalize_beforequeryemb, normalize_currentqueryemb), axis=2)
        long_term_memory_mask = tf.cast(tf.not_equal(self.long_term_before_product_id,0), dtype=tf.float32)
        # current_query_similarity -> [batch_size, long_term_size]
        current_query_similarity = tf.multiply(current_query_similarity, long_term_memory_mask)

        # current_query_similarity -> [batch_size]
        l1_normalize_current_query_similarity = tf.reduce_sum(current_query_similarity, axis=1)
        expand_l1_normalize_current_query_similarity = tf.tile(tf.expand_dims(l1_normalize_current_query_similarity, axis=-1), [1, params.long_term_size])

        self.long_term_attention = tf.divide(current_query_similarity, expand_l1_normalize_current_query_similarity)
        expand_long_term_attention = tf.tile(tf.expand_dims(self.long_term_attention, axis=-1), [1, 1, params.embed_size])
        
        # long_term_query_product-> [batch_size, emb_size]
        self.long_term_query_product = tf.reduce_sum(tf.multiply(expand_long_term_attention, self.long_term_before_productemb), axis=1)
        
        long_term_combine_user_item_emb =  tf.concat([self.user_ID_emb, self.long_term_query_product], 1)
        
        self.long_term_combine_weights=tf.Variable(tf.random_normal([2 * params.embed_size, params.embed_size]))
        self.long_term_combine_bias=tf.Variable(tf.random_normal([params.embed_size]))
        self.long_term_useremb = tf.tanh(tf.matmul(long_term_combine_user_item_emb, self.long_term_combine_weights) + self.long_term_combine_bias)

        # combine_user_emb =  tf.concat([self.long_term_useremb, self.short_term_useremb], 1)
        # self.user_emb_combine_weights=tf.Variable(tf.random_normal([2 * params.embed_size, params.embed_size]))
        # self.user_emb_combine_bias=tf.Variable(tf.random_normal([params.embed_size]))
        # self.useremb = tf.tanh(tf.matmul(combine_user_emb, self.user_emb_combine_weights) + self.user_emb_combine_bias)

        # hyper parameter
        user_long_emb_weights= tf.Variable(tf.random_normal([params.embed_size, params.embed_size]))
        user_short_emb_weights = tf.Variable(tf.random_normal([params.embed_size, params.embed_size]))
        user_long_short_emb_bias = tf.Variable(tf.random_normal([params.embed_size]))
        self.long_short_rate = tf.sigmoid(tf.matmul(self.long_term_useremb, user_long_emb_weights) + tf.matmul(self.short_term_useremb, user_short_emb_weights) + user_long_short_emb_bias)

        if params.user_emb == "Complete":
            self.useremb = self.long_term_useremb * self.long_short_rate + (1 - self.long_short_rate) * self.short_term_useremb
        elif params.user_emb == "Short_term":
            self.useremb = self.short_term_useremb
        elif params.user_emb == "Long_term":
            self.useremb = self.long_term_useremb
        else:
            self.useremb = self.user_ID_emb

        # self.long_short_rate = tf.sigmoid(self.long_term_before_product_len - params.short_term_size)
        # 
        #self.useremb = self.short_term_useremb

        if params.loss_f == "Inner_product":
            self.opt_loss, self.pos_loss, self.neg_loss = Inner_product(self.useremb, self.queryemb, self.product_pos_emb, self.product_neg_emb, params.neg_sample_num)
        elif params.loss_f == "MetricLearning":
            self.opt_loss, self.pos_loss, self.neg_loss = Loss_F(self.useremb, self.queryemb, self.product_pos_emb, self.product_neg_emb, params.neg_sample_num)
        else:
            self.opt_loss, self.pos_loss, self.neg_loss = Pairwise_Loss_F(self.useremb, self.queryemb, self.product_pos_emb, self.product_neg_emb, params.neg_sample_num)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.opt_loss = self.opt_loss + sum(reg_losses)
        
        
        
        # Optimiser
        step = tf.Variable(0, trainable=False)
        
        self.opt = gradients(
            opt=tf.compat.v1.train.AdamOptimizer(learning_rate=params.learning_rate),
            loss=self.opt_loss,
            vars=tf.compat.v1.trainable_variables(),
            step=step
        )
    # (session, u_train, bpp_train, sl_train, lbpp_train, lsl_train, cpp_train, cni_train, qp_train, lp_train, bqp_train, blp_train, query_len_mask_train, long_query_len_mask_train)
    def step(self, session, uid, sbpid, sbpidlen, lbpid, lbpidlen, cpid, cpnid, qp, lp, bqp, blp,qlm, bqlm, testmode = False):
        input_feed = {}
        input_feed[self.UserID.name] = uid
        input_feed[self.short_term_before_product_id.name] = sbpid
        input_feed[self.short_term_before_product_len.name] = sbpidlen
        input_feed[self.long_term_before_product_id.name] = lbpid
        input_feed[self.long_term_before_product_len.name] =  lbpidlen
        input_feed[self.current_product_id.name] = cpid
        input_feed[self.current_product_neg_id.name] = cpnid
        input_feed[self.short_term_query_id.name] = qp
        input_feed[self.short_term_query_len.name] = lp
        input_feed[self.short_term_query_len_mask.name] = qlm
        input_feed[self.long_term_query_id.name] = bqp
        input_feed[self.long_term_query_len.name] = blp
        input_feed[self.long_term_query_len_mask.name] = bqlm
        
        if testmode == False:
            output_feed = [self.opt, self.opt_loss, self.pos_loss, self.neg_loss]
        else:
            #u_plus_q = self.useremb + self.queryemb
            u_plus_q = self.useremb + self.queryemb
            output_feed = [tf.shape(self.UserID)[0], u_plus_q, self.productemb]
        outputs = session.run(output_feed, input_feed)
        return outputs

    def summary(self, session, summarys, uid, cpid, cpnid, qp, lp, qlm, HR, hr, MRR, mrr, NDCG, ndcg,AVG_LOSS,avg_loss):
        input_feed = {}
        input_feed[self.UserID.name] = uid
        input_feed[self.current_product_id.name] = cpid
        input_feed[self.current_product_neg_id.name] = cpnid
        input_feed[self.query_id.name] = qp
        input_feed[self.query_len.name] = lp
        input_feed[self.query_len_mask.name] = qlm
        input_feed[HR.name] = hr
        input_feed[MRR.name] = mrr
        input_feed[NDCG.name] = ndcg
        input_feed[AVG_LOSS.name] = avg_loss
        output_feed = [summarys]
        outputs = session.run(output_feed, input_feed)
        return outputs