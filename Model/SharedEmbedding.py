import tensorflow as tf
import numpy as np
seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed
class Embedding(object):
    def __init__(self, DataSet, params):
        self.userNum = DataSet.userNum
        self.productNum = DataSet.productNum
        self.queryNum = DataSet.queryNum
        self.wordNum = DataSet.wordNum
        self.params = params
        self.max_query_len = DataSet.max_query_len
        
        const = tf.constant_initializer(0.0)
        self.query_linear_w = tf.compat.v1.get_variable('w',
            [params.embed_size, params.embed_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        self.query_linear_b = tf.compat.v1.get_variable('b', 
            [params.embed_size], 
            initializer=const)
        


        # User Embedding
        self.useremb = tf.compat.v1.get_variable(
            'userembedding',
            [self.userNum, params.embed_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        
        # User External Embedding
        self.user_output_emb = tf.compat.v1.get_variable(
            'useroutputembedding',
            [self.userNum, params.embed_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        # product embedding
        self.productemb = tf.compat.v1.get_variable(
            'productembedding',
            [self.productNum, params.embed_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        
        # Word embedding
        self.wordemb = tf.compat.v1.get_variable(
            'wordembedding',
            [self.wordNum, params.embed_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        
        
        
        
    def get_train_query_tanh_mean(self, query, query_len, query_len_mask):
            '''
            input size: (batch, maxQueryLen)
            对query处理使用函数
            tanh(W*(mean(Q))+b)
        
         '''
            #query = self.wordEmbedding_mean(query) 

            # short_term_query: size: ((batch, maxQueryLen))) ---> (batch, maxQueryLen, embedding)
            # long_term_query: (batch, long_term_size, maxQueryLen) ---> (batch, long_term_size, maxQueryLen, embedding)
            query = tf.nn.embedding_lookup(self.wordemb, query)
            
            
            # query len mask 使得padding的向量为0
            #len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(self.max_query_len-int(i.item())) for i in query_len]).unsqueeze(2).to(self.device)
            #query = query.mul(len_mask)
            
            #len_mask = tf.convert_to_tensor([[1.]*int(i.item())+[0.]*(self.max_query_len-int(i.item())) for i in query_len])
            #len_mask = tf.convert_to_tensor([[1.]*int(i)+[0.]*(self.max_query_len-int(i)) for i in query_len])
            
            # short_term_query_len_mask: [batch, maxQueryLen]
            # long_term_qurery_len_mask: [batch, long_term_size, maxQueryLen]
            query_len_mask_dim = len(query_len_mask.get_shape().as_list())
            if query_len_mask_dim == 2: # short_term
                expand_len_mask_matrix = tf.tile(tf.expand_dims(query_len_mask, axis=2),[1, 1, self.params.embed_size])
                query = tf.multiply(query, expand_len_mask_matrix)
                #query = query.sum(dim=1).div(query_len.unsqueeze(1).float())
                query = tf.divide(tf.reduce_sum(query, 1), tf.maximum(tf.cast(tf.expand_dims(query_len, axis=-1), tf.float32), 1))
                querylinear = tf.matmul(query, self.query_linear_w) + self.query_linear_b
            else: # long_term
                # expand_len_mask_matrix: [batch_size, long_term_size, maxQueryLen, embedding]
                expand_len_mask_matrix = tf.tile(tf.expand_dims(query_len_mask, axis=-1),[1, 1, 1, self.params.embed_size])
                query = tf.multiply(query, expand_len_mask_matrix)
                
                query = tf.divide(tf.reduce_sum(query, 2), tf.maximum(tf.cast(tf.expand_dims(query_len, axis=-1), tf.float32), 1))
                #query_dim = query.get_shape().as_list()
                query = tf.reshape(query, [-1, self.params.embed_size])
                querylinear = tf.matmul(query, self.query_linear_w) + self.query_linear_b
                querylinear = tf.reshape(querylinear, [-1, self.params.long_term_size, self.params.embed_size])
            #query = tf.divide(tf.reduce_sum(query, 1), tf.to_float(tf.expand_dims(query_len, axis=1)))
            
            
            #self.queryLinear = nn.Linear(self.embedding_dim, self.embedding_dim)
            #query = tf.contrib.layers.linear(query, self.params.embed_size, activation_fn=tf.nn.tanh)
            #querylinear = tf.matmul(query, self.query_linear_w) + self.query_linear_b

            query_tanh = tf.tanh(querylinear)
            
            return query_tanh
    
    # input : uid -> [batch_size, 1]
    # output : useremb -> [batch_size, params.embed_size]
    def GetUserEmbedding(self, uid):
        return tf.nn.embedding_lookup(self.useremb, uid)
    
    # if pid is before_pid_pos,Then: 
        # input : pid -> [batch_size, max_product_len]
        # output: productemb -> [batch_size, max_product_len, params.embed_size]
    def GetProductEmbedding(self, pid):
        return tf.nn.embedding_lookup(self.productemb, pid)
    
    # input qid-> [batch_size, max_query_len]
    # output queryemb -> [batch_size, params.embed_size]
    def GetQueryEmbedding(self, qid, query_len,query_len_mask):
        return self.get_train_query_tanh_mean(qid, query_len,query_len_mask)
    
    def GetAllProductEmbedding(self):
        return self.productemb
    
    def GetAllUserEmbedding(self):
        return self.useremb
      
        
        