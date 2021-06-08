import gzip,glob,os
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
seed = 42
np.random.seed(seed)


class DataSet(object):
    def __init__(self, UserData, metaData, neg_sample_num, max_query_len, max_review_len, max_product_len, savepath, short_term_size, long_term_size = None, weights=True):
        
        
        # 用户,商品与词统一成v,v与id之间的转换
        self.id_2_v = dict()
        self.v_2_id = dict()
        
        # 用户与id之间的转换
        self.id_2_user = dict()
        self.user_2_id = dict()
        
        # 商品与id之间的转换
        self.id_2_product = dict()
        self.product_2_id = dict()
        
        self.product_2_query = dict()
        
        # query的词都用id表示
        self.word_2_id = dict()
        self.id_2_word = dict()
        
        self.userReviews = dict()
        self.userReviewsCount = dict()
        self.userReviewsCounter = dict()
        self.userReviewsTest = dict()
        
        # 保存之前的信息
        self.before_pid_pos = dict()
        self.before_textlist = dict()
        self.before_textlenlist = dict()
        self.before_querylist_pos = dict()
        
        self.nes_weight = []
        self.word_weight = []
        
        self.max_query_len = int(max_query_len)
        self.neg_sample_num = int(neg_sample_num)
        self.max_review_len = int(max_review_len)
        self.max_product_len = int(max_product_len)
        self.short_term_size = int(short_term_size)

        
        self.init_dict(UserData, metaData)
        if long_term_size == None:
            self.long_term_size = int(self.average_product_len)
        else:
            self.long_term_size = int(long_term_size)
        
        self.train_data = []
        self.test_data = []
        self.eval_data = []

        short_term_size
        self.init_dataset(UserData, self.short_term_size, self.long_term_size)
        
        
        self.init_sample_table()
        self.WriteToFile(savepath)
        #self.train_dataset = tf.data.Dataset.from_generator(self.next_train_batch, (tf.int32,tf.int32, tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32))

    
    
    def trans_to_ids(self, query, max_len, weight_cal = True):
        query = query.split(' ')
        qids = []
        for w in query:
            if w == '':
                continue
            qids.append(self.word_2_id[w])
            # 统计词频
            if weight_cal:
                self.word_weight[self.word_2_id[w]-1] += 1
        for _ in range(len(qids), max_len):
            qids.append(self.word_2_id['<pad>'])
        return qids, len(query)
    
    def init_dict(self, UserData, metaData):
        ProductSet = set()
        words = set()
        vid = 0
        self.v_2_id['<pad>'] = vid
        self.id_2_v[vid] = '<pad>'
        vid += 1
        uid = 0
        for i in range(len(UserData)):
            # 每个用户都唯一而且购买长度大于等于10
            self.id_2_user[uid] = UserData[i].UserID
            self.user_2_id[UserData[i].UserID] = uid
            
            self.id_2_v[vid] = UserData[i].UserID
            self.v_2_id[UserData[i].UserID] = vid
            
            #更新产品集合
            asins = []
            reviewtexts = []
            for j in range(len(UserData[i].UserPurchaseList)):
                asins.append(UserData[i].UserPurchaseList[j]['asin'])
                reviewtexts.append(UserData[i].UserPurchaseList[j]['reviewText'])
            
            # 每个用户的购买记录
            ProductSet.update(asins)
            self.userReviews[uid] = asins
            words.update(set(' '.join(reviewtexts).split()))
            uid += 1
            vid += 1
        self.userNum = uid
        
        pid = 0
        self.product_2_id['<pad>'] = pid
        self.id_2_product[pid] = '<pad>'
        #self.v_2_id['<productpad>'] = vid
        #self.id_2_v[vid] = '<productpad>'
        vid += 1
        pid += 1
        for p in ProductSet:
            try:
                '''
                判断这个product是否有query
                '''
                if (len(metaData.loc[p]['query']) > 0):
                    self.product_2_query[p] = metaData.loc[p]['query']
                    words.update(' '.join(metaData.loc[p]['query']).split(' '))
            except:
                pass
            
            self.id_2_product[pid] = p
            self.product_2_id[p] = pid
            self.id_2_v[vid] = p
            self.v_2_id[p] = vid
            pid += 1
            vid += 1
        
        self.productNum = pid
        self.nes_weight = np.zeros(self.productNum)
        self.queryNum = len(self.product_2_query)
        
        wi = 0
        self.word_2_id['<pad>'] = wi
        self.id_2_word[wi] = '<pad>'
        #self.v_2_id['<wordpad>'] = vid
        #self.id_2_v[vid] = '<wordpad>'
        wi += 1
        vid += 1
        for w in words:
            if (w==''):
                continue
            self.word_2_id[w] = wi
            self.id_2_word[wi] = w
            self.v_2_id[w] = vid
            self.id_2_v[vid] = w
            wi += 1
            vid += 1
        self.wordNum = wi
        self.word_weight = np.zeros(wi)

        # 统计用户平均购买记录
        User_product_len_list = []
        for U in range(len(UserData)):
            uid = self.user_2_id[UserData[U].UserID]
            UserItemLen = len(UserData[U].UserPurchaseList)
            User_product_len_list.append(UserItemLen)
        self.average_product_len = np.mean(User_product_len_list)

    
    def Tran_Uid_2_vid(self, uid_list):
        v_list = []
        for i in range(len(uid_list)):
            v_list.append(self.v_2_id[self.id_2_user[uid_list[i]]])
        return v_list
    
    def Tran_Pid_2_vid(self, pid_list):
        v_list = []
        for i in range(len(pid_list)):
            v_list.append(self.v_2_id[self.id_2_product[pid_list[i]]])
        return v_list
    
    def Tran_Wid_2_vid(self, wi_list):
        v_list = []
        for i in range(len(wi_list)):
            v_list.append(self.v_2_id[self.id_2_word[wi_list[i]]])
        return v_list
    
    
    def init_dataset(self, UserData, short_term_size, long_term_size, weights=True):
        try:
            self.data_X = []
            for U in range(len(UserData)):
                uid = self.user_2_id[UserData[U].UserID]
                # 每个User的数据，方便以后划分数据集
                User_Data_X = []
                
                UserItemLen = len(UserData[U].UserPurchaseList)
                
                # review
                before_review_text_list = []
                before_review_text_len_list = []
                
                # purchase list
                before_pids_pos = []
                
                # the query of purchase list
                before_querylist_pos = []
                before_querylist_pos_len = []
                
                # 向前补充short_term_size-1个信息
                for k in range(0, short_term_size - 1):
                    before_pids_pos.append(self.product_2_id['<pad>'])
                    padding_review_text = ""
                    before_text_ids, before_text_len = self.trans_to_ids(padding_review_text, self.max_review_len)
                    before_review_text_list.append(before_text_ids)
                    before_review_text_len_list.append(before_text_len)
                
                
                for l in range(1, UserItemLen):
                    # v(i-short_term_size), v(i - windows_size +1),...,v(i-1)  
                    before_pid_pos = self.product_2_id[UserData[U].UserPurchaseList[l-1]['asin']]
                    before_pids_pos.append(before_pid_pos)
                    before_review_text = UserData[U].UserPurchaseList[l-1]['reviewText']
                    try:
                        before_text_ids, before_text_len = self.trans_to_ids(before_review_text, self.max_review_len)
                    except:
                        before_text_len = 0
                        before_text_ids = []
                        for i in range(self.max_review_len):
                            before_text_ids.append(self.word_2_id['<pad>'])
                    before_review_text_list.append(before_text_ids)
                    before_review_text_len_list.append(before_text_len)
                    try:
                        before_query_text_array_pos = self.product_2_query[self.id_2_product[before_pid_pos]]
                        before_Qids_pos, before_Len_pos = self.trans_to_ids(Q_text_array_pos[0], self.max_query_len)
                        before_querylist_pos.append(before_Qids_pos)
                        before_querylist_pos_len.append(before_Len_pos)
                    except:
                        print("User:%s Item:%s has not query!\n" % (str(UserData[U].UserID), str(UserData[U].UserPurchaseList[l-1]['asin'])))
                        null_query_list = []
                        for i in range(0, self.max_query_len):
                            null_query_list.append(self.word_2_id['<pad>'])
                        before_querylist_pos.append(null_query_list)
                        before_querylist_pos_len.append(0)
                    # vi
                    current_pid_pos = self.product_2_id[UserData[U].UserPurchaseList[l]['asin']]
                    cur_before_pids_pos = before_pids_pos[-short_term_size:]

                    # generate long-term sequence
                    if len(before_pids_pos) < long_term_size:
                        cur_long_before_pids_pos = [self.product_2_id['<pad>']] * (long_term_size - len(before_pids_pos)) + before_pids_pos
                        cur_long_before_pids_pos_len = len(before_pids_pos) - (short_term_size - 1)
                    else:
                        cur_long_before_pids_pos = before_pids_pos[-long_term_size:]
                        cur_long_before_pids_pos_len = len(cur_long_before_pids_pos)
                    

                    if l < short_term_size:
                        cur_before_pids_pos_len = l
                    else:
                        cur_before_pids_pos_len = short_term_size
                    self.before_pid_pos[str(uid) + "_" + str(current_pid_pos)] = cur_before_pids_pos
                    self.before_textlist[str(uid) + "_" + str(current_pid_pos)] = before_review_text_list[-short_term_size:]
                    self.before_textlenlist[str(uid) + "_" + str(current_pid_pos)] = before_review_text_len_list[-short_term_size:]
                    self.before_querylist_pos[str(uid) + "_" + str(current_pid_pos)] = before_querylist_pos[-short_term_size:]
                    self.nes_weight[current_pid_pos] += 1
                    current_text =  UserData[U].UserPurchaseList[l]['reviewText']
                    try:
                        current_text_ids, current_text_Len = self.trans_to_ids(current_text, self.max_review_len)
                    except:
                        current_text_Len = 0
                        current_text_ids = []
                        for i in range(self.max_review_len):
                            current_text_ids.append(self.word_2_id['<pad>'])
                    try:
                        Q_text_array_pos = self.product_2_query[self.id_2_product[current_pid_pos]]
                    except:
                        # vi物品没有query，不加入数据集
                        Q_text_array_pos = []
                    for Qi in range(len(Q_text_array_pos)):
                        try:
                            Qids_pos, Len_pos = self.trans_to_ids(Q_text_array_pos[Qi], self.max_query_len)
                        except:
                            break
                        product_len_mask = [0.] * (short_term_size - cur_before_pids_pos_len) + [1.] * cur_before_pids_pos_len
                        query_len_mask = [1.] * Len_pos + [0.] * (self.max_query_len - Len_pos)


                        null_query_list = [self.word_2_id['<pad>']] * self.max_query_len
                        if len(before_querylist_pos) < self.long_term_size:
                            cur_long_before_query_pos =  [null_query_list] * (self.long_term_size - len(before_querylist_pos)) + before_querylist_pos
                            null_query_mask = [0.] * self.max_query_len
                            cur_long_before_query_len_mask = [null_query_mask] * (self.long_term_size - len(before_querylist_pos_len)) + [[1.] * i + [0.] * (self.max_query_len - i) for i in before_querylist_pos_len]
                            cur_long_before_query_pos_len = [0] * (self.long_term_size - len(before_querylist_pos_len)) + before_querylist_pos_len
                        else:
                            cur_long_before_query_pos = before_querylist_pos[-self.long_term_size:]
                            cur_long_before_query_len_mask = [[1.] * i + [0.] * (self.max_query_len - i) for i in before_querylist_pos_len[-self.long_term_size:]]
                            cur_long_before_query_pos_len = before_querylist_pos_len[-self.long_term_size:]
                        User_Data_X.append((uid, cur_before_pids_pos, cur_before_pids_pos_len, current_pid_pos, cur_long_before_pids_pos, cur_long_before_pids_pos_len, \
                                            Qids_pos, Len_pos, cur_long_before_query_pos, cur_long_before_query_pos_len,\
                                            current_text_ids, current_text_Len, product_len_mask, query_len_mask,cur_long_before_query_len_mask))
                        #User_Data_X.append((self.Tran_Uid_2_vid([uid]), self.Tran_Pid_2_vid(cur_before_pids_pos), self.Tran_Pid_2_vid([current_pid_pos]), self.Tran_Wid_2_vid(Qids_pos),
                                            #Len_pos, self.Tran_Wid_2_vid(current_text_ids), current_text_Len))
                        try:
                            self.userReviewsCount[uid] += 1
                            self.userReviewsCounter[uid] += 1
                        except:
                            self.userReviewsCount[uid] = 1
                            self.userReviewsCounter[uid] = 1    
                self.data_X.append(User_Data_X)
            
            '''
            数据集划分-根据用户进行划分，划分比例为7:2:1
            '''
            for u in self.data_X:
                u_len = len(u)
                u_train_data = u[:int(0.7*u_len)]
                u_validation_data = u[int(0.7*u_len):int(0.9*u_len)]
                u_test_data = u[int(0.9*u_len):]
                self.train_data.extend(u_train_data)
                self.test_data.extend(u_test_data)
                self.eval_data.extend(u_validation_data)
                
                
            if weights is not False:
                wf = np.power(self.nes_weight, 0.75)
                wf = wf / wf.sum()
                self.weights = wf
                wf = np.power(self.word_weight, 0.75)
                wf = wf / wf.sum()
                self.word_weight = wf
        
        except Exception as e:
            s=sys.exc_info()
            print("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
            with open (r'out.txt','a+') as ff:
                ff.write(str(e)+ '\n')
    
                    
    def neg_sample(self):
        neg_item = []
        neg_word = []
        for ii in range(self.neg_sample_num):
            neg_item.append(self.sample_table_item[np.random.randint(self.table_len_item)])
            neg_word.append(self.sample_table_word[np.random.randint(self.table_len_word)])
        #return self.Tran_Pid_2_vid(neg_item), self.Tran_Wid_2_vid(neg_word)
        return neg_item,neg_word
    
    def init_sample_table(self):
        table_size = 1e6
        count = np.round(self.weights*table_size)
        self.sample_table_item = []
        for idx, x in enumerate(count):
            self.sample_table_item += [idx]*int(x)
        self.table_len_item = len(self.sample_table_item)
        
        count = np.round(self.word_weight*table_size)
        self.sample_table_word = []
        for idx, x in enumerate(count):
            self.sample_table_word += [idx]*int(x)
        self.table_len_word = len(self.sample_table_word)
    
    def next_train_batch(self):
        #uid, before_pids_pos, before_querylist_pos, before_textlist, before_textlenlist, current_pid_pos, Qids_pos, Len_pos, current_text_ids, current_text_Len
        self.train_data = np.array(self.train_data)
        np.random.shuffle(self.train_data)
        train_data_len = len(self.train_data)
        uids = [self.train_data[i,0] for i in range(train_data_len)]
        before_pids_pos = [self.train_data[i,1] for i in range(train_data_len)]
        before_pids_pos_len = [self.train_data[i,2] for i in range(train_data_len)]
        current_pid_pos = [self.train_data[i,3] for i in range(train_data_len)]
        Qids_pos = [self.train_data[i,4] for i in range(train_data_len)]
        Len_pos = [self.train_data[i,5] for i in range(train_data_len)]
        current_text_ids = [self.train_data[i,6] for i in range(train_data_len)]
        current_text_Len =  [self.train_data[i,7] for i in range(train_data_len)]
        for i in range(train_data_len):
            current_neg_item, current_neg_word = self.neg_sample()
            yield np.array(uids[i]), np.array(before_pids_pos[i]),np.array(before_pids_pos_len[i]), np.array(current_pid_pos[i]),\
                            np.array(Qids_pos[i]), np.array(Len_pos[i]), np.array(current_text_ids[i]),\
                            np.array(current_text_Len[i]), np.array(current_neg_item), np.array(current_neg_word)
    
    def next_validation_batch(self):
        self.eval_data = np.array(self.eval_data)
        np.random.shuffle(self.eval_data)
        eval_data_len = len(self.eval_data)
        uids = [self.eval_data[i,0] for i in range(eval_data_len)]
        before_pids_pos = [self.eval_data[i,1] for i in range(eval_data_len)]
        before_pids_pos_len = [self.eval_data[i,2] for i in range(eval_data_len)]
        current_pid_pos = [self.eval_data[i,3] for i in range(eval_data_len)]
        Qids_pos = [self.eval_data[i,4] for i in range(eval_data_len)]
        Len_pos = [self.eval_data[i,5] for i in range(eval_data_len)]
        current_text_ids = [self.eval_data[i,6] for i in range(eval_data_len)]
        current_text_Len =  [self.eval_data[i,7] for i in range(eval_data_len)]
        for i in range(eval_data_len):
            current_neg_item, current_neg_word = self.neg_sample()
            yield np.array(uids[i]), np.array(before_pids_pos[i]),np.array(before_pids_pos_len[i]), np.array(current_pid_pos[i]),\
                            np.array(Qids_pos[i]), np.array(Len_pos[i]), np.array(current_text_ids[i]),\
                            np.array(current_text_Len[i]), np.array(current_neg_item), np.array(current_neg_word)
        
        
        
    def getTestItem(self):
        return self.test_data
    
    def getevaldata(self):
        return self.eval_data
    
    def getbeforeInfo(self, uid, current_pid_pos):
        #before_pids_pos = self.before_pid_pos[str(uid) + "_" + str(current_pid_pos)]
        before_textlist = self.before_textlist[str(uid) + "_" + str(current_pid_pos)]
        before_textlenlist = self.before_textlenlist[str(uid) + "_" + str(current_pid_pos)]
        before_querylist_pos = self.before_querylist_pos[str(uid) + "_" + str(current_pid_pos)]
        return before_textlist, before_textlenlist, before_querylist_pos
    
    def WriteToFile(self, savepath):
        # User_Data_X.append((uid, before_pids_pos, current_pid_pos, Qids_pos, Len_pos, current_text_ids, current_text_Len))
        with open(savepath + "DataSet.txt", "w+") as f:
            for u in self.data_X:
                for i in range(len(u)):
                    eachdata = ""
                    for j in range(len(u[i])):
                        eachdata = eachdata + str(u[i][j]) + " "
                    eachdata += "\n\n"
                    f.write(eachdata)
