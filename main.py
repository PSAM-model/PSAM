import Data_loader.Data_Generator as data
import Model.SharedEmbedding as SE
import Model.PSAM as PSAM
import os,json,argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import evaluate as me
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

def train(psammodel, Embed, Dataset, params):
    filename = "./Performance_PSAM_Model_Windowsize_" + str(params.window_size) + "_Epoch_" +  str(params.epoch) + "_lr_" + str(params.learning_rate) + "_embsize_" + str(params.embed_size) + "_numunit_" + str(params.num_units) + ".txt"
    model_dir = params.model + "Windowsize_" + str(params.window_size) + "_Epoch_" +  str(params.epoch) + "_embsize_" + str(params.embed_size) + "_numunit_" + str(params.num_units) +"/"
    #pig_dir = params.pig + "Windowsize_" + str(params.window_size) + "_Epoch_" +  str(params.epoch) + "_embsize_" + str(params.embed_size)  + "_numunit_" + str(params.num_units) +"/"
    
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    #if not os.path.isdir(pig_dir):
        #os.mkdir(pig_dir)    
    log_dir = os.path.join(model_dir, 'logs')
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
    )) as session:

        # train dataset
        train_dataset = np.array(Dataset.train_data)
        np.random.shuffle(train_dataset)
        
        # eval dataset
        eval_dataset = np.array(Dataset.eval_data)
        np.random.shuffle(eval_dataset)

        # test dataset
        test_dataset = np.array(Dataset.test_data)
        
        
        train_dataset = np.concatenate((train_dataset,eval_dataset),axis=0)
        productnum = Dataset.productNum

        usernum = Dataset.userNum

        querynum = Dataset.queryNum

        print("Users:", usernum)
        print("Product:", productnum)
        print("Query:", querynum)
        
        avg_loss = tf.compat.v1.placeholder(tf.float32, [], 'loss')
        tf.compat.v1.summary.scalar('loss', avg_loss)

        validation_HR = tf.compat.v1.placeholder(tf.float32, [], 'validation_HR')
        tf.compat.v1.summary.scalar('validation_HR', validation_HR)
        
        validation_MRR = tf.compat.v1.placeholder(tf.float32, [], 'validation_MRR')
        tf.compat.v1.summary.scalar('validation_MRR', validation_MRR)
        
        validation_NDCG = tf.compat.v1.placeholder(tf.float32, [], 'validation_NDCG')
        tf.compat.v1.summary.scalar('validation_NDCG', validation_NDCG)
        

        summary_writer = tf.compat.v1.summary.FileWriter(log_dir, session.graph)
        summaries = tf.compat.v1.summary.merge_all()
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        train_saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

        session.run(tf.compat.v1.local_variables_initializer())
        session.run(tf.compat.v1.global_variables_initializer())
        
        # define loss
        losses = []
        pos_losses = []
        neg_losses = []

        # define performance indicator
        HR_list = []
        MRR_list = []
        NDCG_list = []

        total_batch = int(len(train_dataset) / params.batch_size) + 1
        step = 0
        min_loss = 10000.
        best_val = 10000. 
        best_HR = 0.
        best_MRR = 0.
        best_NDCG = 0.

        for e in range(params.epoch):
            train_startpos = 0
            eval_startpos = 0
            for b in range(total_batch):
                
                u_train,bpp_train,sl_train,cpp_train, lbpp_train, lsl_train, qp_train,lp_train, bqp_train, blp_train,cti_train,ctl_train,cni_train,cnw_train, product_len_mask_train, query_len_mask_train, long_query_len_mask_train = data.Get_next_batch(Dataset, train_dataset, train_startpos, params.batch_size)
                _, train_loss, train_pos_loss, train_neg_loss = psammodel.step(session, u_train, bpp_train, sl_train, lbpp_train, lsl_train, cpp_train, cni_train, qp_train, lp_train, bqp_train, blp_train, query_len_mask_train, long_query_len_mask_train)

                
                losses.append(train_loss)
                pos_losses.append(train_pos_loss)
                neg_losses.append(train_neg_loss)
                train_startpos += params.batch_size
                step += 1
                #print(train_startpos)
                #if step % params.log_every == 0 and step != 0:
                    #print("step: {}|  epoch: {}|  batch: {}|  train_loss: {:.6f}| pos_loss: {:.6f} | neg_loss: {:.6f}".format(step, e, b, train_loss, train_pos_loss, train_neg_loss))
                
            print("step: {}|  epoch: {}|  batch: {}|  train_loss: {:.6f}| pos_loss: {:.6f} | neg_loss: {:.6f}".format(step, e, b, train_loss, train_pos_loss, train_neg_loss))
            
            # After every epoch
            #test performance
            test_dataset = np.array(Dataset.test_data)
            u_test,bpp_test,sl_test,cpp_test, lbpp_test, lsl_test, qp_test,lp_test, bqp_test, blp_test, cti_test,ctl_test,cni_test,cnw_test,product_len_mask_test,query_len_mask_test,long_query_len_mask_test = data.Get_next_batch(Dataset, test_dataset, 0, len(test_dataset))
            #print("Start Performance Test of point model!\n")
            bs, itememb, allproductemb = psammodel.step(session, u_test, bpp_test, sl_test, lbpp_test, lsl_test, cpp_test, cni_test, qp_test, lp_test, bqp_test, blp_test, query_len_mask_test, long_query_len_mask_test,True)
            all_hr = 0
            all_mrr = 0
            all_ndcg = 0
            for i in range(len(itememb)):
                per_hr,per_mrr,per_ndcg = me.GetItemRankList(allproductemb, itememb[i], cpp_test[i], params.depth)
                perpos, minposvalue, currentposvalue, maxposvalue = me.FindTheRank(allproductemb, itememb[i], cpp_test[i])
                all_hr += per_hr
                all_mrr+=per_mrr
                all_ndcg+=per_ndcg
            hr = all_hr / float(len(itememb))
            mrr = all_mrr / float(len(itememb))
            ndcg = all_ndcg / float(len(itememb))
            Performance_info = "Performance: EPOCH:{}|HR: {:.6f}| MRR: {:.6f}| NDCG: {:.6f}".format(e, hr, mrr, ndcg)
            print(Performance_info)
            # update the best performance 
            HR_list.append(hr)
            MRR_list.append(mrr)
            NDCG_list.append(ndcg)
            can_save = False
            if best_HR < hr:
                best_HR = hr
                can_save = True
            if best_MRR < mrr:
                best_MRR = mrr
                can_save = True
            if best_NDCG < ndcg:
                best_NDCG = ndcg
                can_save = True
            if can_save == True:
                saver.save(session, model_dir, global_step=step)
        with open(filename, 'a+') as f:
            bestinfo = "Best Performance: HR: {:.6f}| MRR: {:.6f}| NDCG: {:.6f}".format(best_HR, best_MRR, best_NDCG)
            f.write(bestinfo+'\n')
        
    
def main(args):
    if not os.path.isdir(args.model):
        os.mkdir(args.model)

    with open(os.path.join(args.model, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    Dataset = data.Generate_Data(args)
    Embed = SE.Embedding(Dataset, args)
    psammodel = PSAM.PSAM(Embed, args)
    train(psammodel, Embed, Dataset, args)
    
    

    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='path to model output directory')
    parser.add_argument('--pig', type=str, default="./picture/",
                        help='path to picture output directory')
      
    parser.add_argument('--ReviewDataPath', type=str, required=True,
    					help='path to the input review dataset')
    parser.add_argument('--MetaDataPath', type=str, required=True,
    					help='path to the input meta dataset')
                        
    
    # set default is fine.
    parser.add_argument('--ReviewDataSavepath', type=str, default='./Data_loader/AfterPreprocessData/ReviewBin/',
    					help='path to the save the bin of review dataset')
    parser.add_argument('--Review_UserDataSavepath', type=str, default='./Data_loader/AfterPreprocessData/ReviewUserBin/',
    					help='path to the save the bin of user Review dataset')
    parser.add_argument('--Review_CombineUserDataSavepath', type=str, default='./Data_loader/AfterPreprocessData/ReviewUserCombineBin/',
    					help='path to the save the bin of Review dataset combined by user')
    parser.add_argument('--MetaDataSavepath', type=str, default='./Data_loader/AfterPreprocessData/MetaBin/',
    					help='path to the save the bin of Meta dataset')
    parser.add_argument('--Meta_CombineDataSavepath', type=str, default='./Data_loader/AfterPreprocessData/MetaCombineBin/',
    					help='path to the save the combine bin of Meta dataset')
    parser.add_argument('--ProcessInfoPath', type=str, default='./Data_loader/InfoData/',
    					help='path to the save the Info of process data')
    parser.add_argument('--DataSetSavePath', type=str, default='./Data_loader/AfterPreprocessData/DataSet/',
    					help='path to the save the Info of process data')
    

    parser.add_argument('--user-emb', type=str, default="Complete",
                        help='Select type of user embedding: Complete, Short_term and Long_term')
    parser.add_argument('--loss-f', type=str, default="MetricLearning",
                        help='Select type of loss function: MetricLearning and Inner_product')
    parser.add_argument('--window-size', type=int, default=5,
                        help='ProNADE Input Data Window size(0:Use all product bought before)')
    parser.add_argument('--embed-size', type=int, default=100,
                        help='size of the hidden layer of User, product and query')
    parser.add_argument('--short-term-size', type=int, default=5,
                        help='size of User short-term preference')
    parser.add_argument('--long-term-size', type=int, default=15,
                        help='size of User lhort-term preference')                
    parser.add_argument('--activation', type=str, default='sigmoid',
                        help='which activation to use: sigmoid|tanh')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=100,
                        help='train data epoch')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--validation-batch-size', type=int, default=64,
                        help='the batch size of validation')              
    parser.add_argument('--neg-sample-num', type=int, default=5,
                        help='the number of negative sample')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--save-every', type=int, default=500,
                        help='print loss after this many steps')
    parser.add_argument('--depth', type=int, default=20,
                        help='the depth of test performance to use')
    parser.add_argument('--modelsel', type=int, default=1,
                        help='model selection')

    # LSTM-Parameter
    parser.add_argument('--num-units', type=int, default=100,
                        help='the number of hidden unit in lstm model')                 
    
    # Transformer-Parameter
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())