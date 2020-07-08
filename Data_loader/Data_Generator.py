import os,pickle,errno
from Data_loader.User import UserData
import Data_loader.Data_Util as Data_Util
import Data_loader.ReviewDataProcess as ReviewDataProcess
import Data_loader.CombineReviewData as CombineReviewData
import Data_loader.MetaDataProcess as MetaDataProcess
import Data_loader.CombineMetaData as CombineMetaData
import Data_loader.Data as Data
import tensorflow as tf
import numpy as np


def PreProcessData(params, CategorySet):
	print("Start PreProcess Dataset!\n")
	# Read the review dataset
	ReviewFilename_MaxReviewLen_Dict = ReviewDataProcess.DoneAllFile(ReviewDataFilePath=params.ReviewDataPath, ReviewDataFileProcessInfoPath=params.ProcessInfoPath, ReviewDataBinSavepath=params.ReviewDataSavepath, ReviewUserBinSavePath=params.Review_UserDataSavepath)

	# Combine the User from all the review dataset
	ReviewUserCombineData,max_product_len = CombineReviewData.CombineReviewDataSetByUser(ReviewUserBinSavePath=params.Review_UserDataSavepath, CombineReviewUserBinProcessPath=params.ProcessInfoPath,CombineReviewUserBinPath=params.Review_CombineUserDataSavepath)

	# Read the meta dataset
	MetaFilename_MaxQueryLen_Dict = MetaDataProcess.DoneAllFile(MetaDataFilePath=params.MetaDataPath, MetaDataFileProcessInfopath=params.ProcessInfoPath,MetaDataBinSavePath=params.MetaDataSavepath)

	# Combine all the meta dataset
	MetaCombineData = CombineMetaData.CombineMetaDataSet(MetaDataBinSavePath=params.MetaDataSavepath, CombineMetaDataBinProcessInfoPath=params.ProcessInfoPath,CombineMetaDataBinSavePath=params.Meta_CombineDataSavepath)
	
	max_review_len = max(zip(ReviewFilename_MaxReviewLen_Dict.values(), ReviewFilename_MaxReviewLen_Dict.keys()))[0] 
	max_query_len = max(zip(MetaFilename_MaxQueryLen_Dict.values(), MetaFilename_MaxQueryLen_Dict.keys()))[0] 
	# Create dataset
	DataSetSavePath = params.DataSetSavePath + "short_term_size_" + str(params.short_term_size) + "_long_term_size_" + str(params.long_term_size) + "/"
	Dataset = Data.DataSet(ReviewUserCombineData,MetaCombineData,params.neg_sample_num, max_query_len, max_review_len, max_product_len, DataSetSavePath, params.short_term_size, params.long_term_size)

	for i in range(len(CategorySet)):
		DataSetSavePath = DataSetSavePath + CategorySet[i] + "_"
	with open(DataSetSavePath + ".bin", "wb+") as f:
		pickle.dump(Dataset, f)
	#train_dataset = tf.data.Dataset.from_generator(Dataset.next_train_batch, (tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32))
	
	#train_dataset = train_dataset.repeat(params.epoch).batch(params.batch_size)
	print("End PreProcess Dataset!\n")
	return Dataset

def LoadData(params, DataSetSavePath, SaveFile):
	print("Start Load Dataset!\n")
	DataSetSavePath = DataSetSavePath + SaveFile
	with open(DataSetSavePath, "rb+") as f:
		Dataset = pickle.load(f)
	
	print("End Load Dataset!\n")
	
	#train_dataset = tf.data.Dataset.from_generator(Dataset.next_train_batch, (tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32))
	# need repeat epoch and set batch_size
	#train_dataset = train_dataset.repeat(params.epoch).batch(params.batch_size)
	return Dataset


def Generate_Data(params):
	if not os.path.exists(params.ReviewDataPath):
		raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), params.ReviewDataPath)

	if not os.path.exists(params.MetaDataPath):
		raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), params.MetaDataPath)

	CategorySet = list(Data_Util.GetCategory(params.MetaDataPath))
	DataSetSavePath = params.DataSetSavePath + "short_term_size_" + str(params.short_term_size) + "_long_term_size_" + str(params.long_term_size) + "/"
	if not os.path.exists(DataSetSavePath):
		os.makedirs(DataSetSavePath)
	
	SaveFile = Data_Util.FindFile(DataSetSavePath, CategorySet) 
	if (SaveFile is None):
		Dataset = PreProcessData(params, CategorySet)
	else:
		Dataset = LoadData(params, DataSetSavePath, SaveFile) 
	
	return Dataset

def Get_next_batch(Dataset, dataset, startpos, batch_size):
    dataset_len = len(dataset)
    CNIList = []
    CNWList = []
    if (startpos + batch_size) > dataset_len:
        uids = [dataset[i,0] for i in range(startpos, dataset_len)]
        before_pids_pos = [dataset[i,1] for i in range(startpos, dataset_len)]
        before_pids_pos_len = [dataset[i,2] for i in range(startpos, dataset_len)]
        current_pid_pos = [dataset[i,3] for i in range(startpos, dataset_len)]
        long_before_pids_pos = [dataset[i,4] for i in range(startpos, dataset_len)]
        long_before_pids_pos_len = [dataset[i,5] for i in range(startpos, dataset_len)]
        Qids_pos = [dataset[i,6] for i in range(startpos, dataset_len)]
        Len_pos = [dataset[i,7] for i in range(startpos, dataset_len)]
        cur_long_before_query_pos = [dataset[i,8] for i in range(startpos, dataset_len)]
        cur_long_before_query_pos_len = [dataset[i,9] for i in range(startpos, dataset_len)]
        current_text_ids = [dataset[i, 10] for i in range(startpos, dataset_len)]
        current_text_Len =  [dataset[i, 11] for i in range(startpos, dataset_len)]
        product_len_mask = [dataset[i, 12] for i in range(startpos, dataset_len)]
        query_len_mask = [dataset[i, 13] for i in range(startpos, dataset_len)]
        cur_long_before_query_len_mask = [dataset[i, 14] for i in range(startpos, dataset_len)]
        for i in range(dataset_len - startpos):
            current_neg_item, current_neg_word = Dataset.neg_sample()
            CNIList.append(current_neg_item)
            CNWList.append(current_neg_word)
            
    else:
        uids = [dataset[i,0] for i in range(startpos, startpos + batch_size)]
        before_pids_pos = [dataset[i,1] for i in range(startpos, startpos + batch_size)]
        before_pids_pos_len = [dataset[i,2] for i in range(startpos, startpos + batch_size)]
        current_pid_pos = [dataset[i,3] for i in range(startpos, startpos + batch_size)]
        long_before_pids_pos = [dataset[i,4] for i in range(startpos, startpos + batch_size)]
        long_before_pids_pos_len = [dataset[i,5] for i in range(startpos, startpos + batch_size)]
        Qids_pos = [dataset[i,6] for i in range(startpos, startpos + batch_size)]
        Len_pos = [dataset[i,7] for i in range(startpos, startpos + batch_size)]
        cur_long_before_query_pos = [dataset[i,8] for i in range(startpos, startpos + batch_size)]
        cur_long_before_query_pos_len = [dataset[i,9] for i in range(startpos, startpos + batch_size)]
        current_text_ids = [dataset[i, 10] for i in range(startpos, startpos + batch_size)]
        current_text_Len =  [dataset[i, 11] for i in range(startpos, startpos + batch_size)]
        product_len_mask = [dataset[i, 12] for i in range(startpos, startpos + batch_size)]
        query_len_mask = [dataset[i, 13] for i in range(startpos, startpos + batch_size)]
        cur_long_before_query_len_mask = [dataset[i, 14] for i in range(startpos, startpos + batch_size)]
        for i in range(batch_size):
            current_neg_item, current_neg_word = Dataset.neg_sample()
            CNIList.append(current_neg_item)
            CNWList.append(current_neg_word)
    return np.array(uids), np.array(before_pids_pos), np.array(before_pids_pos_len), np.array(current_pid_pos), \
	       np.array(long_before_pids_pos), np.array(long_before_pids_pos_len), np.array(Qids_pos), np.array(Len_pos), \
		   np.array(cur_long_before_query_pos), np.array(cur_long_before_query_pos_len), np.array(current_text_ids), np.array(current_text_Len),np.array(CNIList),np.array(CNWList),\
		   np.array(product_len_mask), np.array(query_len_mask), np.array(cur_long_before_query_len_mask)


# Test
#if __name__ == "__main__":
#	PreProcessData(ReviewDataPath='./Data/Review/',
# 					MetaDataPath='./Data/Meta/',
# 					ReviewDataSavepath='./AfterPreprocessData/ReviewBin/',  
#				    Review_UserDataSavepath = './AfterPreprocessData/ReviewUserBin/',
#				    Review_CombineUserDataSavepath = './AfterPreprocessData/ReviewUserCombineBin/',
#				    MetaDataSavepath ='./AfterPreprocessData/MetaBin/', 
#				    Meta_CombineDataSavepath = './AfterPreprocessData/MetaCombineBin/', 
#				    ProcessInfoPath= './InfoData/')