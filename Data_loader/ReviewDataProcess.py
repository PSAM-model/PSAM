import gzip,json,time,os,pickle,nltk,re
import numpy as np
import pandas as pd
from Data_loader.User import UserData
from Data_loader.Data_Util import ReadFileList
def ReadRawDataFile(Filename, Filepath, FileSavePath):
    g = gzip.open(Filepath, "r")
    ReviewList = []

    for l in g:
        # Remove the Enter for each line
        cut_l = json.loads(l.decode()[:-1])

        ReviewList.append(cut_l)

    ReviewValues = dict()
    ReviewKey = ['reviewerID', 'asin', 'reviewText', 'reviewTime', 'unixReviewTime']

    for k in ReviewKey:
        ReviewValues[k] = []

    for i in range(len(ReviewList)):
        try:
            for k in ReviewKey:
                a = ReviewList[i][k]

            for k in ReviewKey:
                ReviewValues[k].append(ReviewList[i][k])
        except:
            print('The %d row has problem! The row data:' % i, ReviewList[i])

    ReviewDatas = pd.DataFrame(ReviewValues)

    # Save Data
    pdsavepath = FileSavePath + Filename + '_Review_Bin.bin'
    # ReviewDatas.to_csv(pdsavepath)
    with open(pdsavepath, 'wb+') as f:
        pickle.dump(ReviewDatas, f)
    return ReviewDatas

def help_f_cut_stop_word(x):
    x = x.lower()
    x = re.sub(r'([;\.~\!@\#\$\%\^\&\*\(\(\)_\+\=\-\[\]\)/\|\'\"\?<>,`\\])','',x)
    ss = ""
    words = x.split(' ')
    stopwords = nltk.corpus.stopwords.words('english') + list(';.~!@#$:%^&*(()_+=-[])/|\'\"?<>,`\\1234567890')
    for w in words:
        if (w in stopwords):
            pass
        else:
            ss += ' ' + w
    return ss.lower().strip()




def UserAggregation(ReviewData):
    UserList = []
    First = True
    for index, row in ReviewData.iterrows():
        if First == True:
            User = UserData(row['reviewerID'])
            User.AddPurchase(row)
            UserList.append(User)
            First = False
        else:
            if row['reviewerID'] == UserList[len(UserList) - 1].UserID:
                UserList[len(UserList) - 1].AddPurchase(row)
            else:
                User = UserData(row['reviewerID'])
                User.AddPurchase(row)
                UserList.append(User)
    return UserList

def DoneAllFile(ReviewDataFilePath, ReviewDataFileProcessInfoPath, ReviewDataBinSavepath, ReviewUserBinSavePath):
    
    print("Start Review Data Process!\n")
    if not os.path.exists(ReviewDataFileProcessInfoPath):
        os.makedirs(ReviewDataFileProcessInfoPath)

    if not os.path.exists(ReviewDataBinSavepath):
        os.makedirs(ReviewDataBinSavepath)

    if not os.path.exists(ReviewUserBinSavePath):
        os.makedirs(ReviewUserBinSavePath)
    
    FileList = ReadFileList(ReviewDataFilePath)
    # print(FileList)

    Filename_MaxReviewLen_Dict = dict()
    
    for File in FileList:
        Filename = File.replace('reviews_','').replace('_5.json.gz','')
        InfoFilePath = ReviewDataFileProcessInfoPath + Filename + "_ReviewDataProcessInfo.txt"
        if os.path.exists(InfoFilePath):
            #print(Filename + " have been done!\n")
            continue
        with open(InfoFilePath, "w") as f:
            #print(Filename, "Start to read review data")
            startreadtime = time.time()
            ReviewDatas = ReadRawDataFile(Filename, ReviewDataFilePath + File, ReviewDataBinSavepath)
            endreadtime = time.time()
            #print(Filename, "end to read review data, time:", endreadtime - startreadtime)
            f.write("read review data time: %s s\n" % (str(endreadtime - startreadtime)))

            #print(Filename, "Start to cut stop words")
            startcutstopwordtime = time.time()
            ReviewDatas['reviewText'] = ReviewDatas['reviewText'].map(help_f_cut_stop_word)
            endcutstopwordtime = time.time()
            #print(Filename, "end to cut stop words, time:", endcutstopwordtime - startcutstopwordtime)
            f.write("cut stop words time: %s s\n" %( str(endcutstopwordtime - startcutstopwordtime)))

            #print(Filename, "Start to cut words")
            startcutwordtime = time.time()
            r_lens = []
            for i in ReviewDatas['reviewText']:
                r_lens.append(len(i.split(' ')))
            max_review_len = max(r_lens)
            mean_review_len = int(sum(r_lens) / len(r_lens))
            max_review_len = mean_review_len
            Filename_MaxReviewLen_Dict[Filename] = max_review_len
            def cutWord(x):
                if (len(x.split(' ')) > mean_review_len):
                    x = x[:mean_review_len]
                return x
            ReviewDatas['reviewText'] = ReviewDatas['reviewText'].map(cutWord)
            endcutwordtime = time.time()
            #print(Filename, "end to cut words, time:", endcutwordtime - startcutwordtime)
            f.write("cut words time: %s s\n"%( str(endcutwordtime - startcutwordtime)))

            #print(Filename, "Start to sort review data")
            startsorttime = time.time()
            ReviewDatas = ReviewDatas.sort_values(by=['reviewerID','unixReviewTime'], ascending=True)
            endsorttime = time.time()
            #print(Filename, "End to sort review data, time:", endsorttime - startsorttime)
            f.write("sort review data time: %s s\n" % (str(endsorttime - startsorttime)))

            #print(Filename, "Start to Aggregate User")
            startaggregatetime = time.time()
            UserList = UserAggregation(ReviewDatas)
            endaggregatetime = time.time()
            #print(Filename, "End to Aggregate User, time:", endaggregatetime - startaggregatetime)
            f.write("Aggregate Use time: %s s\n" % (str(endaggregatetime - startaggregatetime)))

            with open(ReviewUserBinSavePath + Filename + '_User_Bin.bin', 'wb+') as ff:
                pickle.dump(UserList, ff)
            #print(Filename, "Done!")
    
    MaxReviewLenFilePath = ReviewDataFileProcessInfoPath + "MaxReviewLen.txt"
    if os.path.exists(MaxReviewLenFilePath):
        with open(MaxReviewLenFilePath, "r+") as fff:
            for line in fff.readlines():
                line = line.strip()
                k = line.split(':')[0]
                v = line.split(':')[1]
                Filename_MaxReviewLen_Dict[k] = int(v)
    else:
        # record Each DataSet Max Review Len
        with open(MaxReviewLenFilePath, "w+") as fff:
            for k,v in Filename_MaxReviewLen_Dict.items():
                FMInfo = str(k) + ":" + str(v) + "\n"
                fff.write(FMInfo)
    

    print("End Review Data Process!\n")
    return Filename_MaxReviewLen_Dict


# Test
#if __name__ == "__main__":
#    DoneAllFile(ReviewDataFilePath = '../Data/AmazonData/review/Part/' , ReviewDataFileProcessInfoPath = './InfoData/', ReviewDataBinSavepath = './AfterPreprocessData/ReviewBin/', ReviewUserBinSavePath = './AfterPreprocessData/ReviewUserBin/')
