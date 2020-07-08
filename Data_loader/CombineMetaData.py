import pickle,os
import pandas as pd
from Data_loader.Data_Util import ReadFileList

def CombineMetaDataSet(MetaDataBinSavePath, CombineMetaDataBinProcessInfoPath, CombineMetaDataBinSavePath):
    if not os.path.exists(CombineMetaDataBinProcessInfoPath):
        os.makedirs(CombineMetaDataBinProcessInfoPath)

    if not os.path.exists(CombineMetaDataBinSavePath):
        os.makedirs(CombineMetaDataBinSavePath)

    MetaDataFileList = ReadFileList(MetaDataBinSavePath)
    AllMetaData = []
    MetaDataFileNameList = []
    for i in range(len(MetaDataFileList)):
        EachFileName = MetaDataFileList[i].replace('meta_','').replace('_Meta_Bin.bin','')
        MetaDataFileNameList.append(EachFileName)
    for i in range(len(MetaDataFileNameList)):
        CombineMetaDataBinSavePath = CombineMetaDataBinSavePath + MetaDataFileNameList[i] + "_"
    CombineMetaDataBinSavePath += '.bin'
    if os.path.exists(CombineMetaDataBinSavePath):
        print("Load Combine Meta Data!\n")
        with open(CombineMetaDataBinSavePath, "rb+") as f:
            AllMetaData = pickle.load(f)
        print("Load Combine Meta Data Finished!\n")
        return AllMetaData
    else:
        print("Start Combine Meta Data!\n")
        FileMetaNum = []
        for i in range(len(MetaDataFileList)):
            with open(MetaDataBinSavePath + MetaDataFileList[i], 'rb+') as f:
                EachMetaData = pickle.load(f)
                FileMetaNum.append(len(EachMetaData))
                AllMetaData.append(EachMetaData)

        #print("Load Meta Data File Done!")
        
        TimeToOperate = len(AllMetaData) - 1
        for t in range(TimeToOperate):
            I_MetaList = AllMetaData[0]
            J_MetaList = AllMetaData[1]
            del AllMetaData[0:2]
            I_J_MetaList = pd.concat((I_MetaList, J_MetaList))
            AllMetaData.append(I_J_MetaList)
        

        CMDInfo = ''
        for i in range(len(MetaDataFileNameList)):
            CMDInfo = CMDInfo + MetaDataFileNameList[i] + ' has user: ' + str(FileMetaNum[i]) + "\n"
        CMDInfo = CMDInfo + "CombineList has user: " + str(len(AllMetaData[0])) + "\n"

        with open(CombineMetaDataBinProcessInfoPath + "MetaDataCombineInfo.txt", "w+") as f:
            f.write(CMDInfo)
        
        with open(CombineMetaDataBinSavePath, "wb+") as f:
            pickle.dump(AllMetaData[0], f)
        print("End Combine Meta Data!\n")
        return AllMetaData[0]

# Test
# if __name__ == "__main__":
#     CombineMetaDataSet(MetaDataBinSavePath='./AfterPreprocessData/MetaBin/', CombineMetaDataBinProcessInfoPath='./InfoData/',CombineMetaDataBinSavePath='./AfterPreprocessData/MetaCombineBin/')