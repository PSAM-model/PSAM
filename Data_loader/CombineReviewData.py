import operator,glob,pickle,os
from Data_loader.User import UserData
from Data_loader.Data_Util import ReadFileList
def FindTheSameUser(U1, U2):
    User1List = []
    for i in range(len(U1)):
        User1List.append(U1[i].UserID)
    User2List = []
    for i in range(len(U2)):
        User2List.append(U2[i].UserID)
    Intersection_User = [x for x in User1List if x in User2List]
    CombineList = []
    for i in range(len(U1)):
        if U1[i].UserID in Intersection_User:
            CombineList.append(U1[i])
    for i in range(len(U2)):
        if U2[i].UserID in Intersection_User:
            CombineList.append(U2[i])
    
    cmpfun = operator.attrgetter('UserID')
    CombineList.sort(key=cmpfun)
    return CombineList

def CombineUser(U1, U2):
    CombineList = []
    for i in range(len(U1)):
        CombineList.append(U1[i])
    for i in range(len(u2)):
        CombineList.append(U2[i])
    cmpfun = operator.attrgetter('UserID')
    CombineList.sort(key=cmpfun)
    return CombineList

def Getmaxproductnum(AllReviewUserList):
    UserProLen = []
    for i in range(len(AllReviewUserList)):
        eachUserProlen = AllReviewUserList[i].GetUserPurchaseLen()
        UserProLen.append(eachUserProlen)
    MaxProductNum = max(UserProLen)
    return MaxProductNum

def CombineReviewDataSetByUser(ReviewUserBinSavePath, CombineReviewUserBinProcessPath, CombineReviewUserBinPath):
    
    if not os.path.exists(CombineReviewUserBinPath):
        os.makedirs(CombineReviewUserBinPath)

    ReviewUserFileList = ReadFileList(ReviewUserBinSavePath)
    
    ReviewUserCombineSaveFilePath = CombineReviewUserBinPath
    
    AllReviewUserList = []
    FileNameList = []
    
    for i in range(len(ReviewUserFileList)):
        EachFileName = ReviewUserFileList[i].replace('_User_Bin.bin','')
        FileNameList.append(EachFileName)
    for i in range(len(FileNameList)):
        ReviewUserCombineSaveFilePath = ReviewUserCombineSaveFilePath + FileNameList[i] + "_"
    ReviewUserCombineSaveFilePath = ReviewUserCombineSaveFilePath + '_Combine_User_Bin.bin'
    if os.path.exists(ReviewUserCombineSaveFilePath):
        print("Load Combine Review Data!\n")
        with open(ReviewUserCombineSaveFilePath, "rb+") as f:
            AllReviewUserList = pickle.load(f)
        MaxProductNum = Getmaxproductnum(AllReviewUserList)
        print("Load Combine Review Data Finished!\n")
        return AllReviewUserList, MaxProductNum
    else:
        print("Start Combine Review Data!\n")
        FileUserNum = []
        for i in range(len(ReviewUserFileList)):
            with open(ReviewUserBinSavePath + ReviewUserFileList[i], 'rb+') as f:
                EachReviewUser = pickle.load(f)
                FileUserNum.append(len(EachReviewUser))
                AllReviewUserList.append(EachReviewUser)
        #print("Load Review User File Done!")
        TimeToOperate = len(AllReviewUserList) - 1
        for t in range(TimeToOperate):
            I_UserList = AllReviewUserList[0]
            J_UserList = AllReviewUserList[1]
            del AllReviewUserList[0:2]
            I_J_UserList = CombineUser(I_UserList, J_UserList)
            CombineUserList  = []
            CombineUserList.append(I_J_UserList[0])
            for k in range(1, len(I_J_UserList)):
                if I_J_UserList[k].UserID == CombineUserList[len(CombineUserList) - 1].UserID:
                    for l in range(len(I_J_UserList[k].UserPurchaseList)):
                        CombineUserList[len(CombineUserList) - 1].AddPurchase(I_J_UserList[k].UserPurchaseList[l])
                else:
                    CombineUserList[len(CombineUserList) - 1].UserPurchaseList = sorted(CombineUserList[len(CombineUserList) - 1].UserPurchaseList, key=operator.itemgetter('unixReviewTime'))
                    CombineUserList.append(I_J_UserList[k])
            AllReviewUserList.append(CombineUserList)
        
        # 记录用户的购买序列的最长长度
        MaxProductNum = Getmaxproductnum(AllReviewUserList[0])
        
        
        RUCInfo = ''
        
        for i in range(len(FileNameList)):
            RUCInfo = RUCInfo + FileNameList[i] + ' has user: ' + str(FileUserNum[i]) + "\n"
        RUCInfo = RUCInfo + "CombineList has user: " + str(len(AllReviewUserList[0])) + "\n"
        with open(CombineReviewUserBinProcessPath + "ReviewUserCombineInfo.txt", "w+") as f:
            f.write(RUCInfo)
        with open(ReviewUserCombineSaveFilePath, "wb+") as f:
            pickle.dump(AllReviewUserList[0], f)
        print('End Combine Review Data!\n')
        return AllReviewUserList[0], MaxProductNum
    

    


# Test
#if __name__ == "__main__":
#    CombineReviewDataSetByUser(ReviewUserBinSavePath='./AfterPreprocessData/ReviewUserBin/', CombineReviewUserBinProcessPath='./InfoData/',CombineReviewUserBinPath='./AfterPreprocessData/ReviewUserCombineBin/')
