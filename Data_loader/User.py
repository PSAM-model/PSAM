class UserData:
    def __init__(self, ReviewerID):
        self.UserID = ReviewerID

        self.UserPurchaseList = []

    def AddPurchase(self, ReviewInfo):
        PurchaseList = dict()
        # ReviewKey = ['reviewerID', 'asin', 'reviewText', 'reviewTime', 'unixReviewTime']
        PurchaseList['asin'] = ReviewInfo['asin']

        PurchaseList['reviewTime'] = ReviewInfo['reviewTime']

        PurchaseList['unixReviewTime'] = ReviewInfo['unixReviewTime']

        PurchaseList['reviewText'] = ReviewInfo['reviewText']

        self.UserPurchaseList.append(PurchaseList)

    def PrintUserInfo(self):
        print('UserID:%s' % self.UserID)
        print('UserPurchaseList:', self.UserPurchaseList)
    
    def GetUserPurchaseLen(self):
        return len(self.UserPurchaseList)