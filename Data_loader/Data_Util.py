import os

# All type of Amazon Data
category=[
    'Amazon_Instant_Video', #0
    'Apps_for_Android',  #1
    'Automotive', #2
    'Baby', #3
    'Beauty', #4
    'Books', #5
    'CDs_and_Vinyl', #6
    'Cell_Phones_and_Accessories', #7
    'Clothing_Shoes_and_Jewelry', #8
    'Digital_Music', #9
    'Electronics', #10
    'Grocery_and_Gourmet_Food', #11
    'Health_and_Personal_Care', #12
    'Home_and_Kitchen', #13
    'Kindle_Store', #14
    'Movies_and_TV', #15
    'Musical_Instruments', #16
    'Office_Products', #17
    'Patio_Lawn_and_Garden', #18
    'Pet_Supplies', #19
    'Sports_and_Outdoors', #20
    'Tools_and_Home_Improvement', #21
    'Toys_and_Games', #22
    'Video_Games' #23
]

def ReadFileList(Filepath):
    FileList = []
    for Filename in os.listdir(Filepath):
        FileList.append(Filename)
    return FileList

def FindFile(path, SelectDataset):
    for File in os.listdir(path):
        can_Find = True
        for i in range(len(SelectDataset)):
            if File.find(SelectDataset[i]) == -1:
                can_Find = False
                break
        if can_Find:
            return File
    return None



def GetCategory(Filepath):
    File_Category = list()
    FileList = ReadFileList(Filepath)
    for i in range(len(FileList)):
        for j in range(len(category)):
            if FileList[i].find(category[j]) != -1:
                File_Category.append(category[j])
    return set(File_Category)