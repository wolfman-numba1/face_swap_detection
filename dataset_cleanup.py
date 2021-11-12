import os
testCoPath = "/Volumes/My1TBDrive/FF++/final_data/test/original/co/"
testFbPath = "/Volumes/My1TBDrive/FF++/final_data/test/original/fb/"

trainCoPath = "/Volumes/My1TBDrive/FF++/final_data/train/original/co/"
trainFbPath = "/Volumes/My1TBDrive/FF++/final_data/train/original/fb/"

valCoPath = "/Volumes/My1TBDrive/FF++/final_data/validation/original/co/"
valFbPath = "/Volumes/My1TBDrive/FF++/final_data/validation/original/fb/"
j = 0
for i in sorted(os.listdir(trainCoPath)):
    if j % 4 == 0:
        j = j + 1
    else:
        os.remove(trainCoPath + i)
        os.remove(trainFbPath + i.replace(".npy", ".jpg"))
        j = j + 1