# import the necessary packages
from model import datasets
import numpy as np
import os
import random
import time
import shutil 

inputPath = "data/shuffledDataList.txt"
imagePath = "../image"
dataPath = "data"
tsPath = 'allTestSets'

# need to change some paths and variables according to which dataset, to include some errors
# test_set_list, stat_test_set_list, selector



testSetRatio = 0.1

def main():
    

    # construct the argument parser and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-d", "--dataset", type=str, required=True, help="path to input dataset of images")
    #args = vars(ap.parse_args())# construct the path to the input .txt file that contains information
    # on each house in the dataset and then load the dataset
    print("[INFO] loading attributes...")
    
    df = datasets.load_attributes(inputPath)
    #print(len(df))
    #print(int(len(df) * 0.1))
    selectNum = int(1 / testSetRatio)
    
    for selector in range(10):
        
            
        if not os.path.exists(os.path.join(dataPath, str(selector))):
            os.makedirs(os.path.join(dataPath, str(selector)))
        
        df_test = df[df.index % selectNum == selector]

        test_set_list = os.path.join(dataPath, str(selector), 'test_set_list.txt')
        stat_test_set_list = os.path.join(dataPath, str(selector), 'stat_test_set_list.txt')
        if os.path.exists(test_set_list):
            os.remove(test_set_list)
            
        record = []
        with open(test_set_list, 'a') as f:
            for index, row in df_test.iterrows():
                f.write(str(row['image_path']))
                f.write(str(row['GT_label']))
                f.write('\n')
                if str(row['image_path']) not in record:
                    record.append(str(row['image_path']))


        # load the house images and then scale the pixel intensities to the
        # range [0, 1]
        
        print("[INFO] loading images...")
        dirN = os.path.join(tsPath, str(selector))
                                  
        if not os.path.exists(dirN):
            os.makedirs(dirN)

        # Put all bug image together in /bugs
        # Firstly, remove all existing images in /bugs 
        for f in os.listdir(dirN):
            os.remove(os.path.join(dirN, f))
                                  
        for imageName in record:
            imageName = imageName.strip(' ')
            if not os.path.exists(os.path.join(dirN, imageName)):
                shutil.copy(os.path.join(imagePath, imageName), os.path.join(dirN, imageName))


    

if __name__ == "__main__":
    #start_time = time.time()
    main()
    #print("--- Execution time: %s seconds ---" % (time.time() - start_time))