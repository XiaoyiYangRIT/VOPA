# import the necessary packages
from model import datasets
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
import numpy as np
import argparse
import locale
import os
import matplotlib.pylab as plt
import random
import time
import shutil 

crossAppPath = 'data/crossApps.txt'
inputPath = "data/format_each_allDatas.txt"
imagePath = "../image"
dataPath = "data"
bugPath = 'bugs'
tsPath = 'testSets'
processed_images_filename = "processed_images.npy"
processed_images_test_filename = "processed_images_test.npy"
# need to change some paths and variables according to which dataset, to include some errors
# selector
best_model_filepath1 = "chosenModel/checkpoint_87.73acc_0.478loss_K3_lr2.00e-04_dc1.00e-03_dataset0/best_model_F3.h5"
best_model_filepath2 = "chosenModel/checkpoint_78.81acc_0.541loss_K1_lr2.00e-04_dc1.00e-03_dataset1/best_model_F1.h5"
best_model_filepath3 = "chosenModel/checkpoint_88.04acc_0.345loss_K3_lr2.00e-04_dc1.00e-03_dataset2/best_model_F3.h5"
best_model_filepath4 = "chosenModel/checkpoint_89.93acc_0.295loss_K3_lr2.00e-04_dc1.00e-03_dataset3/best_model_F3.h5"
best_model_filepath5 = "chosenModel/checkpoint_88.37acc_0.517loss_K1_lr2.00e-04_dc1.00e-03_dataset4/best_model_F1.h5"
best_model_filepath6 = "chosenModel/checkpoint_92.08acc_0.185loss_K1_lr2.00e-04_dc1.00e-03_dataset5/best_model_F1.h5"
best_model_filepath7 = "chosenModel/checkpoint_75.96acc_0.730loss_K3_lr2.00e-04_dc1.00e-03_dataset6/best_model_F3.h5"

best_path = [best_model_filepath1, 
             best_model_filepath2, 
             best_model_filepath3, 
             best_model_filepath4, 
             best_model_filepath5, 
             best_model_filepath6, 
             best_model_filepath7]
#numOfApps = 7


def main():
    numOfApps = len(best_path)
    allAccs = []
    allLosses = []
    allPrecisions = []
    allRecalls = []
    allF1Scores = []
    allwp = []
    allwr = []
    allwf = []
    allBiAcc = []
    allBiP = []
    allBiR = []
    allBif1 = []
    allBiTP = []
    allBiFP = []
    allBiTN = []
    allBiFN = []
    for path in best_path:
        (acc, loss, precision, recall, f1score, wp, wr, wf, biList) = myEvaluator(path)
        allAccs.append(acc)
        allLosses.append(loss)
        allPrecisions.append(precision)
        allRecalls.append(recall)
        allF1Scores.append(f1score)
        allwp.append(wp)
        allwr.append(wr)
        allwf.append(wf)
        
        allBiAcc.append(biList[0])
        allBiP.append(biList[1])
        allBiR.append(biList[2])
        allBif1.append(biList[3])
        allBiTP.append(biList[4])
        allBiFP.append(biList[5])
        allBiTN.append(biList[6])
        allBiFN.append(biList[7])
        #[acc, precision, recall, f1, tp, fp, tn, fn]
        
    print('[INFO] All Accuracy:', allAccs)
    print('[INFO] All Losses:', allLosses)
    print('[INFO] All Precisions:', allPrecisions)
    print('[INFO] All Recalls:', allRecalls)
    print('[INFO] All F1-Scores:', allF1Scores)
    print('[INFO] All Weighted Precisions:', allwp)
    print('[INFO] All Weighted Recalls:', allwr)
    print('[INFO] All Weighted F1-Scores:', allwf)
    print('[INFO] All Binary Accuracy:', allBiAcc)
    print('[INFO] All Binary Precisions:', allBiP)
    print('[INFO] All Binary Recalls:', allBiR)
    print('[INFO] All Binary F1-Scores:', allBif1)
    print('[INFO] All Binary TP:', allBiTP)
    print('[INFO] All Binary FP:', allBiFP)
    print('[INFO] All Binary TN:', allBiTN)
    print('[INFO] All Binary FN:', allBiFN)
    
    print("[INFO] Average Accuracy: {:.2f}%".format(sum(allAccs) / numOfApps * 100))
    print("[INFO] Average Losses: {:.2f}%".format(sum(allLosses) / numOfApps * 100))
    print("[INFO] Average Precisions: {:.2f}%".format(sum(allPrecisions) / numOfApps * 100))
    print("[INFO] Average Recalls: {:.2f}%".format(sum(allRecalls) / numOfApps * 100))
    print("[INFO] Average F1-Scores: {:.2f}%".format(sum(allF1Scores) / numOfApps * 100))
    print("[INFO] Average Weighted Precisions: {:.2f}%".format(sum(allwp) / numOfApps * 100))
    print("[INFO] Average Weighted Recalls: {:.2f}%".format(sum(allwr) / numOfApps * 100))
    print("[INFO] Average Weighted F1-Scores: {:.2f}%".format(sum(allwf) / numOfApps * 100))
    print("[INFO] Average Binary Accuracy: {:.2f}%".format(sum(allBiAcc) / numOfApps * 100))
    print("[INFO] Average Binary Precisions: {:.2f}%".format(sum(allBiP) / numOfApps * 100))
    print("[INFO] Average Binary Recalls: {:.2f}%".format(sum(allBiR) / numOfApps * 100))
    print("[INFO] Average Binary F1-Scores: {:.2f}%".format(sum(allBif1) / numOfApps * 100))
    print("[INFO] Average Binary TPs: {:d}".format(int(sum(allBiTP) / numOfApps)))
    print("[INFO] Average Binary FPs: {:d}".format(int(sum(allBiFP) / numOfApps)))
    print("[INFO] Average Binary TNs: {:d}".format(int(sum(allBiTN) / numOfApps)))
    print("[INFO] Average Binary FNs: {:d}".format(int(sum(allBiFN) / numOfApps)))
    

def myEvaluator(best_model_filepath):
    seed_value= 0
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # construct the argument parser and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-d", "--dataset", type=str, required=True, help="path to input dataset of images")
    #args = vars(ap.parse_args())# construct the path to the input .txt file that contains information
    # on each house in the dataset and then load the dataset
    print("[INFO] loading attributes...")
    
    df = datasets.load_attributes(inputPath)
    #selector = 2
    selector = int(best_model_filepath.split('/')[1].split('_')[-1][-1])
    print("[INFO] The selector is:", selector)
    trainset, testset = loadCrossApp(crossAppPath, selector)
    #print(len(flatten(trainset.values())))
    #print(len(flatten(testset.values())))
   
    df_test = df[df["image_path"].isin(flatten(testset.values()))]
    df = df[df["image_path"].isin(flatten(trainset.values()))]
    df = datasets.data_augmentation(df)
    
    # load the house images and then scale the pixel intensities to the
    # range [0, 1]
    print("[INFO] loading images...")
    # split the data into train set and test set, store the data into file for fast load later
    # Randomly select the test set, store the file into different folder by different selector(related to random seed)
    # Generate the dir if not exist
    processedImagePath = os.path.join(dataPath, str(selector), processed_images_filename)
    if not os.path.exists(os.path.join(dataPath, str(selector))):
        os.makedirs(os.path.join(dataPath, str(selector)))
    
    if os.path.exists(processedImagePath):
        images = np.load(processedImagePath)
    else:
        images = datasets.load_images(df, imagePath)
        images = images / 255.0
        np.save(processedImagePath, images)
        
    processedImageTestPath  = os.path.join(dataPath, str(selector), processed_images_test_filename)
    if os.path.exists(processedImageTestPath):
        images_test = np.load(processedImageTestPath)
    else:
        images_test = datasets.load_images(df_test, imagePath)
        images_test = images_test / 255.0
        np.save(processedImageTestPath, images_test)
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    print("[INFO] processing data...")
    
    # Prepare the test set
    testSetImagesX = images_test
    testSetAttrX = df_test
    testSetY = np.array(testSetAttrX["GT_label"]) - 1
    testSetY = to_categorical(testSetY, num_classes=4)
    testSetAttrX = datasets.process_testset(testSetAttrX)

    #Load and evaluate the best model version
    print("[INFO] best model is {}".format(best_model_filepath))
    model = load_model(best_model_filepath)

    # make predictions on the testing data
    print("[INFO] computing accuracy...")
    preds = model.predict([testSetAttrX, testSetImagesX])
    resPred = np.argmax(preds, axis=1)
    print(resPred)
    resGT = np.argmax(testSetY, axis=1)
    print(resGT)
    #print(testSetY)
    
    # just like the method name
    #getBinaryAcc(resPred, resGT)
    
    # Put all bug image together in /bugs
    # Remove all existing images in /bugs 
    bugPathPlusApp = os.path.join(bugPath, str(selector))
    checkPathExist(bugPathPlusApp)
    for f in os.listdir(bugPathPlusApp):
        os.remove(os.path.join(bugPathPlusApp, f))
        
    tsPathPlusApp = os.path.join(tsPath, str(selector))
    checkPathExist(tsPathPlusApp)
    for f in os.listdir(tsPathPlusApp):
        os.remove(os.path.join(tsPathPlusApp, f))
    
    # *****************************************************
    
    # Calculate acc for each app
#    record = {}
#    for i in range(0, len(preds)):
#        pred = resPred[i]
#        gt = resGT[i]
#        img = df_test.iloc[i]['image_path']
#        if img in record:
#            continue
#        #if gt == 0 or gt == 1:
#        #    shutil.copy(os.path.join(imagePath, img), os.path.join(tsPathPlusApp, img))
#        if pred == gt:
#            if pred == 0 or pred == 1:
#                shutil.copy(os.path.join(imagePath, img), os.path.join(bugPathPlusApp, img))
#            record[img] = True
#            continue
#        record[img] = False
    
    # Put all bug image together in /bugs
    # Calculate acc for each object
    record = {}
    total_nodup = 0
    correct_nodup = 0
    # true positive, false positive, true negative, false negative
    numOfClass = 4
    tp = [0] * numOfClass
    fp = [0] * numOfClass
    tn = [0] * numOfClass
    fn = [0] * numOfClass
    validClass = []
    
    # check which classes may not in the test set
    for i in range(numOfClass):
        if (i in resPred) or (i in resGT):
            validClass.append(i)
            continue
        print('Class', i, "is not in the gt or preds")
    
    # process the result
    for i in range(0, len(preds)):
        pred = resPred[i]
        gt = resGT[i]
        img = df_test.iloc[i]['image_path']
        if img in record:
            continue
        #if gt == 0 or gt == 1:
        #    shutil.copy(os.path.join(imagePath, img), os.path.join(tsPathPlusApp, img))
        total_nodup = total_nodup + 1
        if pred == gt:
            correct_nodup = correct_nodup + 1
            
            # Count for calculating precision and recall
            for i in range(numOfClass):
                if i == gt:
                    tp[i] += 1
                else:
                    tn[i] += 1
            
            #if pred == 0 or pred == 1:
            #    shutil.copy(os.path.join(imagePath, img), os.path.join(bugPathPlusApp, img))
            record[img] = True
            continue
        # if prediction not equal to the gt   
        for i in range(numOfClass):
            if i == gt:
                fn[i] += 1
            elif i == pred:
                fp[i] += 1
            else:
                tn[i] += 1
        record[img] = False
    
    #Double check the total number is correct
    assert total_nodup == len(record)
    
    precisions = [0] * numOfClass
    recalls = [0] * numOfClass
    
    for i in validClass:
        print("tp:", tp[i])
        print("fp:", fp[i])
        print("tn:", tn[i])
        print("fn:", fn[i])
        print("Total number:", tp[i] + fp[i] + tn[i] + fn[i])
        assert tp[i] + fp[i] + tn[i] + fn[i] == len(record)
        if tp[i] == 0:
            precisions[i] = 0
            recalls[i] = 0
            continue
        precisions[i] = tp[i] / (tp[i] + fp[i])
        recalls[i] = tp[i] / (tp[i] + fn[i])
        
    
    print("For each precisions: ", precisions)
    print("For each recalls: ", recalls)
    print("How many:", len(validClass))
    # Calculate the average
    precision = sum(precisions) / len(validClass)
    recall = sum(recalls) / len(validClass)
    f1score = 2 * (precision * recall) / (precision + recall)
    
    print("[INFO] Cross-App")
    print("[INFO] Precision: %s" % "{:.2f}%".format(precision * 100))
    print("[INFO] Recall: %s" % "{:.2f}%".format(recall * 100))
    print("[INFO] F1-Score: %s" % "{:.2f}%".format(f1score * 100))
    
    # Calculate weighted mean
    precisions_weighted = [0] * numOfClass
    recalls_weighted = [0] * numOfClass
    for i in validClass:
        precisions_weighted[i] = precisions[i] * np.count_nonzero(resGT == i)
        recalls_weighted[i] = recalls[i] * np.count_nonzero(resGT == i)
        #print("Num of occurrance in gt:", np.count_nonzero(resGT == i))
    p_weighted = sum(precisions_weighted) / len(resGT)
    r_weighted = sum(recalls_weighted) / len(resGT)
    f_weighted = 2 * (p_weighted * r_weighted) / (p_weighted + r_weighted)
    
    print("[INFO] Cross-App")
    print("[INFO] Weighted Precision: %s" % "{:.2f}%".format(p_weighted * 100))
    print("[INFO] Weighted Recall: %s" % "{:.2f}%".format(r_weighted * 100))
    print("[INFO] Weighted F1-Score: %s" % "{:.2f}%".format(f_weighted * 100))
    
    # Result report
    
    
 
    objectAcc = {}
    for key in testset:
        count = 0
        for item in testset[key]:
            if record[item]:
                count = count + 1
        acc = round(count / len(testset[key]) * 100, 2)
        objectAcc[key] = acc
    
    print("[INFO] printing acc for each item:")
    for key in objectAcc:
        print("%s has accuracy: %s" % (key, "{:.2f}%".format(objectAcc[key])))
    
    print("[INFO] another accuracy: {:.2f}%".format(np.sum(resPred == resGT) / len(preds) * 100))     
    #print(preds)
    (eval_loss, eval_accuracy) = model.evaluate( 
        [testSetAttrX, testSetImagesX], testSetY, batch_size=32, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100)) 
    print("[INFO] Loss: {}".format(eval_loss))
    # just like the method name
    biList = getBinaryAcc(resPred, resGT, df_test, tsPathPlusApp, bugPathPlusApp)
    
    return (eval_accuracy, eval_loss, precision, recall, f1score, p_weighted, r_weighted, f_weighted, biList)
    
def getBinaryAcc(pred, gt, df_test, tsPathPlusApp, bugPathPlusApp):
    biPred = (pred <= 1)
    biGT = (gt <= 1)
    
    assert len(biPred) == len(pred)
    assert len(biGT) == len(gt)
    #print(biPred)
    #print(biGT)
    
    N = len(biPred)
    acc = np.sum(biPred == biGT) / N
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(0, N):
        pred = biPred[i]
        gt = biGT[i]
        img = df_test.iloc[i]['image_path']
        
        # Generate test set folder
        if gt == True:
            shutil.copy(os.path.join(imagePath, img), os.path.join(tsPathPlusApp, img))
        
        if pred == gt:
            if pred == True:
                # Generate bug found folder
                shutil.copy(os.path.join(imagePath, img), os.path.join(bugPathPlusApp, img))
                
                tp += 1
                continue
            tn += 1
            continue
        if pred == True:
            fp += 1
            continue
        fn += 1
    
    assert tp + tn + fp + fn == N
    try:
        precision = tp / (tp + fp)
    except:
        precision = -1
    try:
        recall = tp / (tp + fn)
    except:
        recall = -1
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1 = -1
    
    print("[INFO] *******************************************")
    
    
    print("[INFO] Cross App")
    print("[INFO] Binary Accuracy: {:.2f}%".format(acc * 100))
    print("[INFO] Binary Precision: {:.2f}%".format(precision * 100))
    print("[INFO] Binary Recall: {:.2f}%".format(recall * 100))
    print("[INFO] Binary F1-Score: {:.2f}%".format(f1 * 100))
    print("[INFO] Binary tp: %d" % tp)
    print("[INFO] Binary fp: %d" % fp)
    print("[INFO] Binary tn: %d" % tn)
    print("[INFO] Binary fn: %d" % fn)
    
    return [acc, precision, recall, f1, tp, fp, tn, fn]
    
def loadCrossApp(path, selector):
    total = {}
    trainset = {}
    testset = {}
    
    with open(path) as f:
        for line in f:
            line = line.split(':')
            item = line[0]
            imgs = line[1].strip().split(',')
            total[item] = imgs
    count = 0
    #ratio = 0.1
    numTest = 1
    interval = len(total) / numTest
    for key, value in total.items():
        if count % interval == selector:
            print(key, "is in test set")
            testset[key] = value
        else:
            trainset[key] = value
        count = count + 1
    assert len(trainset) + len(testset) == len(total)
    return trainset, testset

"""
Flatten a multiple dimensional list to one dimensional list

Parameter: 
    l: the input list

Return:
    The flattened list
"""
def flatten(l):
    return [item for sublist in l for item in sublist]

"""
Check if the path exists. If not, create the target directory

Parameter:
    path: the target path
    
Return:
    None
"""
def checkPathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
if __name__ == "__main__":
    #start_time = time.time()
    main()
    #print("--- Execution time: %s seconds ---" % (time.time() - start_time))