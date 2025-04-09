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

crossObjectPath = 'data/crossObjects.txt'
inputPath = "data/format_each_allDatas.txt"
imagePath = "../image"
dataPath = "data"
bugPath = 'bugs'
tsPath = 'testSets'
processed_images_filename = "processed_images.npy"
processed_images_test_filename = "processed_images_test.npy"
# need to change some paths and variables according to which dataset, to include some errors
# selector
best_model_filepath = "chosenModel/checkpoint_96.27%acc_0.110loss_K5_lr2.00e-04_dc1.00e-03_dataset7/best_model_F5.h5"



def main():
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
    trainset, testset = loadCrossObject(crossObjectPath, selector)
    #print(len(flatten(trainset.values())))
    print(len(flatten(testset.values())))
    #print(len(trainset))
    #print(len(testset))
   
    df_test = df[df["image_path"].isin(flatten(testset.values()))]
    df = df[df["image_path"].isin(flatten(trainset.values()))]
    df = datasets.data_augmentation(df)
    #print(len(df))
    #print(len(df_test))
    
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
    biResPred = (resPred <= 1)
    biResGT = (resGT <= 1)
    
    # Put all bug image together in /bugs
    # Remove all existing images in /bugs 
    for f in os.listdir(bugPath):
        os.remove(os.path.join(bugPath, f))
    for f in os.listdir(tsPath):
        os.remove(os.path.join(tsPath, f))
    
        
    # Put all bug image together in /bugs
    # Calculate acc for each object
    record = {}
    objPred = {}
    objGT = {}
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
        
        # Store the binary result for calculating the binary result later
        objPred[img] = biResPred[i]
        objGT[img] = biResGT[i]
        
        if img in record:
            continue
        if gt == 0 or gt == 1:
            shutil.copy(os.path.join(imagePath, img), os.path.join(tsPath, img))
        total_nodup = total_nodup + 1
        if pred == gt:
            correct_nodup = correct_nodup + 1
            
            # Count for calculating precision and recall
            for i in range(numOfClass):
                if i == gt:
                    tp[i] += 1
                else:
                    tn[i] += 1
            
            if pred == 0 or pred == 1:
                shutil.copy(os.path.join(imagePath, img), os.path.join(bugPath, img))
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
    
    print("[INFO] Cross-Object")
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
    
    print("[INFO] Cross-Object")
    print("[INFO] Weighted Precision: %s" % "{:.2f}%".format(p_weighted * 100))
    print("[INFO] Weighted Recall: %s" % "{:.2f}%".format(r_weighted * 100))
    print("[INFO] Weighted F1-Score: %s" % "{:.2f}%".format(f_weighted * 100))
 
    # Report result for each object in testset
    objectAcc = {}
    for key in testset:
        count = 0
        for item in testset[key]:
            if record[item]:
                count = count + 1
        acc = round(count / len(testset[key]) * 100, 2)
        objectAcc[key] = acc
    
    print("[INFO] printing acc for each item:")
    checkNum = 0
    for key in objectAcc:
        checkNum = checkNum + len(testset[key])
        print("%s has accuracy %s in %s datas" % (key, "{:.2f}%".format(objectAcc[key]), str(len(testset[key]))))
        
    assert checkNum == len(flatten(testset.values()))
    
    # Report Binary result for each object in testset
    # Repeat some step for easier modification
    print("************************** Binary acc for each item ************************")
    print("[INFO] printing Binary acc for each item:")
    for key in testset:
        obj_N = len(testset[key])
        obj_tp = 0
        obj_fp = 0
        obj_tn = 0
        obj_fn = 0
        
        for item in testset[key]:
            obj_pred = objPred[item]
            obj_gt = objGT[item]
            if obj_pred == obj_gt:
                if obj_pred == True:
                    obj_tp += 1
                else:
                    obj_tn += 1
            else:
                if obj_pred == True:
                    obj_fp += 1
                else:
                    obj_fn += 1
        
        obj_acc = (obj_tp + obj_tn) / obj_N
        
        try:
            obj_precision = obj_tp / (obj_tp + obj_fp)
        except:
            obj_precision = -1
        
        try:
            obj_recall = obj_tp / (obj_tp + obj_fn)
        except:
            obj_recall = -1
        
        try:
            obj_f1 = 2 * (obj_precision * obj_recall) / (obj_precision + obj_recall)
        except:
            obj_f1 = -1
                
        print("[INFO] Binary Cross-Object:", key)
        print("[INFO] Binary Object Accuracy: {:.2f}%".format(obj_acc * 100))
        print("[INFO] Binary Object Precision: {:.2f}%".format(obj_precision * 100))
        print("[INFO] Binary Object Recall: {:.2f}%".format(obj_recall * 100))
        print("[INFO] Binary Object F1-Score: {:.2f}%".format(obj_f1 * 100))
        print("[INFO] Binary Object tp: %d" % obj_tp)
        print("[INFO] Binary Object fp: %d" % obj_fp)
        print("[INFO] Binary Object tn: %d" % obj_tn)
        print("[INFO] Binary Object fn: %d" % obj_fn)
        print("************************** Next ************************")
        
        try:
            assert obj_tp + obj_fp + obj_tn + obj_fn == obj_N
        except:
            print("Num of this item is wrong:", key)
            return
        
    # Final Report
        
    print("************************* Final Report *************************")
    print("[INFO] Accuracy No Dup: %s" % "{:.2f}%".format(correct_nodup / total_nodup * 100))
    
    print("[INFO] another accuracy: {:.2f}%".format(np.sum(resPred == resGT) / len(preds) * 100))     
    #print(preds)
    (eval_loss, eval_accuracy) = model.evaluate( 
        [testSetAttrX, testSetImagesX], testSetY, batch_size=32, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100)) 
    print("[INFO] Loss: {}".format(eval_loss))
    
    # just like the method name
    getBinaryAcc(biResPred, biResGT)
    
def getBinaryAcc(biPred, biGT):
    
    #print(biPred)
    #print(biGT)
    
    N = len(biPred)
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(0, N):
        pred = biPred[i]
        gt = biGT[i]
        if pred == gt:
            if pred == True:
                tp += 1
                continue
            tn += 1
            continue
        if pred == True:
            fp += 1
            continue
        fn += 1
    
    assert tp + tn + fp + fn == N
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print("[INFO] *******************************************")
    
    
    print("[INFO] Cross-Object")
    print("[INFO] Binary Accuracy: {:.2f}%".format(np.sum(biPred == biGT) / N * 100))
    print("[INFO] Binary Precision: {:.2f}%".format(precision * 100))
    print("[INFO] Binary Recall: {:.2f}%".format(recall * 100))
    print("[INFO] Binary F1-Score: {:.2f}%".format(f1 * 100))
    print("[INFO] Binary tp: %d" % tp)
    print("[INFO] Binary fp: %d" % fp)
    print("[INFO] Binary tn: %d" % tn)
    print("[INFO] Binary fn: %d" % fn)
    
def loadCrossObject(path, selector):
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
    ratio = 0.1
    numTestObj = int(len(total) * ratio)
    print("[INFO] Number of objects in test set", numTestObj)
    interval = int(1 / ratio)
    
    for key, value in total.items():
        if count % interval == selector and len(testset) < numTestObj:
            testset[key] = value
        else:
            trainset[key] = value
        count = count + 1
    assert len(testset) == numTestObj
    assert len(trainset) + len(testset) == len(total)
    return trainset, testset

def flatten(l):
    return [item for sublist in l for item in sublist]
    
if __name__ == "__main__":
    #start_time = time.time()
    main()
    #print("--- Execution time: %s seconds ---" % (time.time() - start_time))