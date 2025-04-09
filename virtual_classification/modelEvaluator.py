# import the necessary packages
from model import datasets
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from keras.utils.np_utils import to_categorical
import numpy as np
import argparse
import locale
import os
import sys
import matplotlib.pylab as plt
import random
import time
import shutil 

testsetSelectPath = "data/shuffledDataList.txt"
inputPath = "data/format_each_allDatas.txt"
imagePath = "../image"
dataPath = "data"
bugPath = 'bugs'
tsPath = 'testSets'
processed_images_filename = "processed_images.npy"
processed_images_test_filename = "processed_images_test.npy"
dot_img_file = 'model_plot.png'
modelSummary = 'model_parameters.txt'
# need to change some paths and variables according to which dataset, to include some errors
# selector
best_model_filepath = "chosenModel/checkpoint_93.09%acc_0.209loss_K3_lr2.00e-04_dc1.00e-03_dataset8/best_model_F3.h5"


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
    selector = int(best_model_filepath.split('/')[1].split('_')[-1][-1])
    print("[INFO] The selector is:", selector)
    trainset, testset = splitDataSet(testsetSelectPath, selector)
    #print(len(flatten(trainset.values())))
    #print(len(flatten(testset.values())))
   
    df_test = df[df["image_path"].isin(testset)]
    df = df[df["image_path"].isin(trainset)]
    df = datasets.data_augmentation(df)
    #print(len(df))
    #print(len(df_test))
    print('[INFO] dataset', selector, 'has total data entry', len(df_test))
    

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
    
    # Print out the info
    model_details = extract_model_details(model)
    plot_model(model, to_file=dot_img_file, show_shapes=True)
    with open(modelSummary, 'w') as f:
        # Temporarily set the standard output to the file
        original = sys.stdout
        sys.stdout = f
        model.summary()
        # Restore standard output back to its original value
        sys.stdout = original
        
        for detail in model_details:
            for key, value in detail.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

    # make predictions on the testing data
    print("[INFO] computing accuracy...")
    preds = model.predict([testSetAttrX, testSetImagesX])
    neg = 0
    resPred = np.argmax(preds, axis=1)
    print("Pred:", resPred)
    resGT = np.argmax(testSetY, axis=1)
    print("GT:", resGT)
    #print(testSetY)
    
    # Remove all existing images in /bugs and /testset
    for f in os.listdir(bugPath):
        os.remove(os.path.join(bugPath, f))
    for f in os.listdir(tsPath):
        os.remove(os.path.join(tsPath, f))
    
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
    
    print("[INFO] Classification")
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
    
    print("[INFO] Classification")
    print("[INFO] Weighted Precision: %s" % "{:.2f}%".format(p_weighted * 100))
    print("[INFO] Weighted Recall: %s" % "{:.2f}%".format(r_weighted * 100))
    print("[INFO] Weighted F1-Score: %s" % "{:.2f}%".format(f_weighted * 100))
    
    # Result report
    print("[INFO] *******************************************")
    print("[INFO] Accuracy No Dup: %s" % "{:.2f}%".format(correct_nodup / total_nodup * 100))
    
    print("[INFO] another accuracy: {:.2f}%".format(np.sum(resPred == resGT) / len(preds) * 100))     
    #print(preds)
    (eval_loss, eval_accuracy) = model.evaluate( 
        [testSetAttrX, testSetImagesX], testSetY, batch_size=32, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100)) 
    print("[INFO] Loss: {}".format(eval_loss))
    
    # Calculate Binary Result
    # just like the method name
    getBinaryAcc(resPred, resGT)

def getBinaryAcc(pred, gt):
    biPred = (pred <= 1)
    biGT = (gt <= 1)
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
    
    
    print("[INFO] Classification")
    print("[INFO] Binary Accuracy: {:.2f}%".format(np.sum(biPred == biGT) / N * 100))
    print("[INFO] Binary Precision: {:.2f}%".format(precision * 100))
    print("[INFO] Binary Recall: {:.2f}%".format(recall * 100))
    print("[INFO] Binary F1-Score: {:.2f}%".format(f1 * 100))
    print("[INFO] Binary tp: %d" % tp)
    print("[INFO] Binary fp: %d" % fp)
    print("[INFO] Binary tn: %d" % tn)
    print("[INFO] Binary fn: %d" % fn)

def splitDataSet(path, selector):
    total = []
    trainset = []
    testset = []
    
    with open(path) as f:
        for line in f:
            img = line.rstrip('\n')
            total.append(img)
            
    count = 0
    ratio = 0.1
    numTestObj = int(len(total) * ratio)
    print("[INFO] Number of objects in test set", numTestObj)
    interval = int(1 / ratio)
    
    for img in total:
        if count % interval == selector and len(testset) < numTestObj:
            testset.append(img)
        else:
            trainset.append(img)
        count = count + 1
        
    assert len(testset) == numTestObj
    assert len(trainset) + len(testset) == len(total)
    return trainset, testset

def extract_model_details(model):
    details = []

    # Extracting layer details
    for layer in model.layers:
        config = layer.get_config()
        details.append(config)

    # Extracting optimizer details
    if model.optimizer:
        optimizer_config = model.optimizer.get_config()
        details.append(optimizer_config)

    return details

def flatten(l):
    return [item for sublist in l for item in sublist]
    

if __name__ == "__main__":
    #start_time = time.time()
    main()
    #print("--- Execution time: %s seconds ---" % (time.time() - start_time))