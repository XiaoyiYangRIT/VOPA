# import the necessary packages
from model import datasets
from model import models
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.layers import concatenate
from keras.utils.np_utils import to_categorical 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import argparse
import locale
import os
import matplotlib.pylab as plt
import random
import shutil 
import math
import sys

testsetSelectPath = "data/shuffledDataList.txt"
inputPath = "data/format_each_allDatas_regression.txt"
outputPath = "checkpoints/checkpoint"
imagePath = "../image"
dataPath = "data"
processed_images_filename = "processed_images.npy"
processed_images_test_filename = "processed_images_test.npy"


def main():
    if len(sys.argv) != 2:
        print("[ERROR] Wrong num of argument, need 2 but got", len(sys.argv))
        return
    arg1 = sys.argv[1]
    if not arg1.isdigit():
        print("[ERROR] Argument is not digit")
        return
    #remove output path if exist in case of bug
    if os.path.exists(outputPath):
        try:
            shutil.rmtree(outputPath)
        except OSError as e:
            print("Error: %s : %s" % (outputPath, e.strerror))
    
    seed_value= 0
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    print("[INFO] loading attributes...")
    
    df = datasets.load_attributes(inputPath)
    print('[INFO] Total data number', len(df))
    selector = int(arg1)
    #selector = 0
    trainset, testset = splitDataSet(testsetSelectPath, selector)
    #print(len(flatten(trainset.values())))
    #print(len(flatten(testset.values())))
   
    df_test = df[df["image_path"].isin(testset)]
    df = df[df["image_path"].isin(trainset)]
    
    #print(len(df))
    #print(len(df_test))
    print('[INFO] dataset', selector, 'has total data entry', len(df_test))
    assert len(df_test) == len(testset)
    

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
    # find the largest house price in the training set and use it to
    # scale our house prices to the range [0, 1] (will lead to better
    # training and convergence)
    
    #lst_lr = [2.1e-4, 2.2e-4, 2.3e-4, 2.4e-4, 2.5e-4]
    #for lr in lst_lr:
    #    train(images, df, images_test, df_test, selector, lr, 1e-3)
        
    #lst_dc = [1.2e-3, 1.1e-3, 1e-3, 9e-4, 8e-4]
    #for dc in lst_dc:
    #    train(images, df, images_test, df_test, selector, 2e-4, dc)
    for i in range(4):
        train(images, df, images_test, df_test, selector, 2e-4, 1e-3)

def train(images, df, images_test, df_test, selector, lr = 1e-4, dc = 1e-3):
    dc_base = 200
    Fold = 1
    min_val_loss = 1000
    cv = KFold(n_splits=5, random_state=33, shuffle=True)
    cs = MinMaxScaler()
    
    # Prepare the test set
    testSetImagesX = images_test
    testSetAttrX = df_test
    testSetY = cs.fit_transform(np.array(testSetAttrX["GT_label"]).reshape(-1,1)).reshape(-1)
    testSetAttrX = datasets.process_testset(testSetAttrX)
    
    for train_index, test_index in cv.split(images):
        print("train_index:", len(train_index))
        print("test_index:", len(test_index))

        trainImagesX = images[train_index]
        testImagesX = images[test_index]
        # print("trainImagesX:", trainImagesX.shape)
        # print("testImagesX:", testImagesX.shape)

        trainAttrX = df.iloc[train_index]
        testAttrX = df.iloc[test_index]
        # print("trainAttrX:", trainAttrX)
        # print("testAttrX:", testAttrX)

        # trainY = np.array(trainAttrX["GT_label"])
        # testY = np.array(testAttrX["GT_label"])
        # trainY = to_categorical(trainY, num_classes=4)
        # testY = to_categorical(testY, num_classes=4)

        
        trainY = cs.transform(np.array(trainAttrX["GT_label"]).reshape(-1,1)).reshape(-1)
        testY = cs.transform(np.array(testAttrX["GT_label"]).reshape(-1,1)).reshape(-1)
        #print("trainY:", trainY)
        #print("testY:", testY)

        # process the house attributes data by performing min-max scaling
        # on continuous features, one-hot encoding on categorical features,
        # and then finally concatenating them together
        (trainAttrX, testAttrX) = datasets.process_attributes(df, trainAttrX, testAttrX)
        # print("trainAttrX:", trainAttrX.shape)
        # print("testAttrX:", testAttrX.shape)

        # create the MLP and CNN models
        mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
        cnn = models.create_cnn(108, 240, 3, regress=False)
        # create the input to our final set of layers as the *output* of both
        # the MLP and CNN
        combinedInput = concatenate([mlp.output, cnn.output])
        # print("combinedInput:", combinedInput.shape) #(None, 8)

        # our final FC layer head will have two dense layers, the final one
        # being our regression head
        x = Dense(4, activation="relu")(combinedInput)
        x = Dense(1, activation="sigmoid")(x)
        # our final model will accept categorical/numerical data on the MLP
        # input and images on the CNN input, outputting a single value (the
        # predicted price of the house)
        model = Model(inputs=[mlp.input, cnn.input], outputs=x)
        # compile the model using mean absolute percentage error as our loss,
        # implying that we seek to minimize the absolute percentage difference
        # between our price *predictions* and the *actual prices*
        # filepath = './checkpoint/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5'
        filepath = os.path.join(outputPath, 'best_model_F{}.h5'.format(Fold))
        checkpoint = ModelCheckpoint(filepath, 
            verbose=1, 
            monitor='val_loss',
            save_best_only=True, 
            mode='min'
        ) 
        es = EarlyStopping(monitor='val_loss', mode='min', patience=3)

        opt = Adam(learning_rate=lr, decay=dc / dc_base)
        # opt = RMSprop(learning_rate=1e-4)
        # opt = SGD(learning_rate=1e-4)
        model.compile(loss="mean_squared_error", optimizer=opt)

        # train the model
        print("[INFO] training model...")
        # history  = model.fit(
        #                 x=[trainAttrX, trainImagesX], y=trainY,
        #                 validation_data=([testAttrX, testImagesX], testY),
        #                 epochs=100, batch_size=32, callbacks=[es, checkpoint])

        history  = model.fit(
                        x=[trainAttrX, trainImagesX], y=trainY,
                        validation_data=([testAttrX, testImagesX], testY),
                        epochs=100, batch_size=32, callbacks=[checkpoint])

        # acc = history.history['acc']
        # val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        # plt.plot(epochs, acc, 'r', label='Training acc')
        # plt.plot(epochs, val_acc, 'b', label='Validation acc')
        # plt.title('Training and validation accuracy')
        # plt.ylabel('accuracy') 
        # plt.xlabel('epoch')
        # plt.legend()
        # plt.savefig("accuracy_K{}.jpg".format(Fold))
        # plt.clf()

        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.ylabel('loss') 
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(outputPath, "loss_K{}.jpg".format(Fold)))
        plt.clf()

        if min(val_loss) < min_val_loss:
            best_model_filepath = filepath
            best_model = Fold
            min_val_loss = min(val_loss)
        Fold+=1

    # Use the test set in testing
    K_testAttrX = testSetAttrX
    K_testImagesX = testSetImagesX
    K_testY = testSetY
        
    #Load and evaluate the best model version
    print("[INFO] best model is {}".format(best_model_filepath))
    model = load_model(best_model_filepath)

    # make predictions on the testing data
    print("[INFO] computing accuracy...")
    preds = model.predict([K_testAttrX, K_testImagesX])
    
    mse = mean_squared_error(K_testY, preds)  # calculate the mean square error
    print("[INFO] Mean square error:", mse)
    mae = mean_absolute_error(K_testY, preds)  # calculate the mean square error
    print("[INFO] Mean absolute error:", mae)
    rmse = math.sqrt(mse)
    print("[INFO] Root mean square error:", rmse)
    # (eval_loss, eval_accuracy) = model.evaluate( 
    #     [K_testAttrX, K_testImagesX], K_testY, batch_size=32, verbose=1)
    # print('pred_value', preds.shape)
    # print('pred_value', preds.flatten().shape)
    # print('K_testY', K_testY.shape)
    diff = preds.flatten() - K_testY
    # print('diff', diff)
    diff_round = np.round(preds.flatten(), 1) - K_testY

    #preds_memo = np.round(preds.flatten(), 1)
    preds_memo = preds.flatten()
    #K_testY = K_testY.tolist()
    
    # print(preds_memo)
    # print(K_testY)

    #count = 0
    #for i in range(len(preds_memo)):
    #    if int(preds_memo[i]*10) == int(K_testY[i]*10):
    #        count +=1
    #DiffMatchACC = count/len(K_testY)
    #tempa = classify(preds_memo)
    #tempb = classify(K_testY)

    #countRound = 0
    #for i in range(len(preds_memo)):
    #    if preds_memo[i] == K_testY[i]:
    #        countRound +=1
    #DiffMatchACC_ROUND = countRound/len(K_testY)
    assert len(K_testY) == len(preds_memo) == len(df_test)
    
    resPred = classify(preds_memo)
    resGT = classify(K_testY)
    print(resPred)
    print(resGT)
    
    
    
    # Count acc of each object
    #objectAcc = {}
    #for key in testset:
    #    count = 0
    #    for item in testset[key]:
    #        if record[item]:
    #            count = count + 1
    #    acc = round(count / len(testset[key]) * 100, 2)
    #    objectAcc[key] = acc
    
    #print("[INFO] printing acc for each item:")
    #for key in objectAcc:
    #    print("%s has accuracy: %s" % (key, "{:.2f}%".format(objectAcc[key])))
    
    accNoDup = calAcc(df_test, resGT, resPred)
    print("[INFO] Accuracy No Dup: %s" % accNoDup)
    
    #DiffMatchACC_ROUND = calAcc(classify(preds_memo), classify(K_testY))
    
    #print("[INFO] DiffMatchACC: {:.2f}%".format(DiffMatchACC*100))
    #print("[INFO] Accuracy 1: {:.2f}%".format(DiffMatchACC_ROUND*100))
    #print("[INFO] Accuracy 2: {:.2f}%".format(np.sum(np.absolute(diff) < 0.15) / len(diff) * 100))

    # CeilMatchACC = np.count_nonzero(np.ceil(preds.flatten(), 1) == testY)/len(testY)
    # FloorMatchACC = np.count_nonzero(np.floor(preds.flatten(), 1) == testY)/len(testY)

    #percentDiff = (diff / K_testY) * 100
    #percentDiffRound = (diff_round / K_testY) * 100
    # percentCeilRound = (Ceil_round / testY) * 100
    # percentFloorRound = (Floor_round / testY) * 100

    #absPercentDiff = np.abs(percentDiff)
    #absPercentDiffRound = np.abs(percentDiffRound)
    # absPercentCeilRound = np.abs(percentCeilRound)
    # absPercentFloorRound = np.abs(percentFloorRound)
    # compute the mean and standard deviation of the absolute percentage
    # difference
    #mean = np.mean(absPercentDiff)
    #std = np.std(absPercentDiff)
    # print('mean', mean)
    # print('std', std)
    #meanRound = np.mean(absPercentDiffRound)
    #stdRound = np.std(absPercentDiffRound)
    # meanCeilRound = np.mean(absPercentCeilRound)
    # stdCeilRound = np.std(absPercentCeilRound)
    # meanFloorRound = np.mean(absPercentFloorRound)
    # stdFloorRound = np.std(absPercentFloorRound)
    # print("K_testAttrX[GT_label].mean():", K_testY)
    # print("K_testAttrX[GT_label].std():", K_testY)
    # finally, show some statistics on our model
    #locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    
    #print("[INFO] avg. GT_label: {}, std GT_label: {}".format(
    #    locale.currency(np.array(K_testY).mean(), grouping=True),
    #    locale.currency(np.array(K_testY).std(), grouping=True)))

    #print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
    #print("[INFO] meanRound: {:.2f}%, stdRound: {:.2f}%".format(meanRound, stdRound))

    # Rename the output path to auto-save the result, and label the acc in path
    newOutputPath = '%s_%smse_%smae_%srmse_%sacc_%sacc2_%slr_%sdc_K%d_dataset%d' % (outputPath, 
                                                                          "{:.3f}".format(mse), 
                                                                          "{:.3f}".format(mae), 
                                                                          "{:.3f}".format(rmse), 
                                                                          accNoDup,
                                                                          "{:.3f}".format(np.sum(np.absolute(diff) < 0.15) / len(diff) * 100),
                                                                          "{:.2e}".format(lr),
                                                                          "{:.2e}".format(dc),
                                                                          best_model,
                                                                          selector)
    count = 1
    while os.path.exists(newOutputPath):
        newOutputPath = newOutputPath + '_%d'%(count)
        count = count + 1
    #os.rename(outputPath, newOutputPath)
    shutil.move(outputPath, newOutputPath)

def classify(nparr):
    res = [None] * len(nparr)
    for i in range(len(nparr)):
        if nparr[i] <= 0.25:
            res[i] = 1
            continue
        if nparr[i] <= 0.5:
            res[i] = 2
            continue
        if nparr[i] <= 0.75:
            res[i] = 3
            continue
        if nparr[i] <= 1:
            res[i] = 4
            continue
    return np.array(res)

def calAcc(df_test, resGT, resPred):
    # Calculate acc for each object
    record = {}
    total_nodup = 0
    correct_nodup = 0
    for i in range(0, len(resPred)):
        pred = resPred[i]
        gt = resGT[i]
        img = df_test.iloc[i]['image_path']
        if img in record:
            continue
        #if gt == 0 or gt == 1:
        #    shutil.copy(os.path.join(imagePath, img), os.path.join(tsPath, img))
        total_nodup = total_nodup + 1
        if pred == gt:
            correct_nodup = correct_nodup + 1
            #if pred == 0 or pred == 1:
            #    shutil.copy(os.path.join(imagePath, img), os.path.join(bugPath, img))
            record[img] = True
            continue
        record[img] = False
    
    assert total_nodup == len(record)
    
    return "{:.2f}%".format(correct_nodup / total_nodup * 100)

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

def flatten(l):
    return [item for sublist in l for item in sublist]

if __name__ == "__main__":
    #start_time = time.time()
    main()
    #print("--- Execution time: %s seconds ---" % (time.time() - start_time))