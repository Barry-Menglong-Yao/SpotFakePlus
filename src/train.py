# libraries
import torch
import numpy as np
from keras import optimizers
import matplotlib.pyplot as plt
from keras import initializers
from keras.utils import np_utils
from keras import regularizers
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import concatenate
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization, PReLU

# for running on multiple GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import threading
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def json_to_dict(trainData):
    data_dict={}
    for one_data in trainData:
        data_dict[one_data["id"]]=one_data
    return data_dict 

import pickle

    


# match code
# Check whether Image exists, then get corresponding Text Embeddings, and finally append to respective lists
def gen_formatted_data(train_vgg_poli,trainEmbeddings,trainData):
    train_text = [] # text embeddings
    train_label = [] # labels
    test_text = [] # text embeddings
    test_label = []
    train_image = [] # image embeddings
    test_image = []

    # Train Image IDs
    # Test Image IDs
    trainImageNames = [] # names of the images i.e name.jpg
    trainTextNames = []  # train articles
    testTextNames = []   # test articles
    testImageNames = []  # names of the images in the test folder
    for  img_name in train_vgg_poli:
        prefix=img_name[:17]
        ids=prefix.split("-")
        article_id= int(ids[0]) 
        one_data=trainData[article_id]
    
    
        text_embed=trainEmbeddings[article_id]
        image_embed=train_vgg_poli[img_name]
            
        trainImageNames.append(img_name)
        trainTextNames.append(article_id)
        train_text.append(text_embed)
        train_image.append(image_embed)
        train_label.append(one_data["label"])
    return train_text,train_image,train_label,trainImageNames,trainTextNames
    
    
def train():
    with open('data/feature/finalTestEmbeddings2.pkl', 'rb') as f:
        trainEmbeddings = pickle.load(f)
    with open('data/feature/finalTestEmbeddings2.pkl', 'rb') as f:
        testEmbeddings = pickle.load(f)
        
    # dictionary---> Article text: ( label, articleURL, ImageId )
    import json
    with open('data/train/news_article.json', 'r') as f:
        trainData = json_to_dict(json.load(f))
    with open('data/train/news_article.json', 'r') as f:
        testData = json_to_dict(json.load(f))
        
        

    for i in trainEmbeddings:
        trainEmbeddings[i] = [torch.mean(j[0], axis=1) for j in trainEmbeddings[i]]#trainEmbeddings[i][0][0] ([1, 54, 768])
            
            
    for i in testEmbeddings:
        testEmbeddings[i] = [torch.mean(j[0], axis=1) for j in testEmbeddings[i]]
        
    # for i in testEmbeddings:
    #     temp = testEmbeddings[i]
    #     break

    # padding
    # if a paragraph has more than 50 sentences then crop, if less than 50 then pad.

    for i in trainEmbeddings:
        if len(trainEmbeddings[i]) >=50:
            trainEmbeddings[i] = trainEmbeddings[i][0:50]
        else:
            deficit = 50 - len(trainEmbeddings[i])
            for j in range(deficit):
                trainEmbeddings[i].append(torch.zeros((1,768), dtype=torch.float32, device='cuda:0'))
        temp = torch.empty(50,768, dtype=torch.float32, device='cuda:0')
        for j in range(len(trainEmbeddings[i])):
            temp[j][:] = trainEmbeddings[i][j]
        trainEmbeddings[i] = temp
        
    for i in testEmbeddings:
        if len(testEmbeddings[i]) >=50:
            testEmbeddings[i] = testEmbeddings[i][0:50]
        else:
            deficit = 50 - len(testEmbeddings[i])
            for j in range(deficit):
                testEmbeddings[i].append(torch.zeros((1,768), dtype=torch.float32, device='cuda:0'))
        temp = torch.empty(50,768, dtype=torch.float32, device='cuda:0')
        for j in range(len(testEmbeddings[i])):
            temp[j][:] = testEmbeddings[i][j]
        testEmbeddings[i] = temp
        
    with open('/home/menglong/workspace/code/multimodal/mocheg/SpotFakePlus/data/feature/image.pickle', 'rb') as f:
        train_vgg_poli = pickle.load(f)
    with open('/home/menglong/workspace/code/multimodal/mocheg/SpotFakePlus/data/feature/image.pickle', 'rb') as f:
        test_vgg_poli = pickle.load(f)
    train_text,train_image,train_label,trainImageNames,trainTextNames=gen_formatted_data(train_vgg_poli,trainEmbeddings,trainData)
    test_text,test_image,test_label,testImageNames,testTextNames=gen_formatted_data(test_vgg_poli,testEmbeddings,testData)
    import pandas as pd
    df=pd.DataFrame()
    df['article']=testTextNames
    df['image']=testImageNames
    df['label']=test_label
    df.to_csv('data/csv/politifact_test.csv', sep='\t')
    train_label = to_categorical(train_label)
    test_label = to_categorical(test_label)
    # df['article']=trainTextNames
    # df['image']=trainImageNames
    # df['label']=train_label
    train_text=[torch.Tensor.numpy(i.cpu()) for i in train_text]
    test_text=[torch.Tensor.numpy(i.cpu()) for i in test_text]
    train_text_matrix = np.ndarray(shape=(len(train_text), 50,768))
    counter = 0
    for i in train_text:
        train_text_matrix[counter][:][:] = i
        counter += 1
    test_text_matrix = np.ndarray(shape=(len(test_text), 50,768))
    counter = 0
    for i in test_text:
        test_text_matrix[counter][:][:] = i
        counter += 1
    train_len=len(train_image)
    test_len=len(test_image)
    train_image_matrix = np.ndarray(shape=(len(train_image), 4096,1))
    counter = 0
    for i in train_image:
        train_image_matrix[counter][:][:] = i.reshape(4096,1)
        counter += 1
    test_image_matrix = np.ndarray(shape=(len(test_image), 4096,1))
    counter = 0
    for i in test_image:
        test_image_matrix[counter][:][:] = i.reshape(4096,1)
        counter += 1
    train_image_matrix = train_image_matrix.reshape(train_len,4096)
    test_image_matrix = test_image_matrix.reshape(test_len,4096)
    model=gen_model()
    checkpoint = ModelCheckpoint(filepath='checkpoints/dense_MM_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit([train_text_matrix, train_image_matrix],train_label,validation_data=([test_text_matrix,test_image_matrix],test_label),batch_size =32,epochs =100,callbacks=callbacks_list)
    with open('output/XL_poli_history.json', 'w') as f:
        json.dump(str(history.history), f)
def show():
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
def gen_model():
    input_text = Input(shape=(50,768))
    text_flat = Flatten()(input_text)
    dense_text = Dense(1000,activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=0))(text_flat)
    #dense_text = Dropout(0.4)(dense_text)
    dense_text = Dense(500,activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=0))(dense_text)
    #dense_text = Dropout(0.4)(dense_text)
    dense_text = Dense(100,activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=0))(dense_text)
    dense_text = BatchNormalization()(dense_text)
    dense_text_drop = Dropout(0.4)(dense_text)

    input_image = Input(shape=(4096,))
    dense_image = Dense(2000,activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=0))(input_image)
    #dense_image = Dropout(0.4)(dense_image)
    dense_image = Dense(1000, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=0))(dense_image)
    #dense_image = Dropout(0.4)(dense_image)
    dense_image = Dense(100,activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=0))(dense_image)
    dense_image = BatchNormalization()(dense_image)
    dense_image_drop = Dropout(0.4)(dense_image)

    concat = concatenate([dense_text_drop,dense_image_drop])

    inter1_dense = Dense(200,activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=0))(concat)
    inter1_dense = Dense(100,activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=0))(inter1_dense)
    final_dense = Dense(50,activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=0))(inter1_dense)
    final_dropout = Dropout(0.4)(final_dense)
    output = Dense(3, activation='softmax')(final_dropout)

    model = Model(inputs=[input_text,input_image], outputs=output)
    adam = optimizers.Adam(lr=1e-4)
    #adagrad = optimizers.Adagrad(lr=1e-4)
    #adamax = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999)
    

    #sgd = optimizers.SGD(lr=1e-4, clipnorm=1.)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
train()