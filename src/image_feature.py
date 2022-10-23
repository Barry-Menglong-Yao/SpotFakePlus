


import numpy as np
import torch
import pandas as pd
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import Sequential
from nltk.tokenize import sent_tokenize
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pickle as pickle
import os 
import keras
import tensorflow as tf
from keras import layers,models
import keras
import os
import gc

from tqdm import tqdm 
def proc(img):
    
    from keras.preprocessing.image import load_img
    # load an image from file
    image = load_img(img, target_size=(224, 224))


    from keras.preprocessing.image import img_to_array
    # convert the image pixels to a numpy array
    image = img_to_array(image)


    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    from keras.applications.vgg16 import preprocess_input
    # prepare the image for the VGG model
    image = preprocess_input(image)
    
    return image

def test_gen_image():
    images=[]
    im=[]
    im.append(proc("/home/menglong/workspace/code/multimodal/mocheg/SpotFakePlus/data/images/73.jpg"))
    images.append(im)
    # print(images)
    
def gen_image_feature(data_folder,mode):
    img_folder=os.path.join(data_folder,mode,"images")
    

    #installing vgg19
    vgg16_model = keras.applications.vgg16.VGG16(weights='imagenet')
    vgg16_model.summary()
    # type(vgg19_model) #not Sequential model 


    from keras.models import Sequential
    model = Sequential()
    for layer in vgg16_model.layers[:-2]:
        model.add(layer)
    model.summary() 
    # from keras import optimizers
    # adam = optimizers.Adam(lr = 1e-4)
    # model.compile(adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # last_layer = tf.keras.models.Model(inputs=vgg16_model.inputs, outputs=vgg16_model.layers[-2].output)

    os_error =[]
    em ={}
    t=0
    ar = os.listdir(img_folder)
    n=500
    final = ar
    for img_name in tqdm(final):
        prefix=img_name[:18]
        ids=prefix.split("-")
        claim_id= int(ids[0])  
        relevant_document_id= ids[1] 
        if relevant_document_id=="proof":
            # print(i)
            try: 
                em[img_name] = model.predict(proc(os.path.join(img_folder,img_name)))
        
            except Exception as e:
                print(f"error {e}") 
                os_error.append(i)
    print(len(em))
    with open(f'data/feature/image_embed_{mode}.pickle', 'wb') as handle:
        pickle.dump(em, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"error image: {os_error}")
    #     gc.collect()
    
    
import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,help=" ",default="val") #politifact_v3,mode3_latest_v5
    
    args = parser.parse_args()
    return args


  
  
  
  
if __name__ == '__main__':
    args = parser_args()
    
        
    gen_image_feature('/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2',args.mode )        #/home/menglong/workspace/code/multimodal/mocheg/SpotFakePlus/data/images
