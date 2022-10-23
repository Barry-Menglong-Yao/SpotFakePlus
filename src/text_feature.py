from nltk.corpus import stopwords
from string import punctuation,digits
import pandas as pd
import json
import numpy as np
import re
#import word2vecReader as godin_embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm_notebook as tqdm
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
import torch
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import keras.backend as K
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error,r2_score
#from aspect_specific_prob import get_normalized_sentence_relation_vector
from math import sqrt
#from gensim.models import KeyedVectors
from sklearn.model_selection import KFold
import nltk
# nltk.download('stopwords')
import json
from embedding_as_service.text.encode import Encoder

from transformers import XLNetTokenizer, XLNetModel
# from pytorch_transformers import XLNetModel, XLNetTokenizer
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import threading
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from nltk.tokenize import sent_tokenize

# Generate embeddings for a paragraph
def torchEmbeddings(data,tokenizer,model):
    embed = []
    sentences= sent_tokenize(data)
    for sent in sentences:
        input_ids = torch.tensor(tokenizer.encode(sent)).unsqueeze(0)
        input_ids = input_ids.to('cuda')
        tempEmbedding = model(input_ids)
        #temp=en.encode([sent],pooling='reduce_mean')
        embed.append(tempEmbedding)
    return embed

def gen_text_embedding(data_path):
    

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetModel.from_pretrained('xlnet-base-cased')
    model.to('cuda')
    # Load Training and Testing Data 
    with open(data_path, 'r') as f:
        testData = json.load(f)
        embeddings = {}
        lengths = []    
        queryList =  testData 
        exceptions = []
        from tqdm  import tqdm
        with torch.no_grad():
            embeddings = {}
            for i in tqdm(queryList):
                try:
                    embeddings[i["id"]] = torchEmbeddings(i["text"],tokenizer,model)
                except Exception as e:
                    print(e)
                    exceptions.append(i)
            with open('data/feature/finalTestEmbeddings2.pkl', 'wb') as f:
                pickle.dump(embeddings, f)
        print(embeddings)
                    
        
if __name__ == "__main__":
    
        
    gen_text_embedding('/home/menglong/workspace/code/multimodal/mocheg/SpotFakePlus/data/train/news_article.json')        
