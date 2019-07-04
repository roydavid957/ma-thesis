"""
file containing useful, small functions for SVM_original.py and BiLSTM.py
"""

import re
import csv
import time
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from random import shuffle
import pandas as pd
import os
from nltk.tokenize import TweetTokenizer
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText as FT_gensim
import io
import os
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

def loaddata(dataSet, trainPath, testPath, cls, TASK):
    """ loads data set """
    IDsTrain = []
    Xtrain = []
    Ytrain = []
    IDsTest = []
    Xtest = []
    Ytest = []
    if dataSet == 'WaseemHovy':
        if TASK == 'binary':
            IDsTrain,Xtrain,Ytrain = read_corpus_WaseemHovy(trainPath,cls)
        else:
            IDsTrain,Xtrain,Ytrain = read_corpus_WaseemHovy(trainPath,cls)

    elif dataSet == 'standard':
        IDsTrain,Xtrain,Ytrain = read_corpus(trainPath,cls)
        # Also add SemEval dev-data
        IDsStandard_test,Xstandard_test,Ystandard_test = read_corpus('/'.join(trainPath.split('/')[:-1])+'/dev_en.tsv',cls)
        for id,x,y in zip(IDsStandard_test,Xstandard_test,Ystandard_test):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        if testPath.split('/')[-1] == 'test_en.tsv':
            IDsTest,Xtest,Ytest = read_corpus(testPath,cls)

    elif dataSet == 'offenseval':
        IDsTrain,Xtrain,Ytrain = read_corpus_offensevalTRAIN(trainPath,cls)
        if testPath.split('/')[-1] == 'testset-levela.tsv':
            IDsTest,Xtest,Ytest = read_corpus_offensevalTEST(testPath,cls)

    elif dataSet == 'cross':
        ''' load train data '''

        if trainPath.split('/')[-1] == 'Full_Tweets_June2016_Dataset.csv':
            IDsTrain,Xtrain,Ytrain = read_corpus_WaseemHovy(trainPath,cls)

        elif trainPath.split('/')[-1] == 'train_en.tsv':
            IDsTrain,Xtrain,Ytrain = read_corpus(trainPath,cls)
            # Also add SemEval dev-data
            IDsStandard_test,Xstandard_test,Ystandard_test = read_corpus('/'.join(trainPath.split('/')[:-1])+'/dev_en.tsv',cls)
            for id,x,y in zip(IDsStandard_test,Xstandard_test,Ystandard_test):
                IDsTrain.append(id)
                Xtrain.append(x)
                Ytrain.append(y)

        elif trainPath.split('/')[-1] == 'olid-training-v1.0.tsv':
            IDsTrain,Xtrain,Ytrain = read_corpus_offensevalTRAIN(trainPath,cls)

        ''' load test data '''

        if testPath.split('/')[-1] == 'Full_Tweets_June2016_Dataset.csv':
            IDsTest,Xtest,Ytest = read_corpus_WaseemHovy(testPath,cls)

        elif testPath.split('/')[-1] == 'test_en.tsv':
            IDsTest,Xtest,Ytest = read_corpus(testPath,cls)
        
        elif testPath.split('/')[-1] == 'testset-levela.tsv':
            IDsTest,Xtest,Ytest = read_corpus_offensevalTEST(testPath,cls)

        elif testPath.split('/')[-1] == 'offensive.csv':
            IDsTest,Xtest,Ytest = read_corpus_stackoverflow(testPath,cls)

    return IDsTrain, Xtrain, Ytrain, IDsTest, Xtest, Ytest

def read_corpus_WaseemHovy(corpus_file,cls):
    '''Reading in data from corpus file'''
    print('Reading WaseemHovy data...')
    ids = []
    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='ISO-8859-1') as fi:
        for line in fi:
            data = line.strip().strip('\n').split(',')
            ids.append(data[0])
            if len(data)<3 or data[2] == 'NA':
                continue
            if len(data)>3:
                if data[1] == 'racism' or data[1] == 'sexism' or data[1] == 'none':
                    tweets.append("".join(data[2:len(data)]))
                else:
                    tweets.append("".join(data[1:len(data) - 2]))
            elif data[1] == 'racism' or data[1] == 'sexism' or data[1] == 'none':
                tweets.append(data[2])
            else:
                tweets.append(data[1])
            if cls == 'bilstm':
                if data[1] == 'none':
                    labels.append(0)
                elif data[len(data)-1] == 'none':
                    labels.append(0)
                else:
                    labels.append(1)
            else:
                if data[1] == 'none':
                    labels.append('NOT')
                elif data[len(data)-1] == 'none':
                    labels.append('NOT')
                else:
                    labels.append('OFF')
    mapIndexPosition = list(zip(ids, tweets, labels))
    shuffle(mapIndexPosition)
    ids, tweets, labels = zip(*mapIndexPosition)

    print("read " + str(len(tweets)) + " tweets.")
    return ids, tweets, labels

def read_corpus_offensevalTRAIN(corpus_file, cls, binary=True):
    '''Reading in data from corpus file'''
    print('Reading OffensEvalTRAIN data...')
    ids = []
    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # making sure no missing labels
            if len(data) != 5:
                raise IndexError('Missing data for tweet "%s"' % data[0])

            ids.append(data[0])
            tweets.append(data[1])
            if cls == 'bilstm':
                if data[2] == 'OFF':
                    labels.append(1)
                elif data[2] == 'NOT':
                    labels.append(0)
            else:
                if data[2] == 'OFF':
                    labels.append('OFF')
                elif data[2] == 'NOT':
                    labels.append('NOT')

    print("read " + str(len(tweets[1:])) + " tweets.")
    return ids[1:], tweets[1:], labels

def read_corpus_offensevalTEST(corpus_file, cls, binary=True):
    '''Reading in data from corpus file'''
    print('Reading OffensEvalTEST data...')
    ids = []
    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # making sure no missing labels
            if len(data) != 2:
                raise IndexError('Missing data for tweet "%s"' % data[0])

            ids.append(data[0])
            tweets.append(data[1])
    with open(('/').join(corpus_file.split('/')[:-1])+'/labels-levela.csv', 'r', encoding='ISO-8859-1') as fi2:
        for line in fi2:
            data = line.strip().split(',')
            if len(data)<2:
                continue
            if cls == 'bilstm':
                if data[1] == 'NOT':
                    labels.append(0)
                elif data[1] == 'OFF':
                    labels.append(1)
            else:
                if data[1] == 'NOT':
                    labels.append('NOT')
                elif data[1] == 'OFF':
                    labels.append('OFF')

    print("read " + str(len(tweets[1:])) + " tweets.")
    return ids[1:], tweets[1:], labels

def read_corpus(corpus_file, cls, binary=True):
    '''Reading in data from corpus file'''
    print('Reading HatEval data...')
    ids = []
    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # making sure no missing labels
            if len(data) != 5:
                raise IndexError('Missing data for tweet "%s"' % data[0])

            ids.append(data[0])
            tweets.append(data[1])
            if cls == 'bilstm':
                if data[2] == '1':
                    labels.append(1)
                elif data[2] == '0':
                    labels.append(0)
            else:
                if data[2] == '1':
                    labels.append('OFF')
                elif data[2] == '0':
                    labels.append('NOT')

    if corpus_file.split('/')[-1] == 'test_en.tsv':
        print("read " + str(len(tweets[1:])) + " tweets.")
        return ids, tweets, labels
    else:
        print("read " + str(len(tweets[1:])) + " tweets.")
        return ids[1:], tweets[1:], labels


def read_corpus_stackoverflow(corpus_file,cls):
    '''Reading in data from corpus file'''
    print('Reading StackOverflow data...')

    ids = []
    tweets = []
    labels = []
    count = 0
    with open(corpus_file) as csvfile:
        next(csvfile)
        readCSV = csv.reader(csvfile, delimiter=',')
        for line in readCSV:
            text = line[1]
            flag = line[2]

            if cls == 'bilstm':
                labels.append(1)
                tweets.append(text)
                ids.append(count)
                count += 1
            else:
                labels.append('OFF')
                tweets.append(text)
                ids.append(count)
                count += 1

    print("read " + str(len(tweets)) + " tweets.")
    return ids, tweets, labels


def load_embeddings(embedding_file):
    '''
    loading embeddings from file
    input: embeddings
    output: embeddings in a dict-like structure available for look-up, vocab covered by the embeddings as a set
    '''

    print('Using embeddings: ', embedding_file)
    if embedding_file.endswith('.txt') or embedding_file.endswith('.vec'):
        w2v = {}
        vocab = []
        try:
            f = open(embedding_file,'r')
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    float(values[1])
                except ValueError:
                    continue
                coefs = np.asarray(values[1:], dtype='float')
                w2v[word] = coefs
                vocab.append(word)
        except UnicodeDecodeError:
            f = open(embedding_file,'rb')
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    float(values[1])
                except ValueError:
                    continue
                coefs = np.asarray(values[1:], dtype='float')
                w2v[word] = coefs
                vocab.append(word)
        f.close()

    else:
        
        try:
            w2v = FT_gensim.load(embedding_file)
            vocab = w2v.wv.vocab.keys()
            print('using FastText gensim...')
        except:
            try:
                w2v = FT_gensim.load_fasttext_format(embedding_file)
                vocab = w2v.wv.vocab.keys()
                print('using gensim Facebook FastText...')
            except:
                w2v, vocab = load_vectors(embedding_file)
                print('using Facebook fastText')

    try:
        print("Done.",len(w2v)," words loaded!")
    except:
        pass

    return w2v, vocab

def knn_embeddings(pte,keywords):
    """ nearest neighbors for embeddings based on given keywords """
    print('loading KeyedVectors...')
    if pte.endswith('.vec'):
        m = gensim.models.KeyedVectors.load_word2vec_format(pte)
    elif pte.endswith('.txt'):
        print('creating w2v file: {}'.format(pte.split('.txt')[0]+'_w2v.txt'))
        glove2word2vec(pte, pte.split('.txt')[0]+'_w2v.txt')
        m = gensim.models.KeyedVectors.load_word2vec_format(pte.split('.txt')[0]+'_w2v.txt')

    for k in keywords:
        print('\n-- nearest neighbours of {} :'.format(k))
        try:
            neighbours = m.most_similar(k)
            print(neighbours)
        except KeyError:
            print("word '%s' not in vocabulary" % k)

    try:
        os.remove(pte.split('.txt')[0]+'_w2v.txt')
    except:
        pass

def load_vectors(fname):
    
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    vcb = []
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        vcb.append(tokens[0])
    return data, vcb

def url_replacer(text):
    url = [r'http\S+', r'\S+https', r'\S+http']
    for item in url:
        text = re.sub(item, ' <url> ', text)
    return text

def user_replacer(text):
    url = [r'@\S+', r'\S+@']
    for item in url:
        text = re.sub(item, ' <user> ', text)
    return text

def hashtag_replacer(text):
    url = [r'#\S+', r'\S+#']
    for item in url:
        text = re.sub(item, ' <hashtag> ', text)
    return text

def clean_samples(samples):
    '''
    Simple cleaning: removing URLs, line breaks, abstracting away from user names etc.
    '''

    uuh_list = ['<user>','<url>','<hashtag>']
    new_samples = []
    for tw in samples:
        
        tw = tw.lower()
        tw = url_replacer(tw)
        tw = user_replacer(tw)
        tw = hashtag_replacer(tw)
        tw = tw.replace('\n',' ').split()

        tw = ' '.join(tw)
        tw = ' '.join(TweetTokenizer().tokenize(tw))

        new_samples.append(tw)

    return new_samples

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
