'''
SVM systems for ALW, modified for English data from twitter and reddit, with crossvalidation and no test phase
original by Caselli et al: https://github.com/malvinanissim/germeval-rug
'''

glove_embeds_path = '../../embeddings/glove.twitter.27B.200d.txt'

#########################################################

import helperFunctions
import transformers
import argparse
import re
import statistics as stats
import stop_words
import json
import pickle
import gensim.models as gm
import os

import features
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import shuffle

from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nltk.tokenize import TweetTokenizer, word_tokenize, MWETokenizer
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


seed = 1337
np.random.seed(seed)


MWET = MWETokenizer([   ('<', 'url', '>'),
                        ('<', 'user', '>'),
                        ('<', 'smile', '>'),
                        ('<', 'lolface', '>'),
                        ('<', 'sadface', '>'),
                        ('<', 'neutralface', '>'),
                        ('<', 'heart', '>'),
                        ('<', 'number', '>'),
                        ('<', 'hashtag', '>'),
                        ('<', 'allcaps', '>'),
                        ('<', 'repeat', '>'),
                        ('<', 'elong', '>'),
                    ], separator='')

def ntlktokenizer(x):
    tokens = word_tokenize(x)           # tokenize
    tokens = MWET.tokenize(tokens)      # fix <url> and <user> etc.

    return ' '.join(tokens)

def main():

    parser = argparse.ArgumentParser(description='ALW')
    parser.add_argument('-src', type=str, default='Twitter', help='Twitter')
    parser.add_argument('-ftr', type=str, help='knn/vocab/mif/ngram/embeddings/embeddings+ngram')
    parser.add_argument('-cls', type=str, help='bilstm/none')
    parser.add_argument('-ds', type=str, help='WaseemHovy/standard/offenseval/cross')
    parser.add_argument('-mh5', type=str, help='modelh5')
    parser.add_argument('-tknzr', type=str, help='tknzr')
    parser.add_argument('-trnp', type=str, help='trainPath')
    parser.add_argument('-tstp', type=str, help='testPath')
    parser.add_argument('-pte', type=str, help='path_to_embs')
    parser.add_argument('-evlt', type=str, help='traintest/cv10/none')
    parser.add_argument('-cln', type=str, default='std', help='std/ruby')
    parser.add_argument('-eps', type=int, default=0, help='epochs')
    parser.add_argument('-ptc', type=int, default=0, help='patience')
    parser.add_argument('-vb', type=int, default=0, help='verbose')
    parser.add_argument('-bs', type=int, default=0, help='batch_size')
    parser.add_argument('-rev', type=helperFunctions.str2bool, default=helperFunctions.str2bool('False'), help='reverse 3vs1')
    parser.add_argument('-lstmTrn', type=helperFunctions.str2bool, default=helperFunctions.str2bool('False'), help='bilstm training True/False')
    parser.add_argument('-lstmOp', type=helperFunctions.str2bool, default=helperFunctions.str2bool('False'), help='bilstm output True/False')
    parser.add_argument('-lstmTd', type=helperFunctions.str2bool, default=helperFunctions.str2bool('False'), help='bilstm traindev True/False')
    parser.add_argument('-lstmCV', type=helperFunctions.str2bool, default=helperFunctions.str2bool('False'), help='bilstm cv True/False')
    parser.add_argument('-prb', type=helperFunctions.str2bool, default=helperFunctions.str2bool('False'), help='yguess_output/probabilities True/False')
    parser.add_argument('-cnct', type=helperFunctions.str2bool, default=helperFunctions.str2bool('False'), help='concat')
    parser.add_argument('-pool', type=str, default='max', help='pool max/average/concat')
    parser.add_argument('-nrange', type=str, default='None', help='3to3/3to6/none')
    parser.add_argument('-pte2', type=str, default='', help='path_to_embs2')
    args = parser.parse_args()

    source = args.src
    ftr = args.ftr
    cls = args.cls
    dataSet = args.ds
    modelh5 = args.mh5
    tknzr = args.tknzr
    trainPath = args.trnp
    testPath = args.tstp
    path_to_embs = args.pte
    evlt = args.evlt
    clean = args.cln
    lstmTraining = args.lstmTrn
    lstmOutput = args.lstmOp
    lstmTrainDev = args.lstmTd
    lstmCV = args.lstmCV
    lstmEps = args.eps
    lstmPtc = args.ptc
    reverse = args.rev
    vb = args.vb
    bs = args.bs
    prob = args.prb
    concat = args.cnct
    pool = args.pool
    nrange = args.nrange
    path_to_embs2 = args.pte2

    TASK = 'binary'
    #TASK = 'multi'
    
    if ftr == 'knn':
        keywords = ['woman', 'homosexual', 'black', 'gay', 'man', 'immigrant', 'immigrants', 'migrant', 'migrants', 'trans', 'gun', 'afroamerican', 'feminism', 'feminist', 'abortion', 'religion', 'god', 'trump', 'islam', 'muslim']
        helperFunctions.knn_embeddings(path_to_embs,keywords)
        exit()

    '''
    Preparing data
    '''

    print('Reading in ' + source + ' training data using ' + dataSet + 'dataset...')

#    IDsTrain, Xtrain, Ytrain, FNtrain, IDsTest, Xtest, Ytest, FNtest = helperFunctions.loaddata(dataSet, trainPath, testPath, cls, TASK, reverse)
    IDsTrain, Xtrain, Ytrain, IDsTest, Xtest, Ytest = helperFunctions.loaddata(dataSet, trainPath, testPath, cls, TASK, reverse)


    print('Done reading in data...')
    
    offensiveRatio = Ytrain.count('OFF')/len(Ytrain)
    nonOffensiveRatio = Ytrain.count('NOT')/len(Ytrain)

    # Minimal preprocessing / cleaning

    if clean == 'std':
        Xtrain = helperFunctions.clean_samples(Xtrain)
        if testPath != '':
            Xtest = helperFunctions.clean_samples(Xtest)
    if clean == 'ruby':
        Xtrain = helperFunctions.clean_samples_ruby(Xtrain)
        if testPath != '':
            Xtest = helperFunctions.clean_samples_ruby(Xtest)

    print(len(Xtrain), 'training samples after cleaning!')
    if Xtest:
        print(len(Xtest), 'testing samples after cleaning!')

    '''
    Preparing vectorizer and classifier
    '''

    # Vectorizing data / Extracting features
    print('Preparing tools (vectorizer, classifier) ...')

    # unweighted word uni and bigrams
    ### This gives the stop_words may be inconsistent warning
    if source == 'Twitter':
        tokenizer = TweetTokenizer().tokenize
    elif source == 'Reddit':
        # tokenizer = None
        tokenizer = ntlktokenizer
    else:
        tokenizer = None

    if ftr == 'vocab':
        count_word = CountVectorizer(stop_words=stop_words.get_stop_words('en'), tokenizer=tokenizer)
        if testPath != None:
            word_count_vector = count_word.fit_transform(Xtrain+Xtest)
        else:
            word_count_vector = count_word.fit_transform(Xtrain)
        vocab_data = list(count_word.vocabulary_.keys())
        embeddings, vocab = helperFunctions.load_embeddings(path_to_embs)
        print("vocab data: {}".format(str(len(vocab_data))))
        print("vocab embds: {}".format(str(len(vocab))))
        print("in vocab words: {}".format(str(len(set(vocab_data) & set(vocab)))))
        print("overlap: {} %".format(float(((len(set(vocab_data) & set(vocab))) / len(vocab_data)) * 100)))
        exit()

    if ftr == 'mif':
        print("Top 10 1-2 word n-grams features for Xtrain from:", dataSet)
        print(transformers.frequencyFilter.getKMostImportantToken(transformers.frequencyFilter(10, path_to_embs, 'word', 1, 2), Xtrain))
        print("Top 10 3-7 char n-grams features for Xtrain from:", dataSet)
        print(transformers.frequencyFilter.getKMostImportantToken(transformers.frequencyFilter(10, path_to_embs, 'char', 3, 7), Xtrain))
        exit()

    if ftr == 'ngram':
        count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('en'), tokenizer=tokenizer)
        count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))
        vectorizer = FeatureUnion([ ('word', count_word),
                                    ('char', count_char)])

    elif ftr =='custom':
        vectorizer = FeatureUnion([
                               # ('tweet_length', features.TweetLength()),
                               ('file_name', features.FileName(FNtrain, FNtest))
                               ])

    elif ftr == 'embeddings':
        # print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
        embeddings, vocab = helperFunctions.load_embeddings(path_to_embs)
        if path_to_embs2 != '':
            embeddings2, vocab2 = helperFunctions.load_embeddings(path_to_embs2)
        else:
            embeddings2 = {}
        glove_embeds = {}
        if concat:
            if path_to_embs == glove_embeds_path:
                glove_embeds = embeddings
            else:
                glove_embeds, glove_vocab = helperFunctions.load_embeddings(glove_embeds_path)
        print('Done')
        vectorizer = features.Embeddings(embeddings, glove_embeds, nrange, path_to_embs, pool='max')
#        vectorizer = FeatureUnion([
#                                   ('word_embeds', features.Embeddings(embeddings, glove_embeds, pool=pool)),
#                                   # ('tweet_length', features.TweetLength()),
#                                   ('file_name', features.FileName(FNtrain, FNtest))
#                                   ])

    elif ftr == 'embeddings+ngram':
        count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('en'), tokenizer=tokenizer)
        count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))
        # path_to_embs = 'embeddings/model_reset_random.bin'
        print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
        embeddings, vocab = helperFunctions.load_embeddings(path_to_embs)
        glove_embeds = {}
        if concat:
            if path_to_embs == glove_embeds_path:
                glove_embeds = embeddings
            else:
                glove_embeds, glove_vocab = helperFunctions.load_embeddings(glove_embeds_path)
        print('Done')
        vectorizer = FeatureUnion([ ('word', count_word),
                                    ('char', count_char),
                                    ('word_embeds', features.Embeddings(embeddings, glove_embeds, nrange, path_to_embs, pool='max'))])

    if cls == 'bilstm':
        from BiLSTM import biLSTM
        if lstmTrainDev:
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain, Ytrain, test_size=0.33, random_state=seed)
        print('Train labels', set(Ytrain), len(Ytrain))
        print('Test labels', set(Ytest), len(Ytest))
        print(cls)
        Ytest, Yguess = biLSTM(Xtrain, Ytrain, Xtest, Ytest, lstmTraining, lstmOutput, embeddings, tknzr, modelh5, lstmCV, lstmEps, lstmPtc, dataSet, vb, bs, prob, path_to_embs, nrange, path_to_embs2, embeddings2)


    # Set up SVM classifier with unbalanced class weights
    if TASK == 'binary' and cls != 'bilstm':
        # cl_weights_binary = None
        cl_weights_binary = {'NOT':1/nonOffensiveRatio, 'OFF':1/offensiveRatio}
        # clf = LinearSVC(class_weight=cl_weights_binary, random_state=1337)
        clf = SVC(kernel='linear', probability=True, class_weight=cl_weights_binary, random_state=seed)
    elif TASK == 'multi':
        # cl_weights_multi = None
        cl_weights_multi = {'OTHER':0.5,
                            'ABUSE':3,
                            'INSULT':3,
                            'PROFANITY':4}
        # clf = LinearSVC(class_weight=cl_weights_multi, random_state=1337)
        clf = SVC(kernel='linear', class_weight=cl_weights_multi, probability=True, random_state=seed)

    if cls != 'bilstm':
        classifier = Pipeline([
                                ('vectorize', vectorizer),
                                ('classify', clf)])



    '''
    Actual training and predicting:
    '''

    if evlt == 'cv10':
        print('10-fold cross validation results:')
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        Xtrain = np.array(Xtrain)
        Ytrain = np.array(Ytrain)

        accuracy = 0
        precision = 0
        recall = 0
        fscore = 0
        
        NOT_prec = 0
        NOT_rec = 0
        OFF_prec = 0
        OFF_rec = 0
        all_f1 = 0
        for train_index, test_index in kfold.split(Xtrain, Ytrain):
            X_train, X_test = Xtrain[train_index], Xtrain[test_index]
            Y_train, Y_test = Ytrain[train_index], Ytrain[test_index]

            classifier.fit(X_train,Y_train)
            if ftr == 'ngram':
                helperFunctions.print_top10(vectorizer, clf, clf.classes_)
            Yguess = classifier.predict(X_test)

            accuracy += accuracy_score(Y_test, Yguess)
            precision += precision_score(Y_test, Yguess, average='macro')
            recall += recall_score(Y_test, Yguess, average='macro')
            fscore += f1_score(Y_test, Yguess, average='macro')
            report = classification_report(Y_test, Yguess)
            
            list_Xtrain = X_train.tolist()
            list_Xtest = X_test.tolist()
            list_Ytrain = Y_train.tolist()
            list_Ytest = Y_test.tolist()
            reportlist = report.strip().split('\n')
            newlist = []
            for i in reportlist:
                if i != '':
                    i = i.split()[-4:-1]
                    newlist.append(i)
            newlist = newlist[1:-1]
            NOT_prec += float(newlist[0][0])
            NOT_rec += float(newlist[0][1])
            all_f1 += (float(newlist[0][2]) + float(newlist[1][2])) / 2
            OFF_prec += float(newlist[1][0])
            OFF_rec += float(newlist[1][1])
        
        print("NOTprec  {}  NOTrec  {}".format(NOT_prec/10,NOT_rec/10))
        print("OFFprec  {}  OFFrec  {}".format(OFF_prec/10,OFF_rec/10))
        print("ALLf1  {}".format(all_f1/10))
        
        print('accuracy_score: {}'.format(accuracy / 10))
        print('precision_score: {}'.format(precision / 10))
        print('recall_score: {}'.format(recall / 10))
        print('f1_score: {}'.format(fscore / 10))

        # results = (cross_validate(classifier, Xtrain, Ytrain,cv=10, verbose=1))
        # # print(results)
        # print(sum(results['test_score']) / 10)

        print('\n\nDone, used the following parameters:')
        print('train: {}'.format(trainPath))
        if ftr != 'ngram':
            print('embed: {}'.format(path_to_embs))
        print('feats: {}'.format(ftr))
        print('prepr: {}'.format(clean))
        print('sourc: {} - datas: {}'.format(source, dataSet))
    elif evlt == 'traintest':
        if cls != 'bilstm':
            classifier.fit(Xtrain,Ytrain)
            if ftr == 'ngram':
                helperFunctions.print_top10(vectorizer, clf, clf.classes_)
            Yguess = classifier.predict(Xtest)
#            print(classifier.show_most_informative_features(20))
        print('train test results:')
        print(accuracy_score(Ytest, Yguess))
        print(precision_recall_fscore_support(Ytest, Yguess, average='macro'))
        print(classification_report(Ytest, Yguess))
        print('\n\nDone, used the following parameters:')
        print('train: {}'.format(trainPath))
        print('tests: {}'.format(testPath))
        if ftr != 'ngram':
            print('embed: {}'.format(path_to_embs))
        print('feats: {}'.format(ftr))
        print('prepr: {}'.format(clean))
        print('sourc: {} - datas: {}'.format(source, dataSet))

    if prob:
        with open('probas_SVC_' + dataSet + '_' + trainPath.split("/")[-1] + '_' + ftr + '_' + path_to_embs.split("/")[-1] + '_concat=' + str(concat) + '.txt', 'w+') as yguess_output:
            for i in classifier.predict_proba(Xtest):
                yguess_output.write('%s\n' % i[1])
        # print(classifier.predict_proba(Xtest))
        # print(Yguess)

if __name__ == '__main__':
    main()
