#!/usr/bin/python3
# Leon Graumans: original
# edited by: Roy David
#
# script for parsing scraped Tweets

import os
import re
import gzip
import json
import pickle
import pprint
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
import string
import sys
from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath
import numpy as np
import argparse
seed = 1337
np.random.seed(seed)

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

def check_polarisation(text, hashtags):
    r = re.compile(r'([#])(\w+)\b')
    items = r.findall(text)
    for item in items:
        if item[1].lower() in hashtags:
            return True

def check_flags(text, hashtags):
    try:
        for item in hashtags:
            if item in text:
                return True
    except:
        pass

def punct_replacer(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text

def norm(corpus_file):
    """ normalize tweet messages and write them to file """
    uuh_list = ['<user>','<url>','<hashtag>']
    save_file = corpus_file.split('.txt.gz')[0]+'_'+'TWEETSpl.txt'
    print('writing to {}'.format(save_file))
    f2 = open(save_file,'w')

    with gzip.open(corpus_file,'rt') as f:
        for line in f:
            json_line = json.loads(line.strip('\n'))
            text = json_line['text']
            text = text.lower()
            text = url_replacer(text)
            text = user_replacer(text)
            text = hashtag_replacer(text)
            text = text.replace('\n',' ').split()
            try:
                while text[-1] in uuh_list:
                    del text[-1]
            except:
                pass
            text = ' '.join(text)
#            text = punct_replacer(text)
#            yield list(TweetTokenizer().tokenize(text))

            text = ' '.join(TweetTokenizer().tokenize(text))
            if text != '':
                f2.write(text+'\n')
    f2.close()

def scraper(dir_path):
    """ filter tweets collected from karora based on requirements """
    keywords = ['trump','Hillary','MAGA','realDonaldTrump','buildthewall','MakeAmericaGreatAgain','TrumpTrain','Trump2016','AmericaFirst','TrumpForPrison','TrumpLies','TrumpsArmy','Trumpstrong','ChristiansForTrump','WomenForTrump','DrainTheSwamp','HillaryClinton','pizzagate','ClintonEmails','ClintonCrimeFamily','HillaryForPrison','Hillary2016','CriminalClinton','NeverHillary','DonaldTrump','WomenWhoVoteTrump','dumptrump','VoteTrump','CrookedHillary','Women4Trump','Blacks4Trump','fucktrump']
#    keywords = ['🇨🇦','🇬🇧','🇺🇸','🏳️‍🌈']
    hashtags = []
    for word in keywords:
        hashtags.append(word.lower())

    p_list = []        # polarised
    wordcount_p = 0
    c = 0

    directory = dir_path

    p_file = gzip.open('../../../data/s2764989/roy/tweets/txt_twitter_p2_'+hashtags[0]+'_'+hashtags[1]+'_'+hashtags[2]+'_'+directory.split('/')[-1]+'.txt.gz', 'wt')
#    feedback = open('txt_twitter_feedback2_'+hashtags[0]+'_'+hashtags[1]+'_'+hashtags[2]+'_'+directory.split('/')[-1]+'.txt', 'w')

    for root, dirs, files in os.walk(directory):
        print('Total     files: {:,}\n'.format(len(files)))
        for file in files:
            c += 1
            if file.endswith('.out.gz'):
                print('Reading ', os.path.join(directory, file), c, '...')
                with gzip.open(os.path.join(directory, file), 'r') as f:
                    for line in f:
                        json_line = json.loads(line)
                        
                        text = json_line['text']
#                        text = json_line['user']['name']
#                        text = json_line['user']['description']

                        if check_polarisation(text, hashtags):
#                        if check_flags(text, keywords):
                            # p_list.append(text)
                            json.dump(json_line,p_file)
                            p_file.write('\n')
                            wordcount_p += len(word_tokenize(text))

                file_wc = str(file) + ' - ' + str(wordcount_p)
#                feedback.write(file_wc+'\n')
                print('Total     polarised: {:,}\n'.format(wordcount_p))

    p_file.close()
#    feedback.close()

def read_data(corpus_file):
    """ count the distribution of keywords """
    keywords = ['trump','Hillary','MAGA','realDonaldTrump','buildthewall','MakeAmericaGreatAgain','TrumpTrain','Trump2016','AmericaFirst','TrumpForPrison','TrumpLies','TrumpsArmy','Trumpstrong','ChristiansForTrump','WomenForTrump','DrainTheSwamp','HillaryClinton','pizzagate','ClintonEmails','ClintonCrimeFamily','HillaryForPrison','Hillary2016','CriminalClinton','NeverHillary','DonaldTrump','WomenWhoVoteTrump','dumptrump','VoteTrump','CrookedHillary','Women4Trump','Blacks4Trump','fucktrump']
    hashtags = []
    for word in keywords:
        hashtags.append('#'+word.lower())
    hash_dict = {}
    c = 0
    with gzip.open(corpus_file,'rt') as f:
        print("Counting occurences of hashtags in {}...".format(corpus_file.split('/')[-1]))
        for line in f:
            json_line = json.loads(line.strip('\n'))
            text = json_line['text']
            c+=1
            text = text.lower()
            for hash in hashtags:
                if hash in text:
                    if hash not in hash_dict:
                        hash_dict[hash] = 1
                    else:
                        hash_dict[hash] += 1
    for k,v in hash_dict.items():
        v2 = "{0:.3f}".format((v/c)*100)
        v = "{:,}".format(v)
        hash_dict[k] = [v,float(v2)]
    [ print(key , " :: " , value) for (key, value) in sorted(hash_dict.items() , reverse=True, key=lambda x: x[1][1]  ) ]
    print("{:,}".format(c))

def mix_data(corpus_file):
    """ shuffle the data """
    mixlist = []
    with open(corpus_file,'r') as f:
        for line in f:
            line = line.strip("\n")
            mixlist.append(line)
    np.random.shuffle(mixlist)
    fPath = corpus_file.split(".txt")[0] + "_v2.txt"
    print("writing to "+fPath)
    with open(fPath,'w') as f2:
        for line in mixlist:
            f2.write(line+"\n")
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrapes tweets, normalizes the messages, counts the distribution, shuffles the data')
    parser.add_argument('-cmd', type=str, help='scrape (scrape tweets from karora)/norm (normalize tweet messages)/count (count the distribution of keywords)/mix (shuffle data)')
    parser.add_argument('-input', type=str, help='either directory path or corpus file path ("scrape" needs dirPath, others need filePath)')
    args = parser.parse_args()
    
    cmd = args.cmd
    ip = args.input
    
    if cmd == 'scrape':
        scraper(ip)
    if cmd == 'norm':
        norm(ip)
    if cmd == 'count':
        read_data(ip)
    if cmd == 'mix':
        mix_data(ip)
