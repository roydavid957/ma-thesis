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
    """ normalize tweet messages """
    uuh_list = ['<user>','<url>','<hashtag>']
#    save_file = corpus_file.split('.txt.gz')[0]+'_'+'TWEETSpl.txt'
#    print('writing to {}'.format(save_file))
#    f2 = open(save_file,'w')

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
            yield list(TweetTokenizer().tokenize(text))

#            text = ' '.join(TweetTokenizer().tokenize(text))
#            if text != '':
#                f2.write(text+'\n')
#    f2.close()

def gensim_ftm(corpus_file):
#https://radimrehurek.com/gensim/models/fasttext.html
#https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb
#https://fasttext.cc/docs/en/support.html
    save_file = "embeddings/twitterPOL_FTgensim"
    
    try:
        model_gensim = FT_gensim.load(save_file)
        print(model_gensim)
        b = False
    except:
        b = True
        pass

    if b:
        # Set file names for train and test data
        # corpus_file = datapath('lee_background.cor')
        print("corpus_file read: {}".format(corpus_file))
        
        siz = 300
        minc = 1
        win = 5
        print("model set: size={}, min_count={}, window={}".format(str(siz),str(minc),str(win)))
        model_gensim = FT_gensim(size=siz, min_count=minc, window=win)
        
        # build the vocabulary
        print("build_vocab: {}".format(corpus_file))
        model_gensim.build_vocab(sentences=norm(corpus_file))
        
        # train the model
        print("training model: corpus_file={}, epochs={}, total_examples={:,}, total_words={:,}".format(corpus_file, model_gensim.epochs, model_gensim.corpus_count, model_gensim.corpus_total_words))
        model_gensim.train(corpus_file=corpus_file, epochs=model_gensim.epochs, total_examples=model_gensim.corpus_count, total_words=model_gensim.corpus_total_words)
        
        print(model_gensim)
        print("save to: {}".format(save_file))
        model_gensim.save(save_file)

    keywords = ['man', 'woman', 'gay', 'homosexual', 'migrants', 'immigrants', 'black', 'trans', 'maga', 'trump', 'fakenews', 'god', 'racism', 'atheism', 'abortion', 'guncontrol', 'americafirst', 'humanity', 'metoo']
    for word in keywords:
        most_sim_word = word
        print(most_sim_word)
        most_sim = model_gensim.most_similar(most_sim_word)
        print(most_sim)

def scraper(dir_path):#,hashtags):
    """ filter the tweets from karora based on requirements """
    keywords = ['trump','Hillary','MAGA','realDonaldTrump','buildthewall','MakeAmericaGreatAgain','TrumpTrain','Trump2016','AmericaFirst','TrumpForPrison','TrumpLies','TrumpsArmy','Trumpstrong','ChristiansForTrump','WomenForTrump','DrainTheSwamp','HillaryClinton','pizzagate','ClintonEmails','ClintonCrimeFamily','HillaryForPrison','Hillary2016','CriminalClinton','NeverHillary','DonaldTrump','WomenWhoVoteTrump','dumptrump','VoteTrump','CrookedHillary','Women4Trump','Blacks4Trump','fucktrump']
#    keywords = ['liberal','conservative','dems','democrat','DemocratsAreDestroyingAmerica','LiberalHypocrisy','LiberalismIsAMentalDisorder','republican','democrats','LiberalsAreFascists','WalkAwayFromDemocratsForever','SocialismKills','WalkAwayFromDemocrats','CriminalDemocrats','DemocratsAreThePartyOfSocialism','NeverDemocrat','nationalist','RepublicanParty',]
#    keywords = ['🇨🇦','🇬🇧','🇺🇸','🏳️‍🌈']
    hashtags = []
    for word in keywords:
        hashtags.append(word.lower())
    """
    with open('hashtags.txt', 'r') as f:
        for line in f:
            hashtags.append(line.rstrip('\n').lower())
    """

    p_list = []        # polarised
    np_list = []       # non-polarised
    wordcount_p = 0
    wordcount_np = 0
    c = 0

    directory = dir_path

    p_file = gzip.open('../../../data/s2764989/roy/tweets/txt_twitter_p2_'+hashtags[0]+'_'+hashtags[1]+'_'+hashtags[2]+'_'+directory.split('/')[-1]+'.txt.gz', 'wt')
#    feedback = open('txt_twitter_feedback2_'+hashtags[0]+'_'+hashtags[1]+'_'+hashtags[2]+'_'+directory.split('/')[-1]+'.txt', 'w')

    for root, dirs, files in os.walk(directory):
        print('Total     files: {:,}\n'.format(len(files)))
        for file in files:
            c += 1
            """
            if file not in out_files:
                out_files.append(str(file))
            """
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
                        # else:
                        #     np_list.append(text)
                        #     np_file.write(text)
                        #     wordcount_np += len(word_tokenize(text))

                file_wc = str(file) + ' - ' + str(wordcount_p)
#                feedback.write(file_wc+'\n')
                print('Total     polarised: {:,}\n'.format(wordcount_p))
                # print('Total non-polarised: {:,}\n'.format(wordcount_np))

    p_file.close()
    # np_file.close()
#    feedback.close()

    # print(p_list)

def read_data(corpus_file):
    """ count the distrinution of the keywords """
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
    """ shuffle the data for reliability check of the embeddings purposes """
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
    parser.add_argument('-cmd', type=str, help='scrape/norm/count/mix')
    parser.add_argument('-input', type=str, help='either directory path or corpus file path')
    args = parser.parse_args()
    
    cmd = args.cmd
    ip = args.input
    
    if cmd == 'scrape':
        scraper(ip)
    if cmd == 'norm':
#    gensim_ftm(sys.argv[1])
        norm(ip)
    if cmd == 'count':
        read_data(ip)
    if cmd == 'mix':
        mix_data(ip)
