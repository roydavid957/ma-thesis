# ma-thesis

Python programs used for ma-thesis on: Politically Oriented Embeddings for Abusive Language Detection

twitter.py: used for scraping the data, normalizing the messages, counting the distribution of the keywords, shuffling the data.

jsd.py: used for calculating the Jensen Shannon Divergence scores between data sets.

SVM_original.py: used as the main program, contains two LinearSVC models: one based on char and word ngrams using a CountVectorizer, the other uses embeddings. Calls the BiLSTM.

helperFunctions.py: contains small helpful functions for SVM_original.py, such as read functions for the data sets and embeddings.

features.py: contains features used in SVM_original.py.

transformers.py: containes transformers used in SVM_original.py.

BiLSTM.py: contains the BiLSTM model using embeddings.



Most of these scripts are re-used and modified, original versions from:

-https://github.com/malvinanissim/germeval-rug

-https://github.com/BalintHompot/RUG_Offenseval

-https://bitbucket.org/grunn2018/sharedhate_repo/src/master/
