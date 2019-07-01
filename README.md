# ma-thesis

Python programs used for ma-thesis on: Politically Oriented Embeddings for Abusive Language Detection

twitter.py: used for scraping the data, normalizing the messages, counting the distributions, shuffling the data.

jsd.py: used for calculating the Jensen Shannon Divergence scores between data sets.

SVM_original.py: used as the main program, contains two LinearSVC models: one based on ngrams using a CountVectorizer, the other uses embeddings. Calls the BiLSTM.

helperFunctions.py: contains small helpful functions for SVM_original.py.

features.py: contains features used in SVM_original.py.

transformers.py: containes features used in SVM_original.py.

BiLSTM.py: contains the BiLSTM model using embeddings.
