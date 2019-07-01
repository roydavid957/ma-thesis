# ma-thesis

Python programs used for ma-thesis on: Politically Oriented Embeddings for Abusive Language Detection

twitter.py: used for scraping the data, normalizing the messages, counting the distributions, shuffling the data.

jsd.py: used for calculating the Jensen Shannon Divergence scores between data sets.

SVM_original.py: used as the main program, contains two LinearSVC models: one based on ngrams using a CountVectorizer, the other uses embeddings. Calls the BiLSTM. (Original version: https://github.com/malvinanissim/germeval-rug)

helperFunctions.py: contains small helpful functions for SVM_original.py.

features.py: contains features used in SVM_original.py. (Original version: https://github.com/malvinanissim/germeval-rug)

transformers.py: containes features used in SVM_original.py. (Original version: https://github.com/malvinanissim/germeval-rug)

BiLSTM.py: contains the BiLSTM model using embeddings. (Original version: https://github.com/malvinanissim/germeval-rug)
