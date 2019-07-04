# ma-thesis RUG IK
## Politically Oriented Embeddings for Abusive Language Detection

### Structure
- **prog** contains python programs used for ma-thesis on: Politically Oriented Embeddings for Abusive Language Detection:
  - **twitter.py**: used for scraping the data, normalizing the messages, counting the distribution of the keywords, shuffling the data.
  - **jsd.py**: used for calculating the Jensen Shannon Divergence scores between data sets.
  - **SVM_original.py**: used as the main program, contains two LinearSVC models: one based on char and word ngrams using a CountVectorizer, the other uses embeddings. Calls the BiLSTM.
  - **helperFunctions.py**: contains small helpful functions for SVM_original.py, such as read functions for the data sets and embeddings.
  - **features.py**: contains features used in SVM_original.py.
  - **transformers.py**: contains transformers used in SVM_original.py.
  - **BiLSTM.py**: contains the BiLSTM model using embeddings.

### Data
- [Waseem & Hovy 2016](https://github.com/ZeerakW/hatespeech)
- [OffensEval Task A](https://competitions.codalab.org/competitions/20011)
- [HatEval English Task A](https://competitions.codalab.org/competitions/19935)
- StackOverflow Offensive comments

### Embeddings
- [GloVe Twitter](https://nlp.stanford.edu/projects/glove/)
- [fastText wiki news](https://fasttext.cc/docs/en/english-vectors.html)
- [fastText wiki news with subword information](https://fasttext.cc/docs/en/english-vectors.html)
- polarized fastText embeddings created from politically oriented tweet messages

## Abstract
In this work, we present a methodology for generating biased word embedding representations from social media, based on politically oriented messages, as a way to improve the portability of trained models across data sets labelled with different abusive language phenomena. We thus compare and investigate the performance of the polarized embeddings against pre-trained generic embeddings across three related in-domain data sets and one out-of-domain data set for abusive language detection. Furthermore we explore the in-data and cross-data use of a linear and a deep-learning model, two embedding libraries and several methods for data-collection.

*Most of these scripts are re-used and modified, original versions from:*

- [Nissim et al, GermEval](https://github.com/malvinanissim/germeval-rug)
- [Balint Hompot, OffensEval](https://github.com/BalintHompot/RUG_Offenseval)
- [Grunn2019 at SemEval 2019 Task 5](https://bitbucket.org/grunn2018/sharedhate_repo/src/master/)
