from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Bidirectional, Embedding, concatenate
from keras.layers import Concatenate, Conv1D, Activation, TimeDistributed, Flatten, RepeatVector, Permute,multiply
from keras.layers import GlobalMaxPool1D, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, SpatialDropout1D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

seed = 1337
np.random.seed(seed)
# Import Files



class AttentionWithContext(Layer):
    '''
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        'Hierarchical Attention Networks for Document Classification'
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        '''

    def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.built = True

    def compute_mask(self, input, mask):
        return None

    def call(self, x, mask=None):
        # (x, 40, 300) x (300, 1)
        multData =  K.dot(x, self.kernel) # (x, 40, 1)
        multData = K.squeeze(multData, -1) # (x, 40)
        multData = multData + self.b # (x, 40) + (40,)

        multData = K.tanh(multData) # (x, 40)

        multData = multData * self.u # (x, 40) * (40, 1) => (x, 1)
        multData = K.exp(multData) # (X, 1)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            multData = mask*multData #(x, 40) * (x, 40, )

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        multData /= K.cast(K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        multData = K.expand_dims(multData)
        weighted_input = x * multData
        return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)



def training_biLSTM(Xtrain, Ytrain, Xtest, Ytest, embeddings_index, tknzr, modelh5, eps, ptc, vb, bs, cv, c, pte, nrange):
    if cv:
        tknzr = tknzr.rstrip('.pickle') + '_CV_' + str(c) + '.pickle'
        modelh5 = modelh5.rstrip('.mh5') + '_CV_' + str(c) + '.mh5'

    y_train_reshaped = to_categorical(Ytrain, num_classes=2)

    t = Tokenizer()
    t.fit_on_texts(Xtrain)
    vocab_size = len(t.word_index) + 1
    Xtrain = t.texts_to_sequences(Xtrain)
    max_length = max([len(s) for s in Xtrain + Xtest])
    X_train_reshaped = pad_sequences(Xtrain, maxlen=max_length, padding='post')

    with open(tknzr, 'wb') as handle:
        pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Padded the data')

    ## Loading in word embeddings and setting up matrix
    print('Loaded %s word vectors' % len(embeddings_index))
    if pte.split('/')[-1] == 'glove.twitter.27B.200d.txt':
        dim = 200
    else:
        dim = 300
    embedding_matrix = np.zeros((vocab_size, dim)) #Dimension vector in embeddings
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
        if embedding_vector is None:
        
            if nrange == '3to3':
                if len(word) > 3:
                    first_tri = word[:3]
                    last_tri = word[-3:]
                    embedding_vector1 = embeddings_index.get(first_tri)
                    embedding_vector2 = embeddings_index.get(last_tri)
                    if embedding_vector1 is not None:
                        embedding_matrix[i] = embedding_vector1
#                        if embedding_vector2 is not None:
#                            embedding_matrix[i].append(embedding_vector2)
                    elif embedding_vector2 is not None:
                        embedding_matrix[i] = embedding_vector2
            
            if nrange == '3to6':
                if len(word) > 3:
                    tri_grams = [word.lower()[i:i+3] for i in range(len(word.lower())-1)]
                    for gram in tri_grams:
                        if len(gram) == 3:
                            if gram.lower() in word_embeds:
                                list_of_embeddings.append(word_embeds[gram.lower()])
                
                if len(word) > 4:
                    tri_grams = [word.lower()[i:i+4] for i in range(len(word.lower())-1)]
                    for gram in tri_grams:
                        if len(gram) == 4:
                            if gram.lower() in word_embeds:
                                list_of_embeddings.append(word_embeds[gram.lower()])
                
                if len(word) > 5:
                    tri_grams = [word.lower()[i:i+5] for i in range(len(word.lower())-1)]
                    for gram in tri_grams:
                        if len(gram) == 5:
                            if gram.lower() in word_embeds:
                                list_of_embeddings.append(word_embeds[gram.lower()])
                
                if len(word) > 6:
                    tri_grams = [word.lower()[i:i+6] for i in range(len(word.lower())-1)]
                    for gram in tri_grams:
                        if len(gram) == 6:
                            if gram.lower() in word_embeds:
                                list_of_embeddings.append(word_embeds[gram.lower()])
                                    
    print('Loaded embeddings')

    ### Setting up model
    print('Setting up model..')
    embedding_layer = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_length, trainable=False, mask_zero=True)
    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm = Bidirectional(LSTM(512, return_sequences=True))(embedded_sequences)
    l_drop = Dropout(0.4)(l_lstm)
    l_att = AttentionWithContext()(l_drop)
    preds = Dense(2, activation='softmax')(l_att)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
    print(model.summary())

    ######## Preparing test data
    y_test_reshaped = to_categorical(Ytest, num_classes=2)
    X_test = t.texts_to_sequences(Xtest)
    X_test_reshaped = pad_sequences(X_test, maxlen=max_length, padding='post')
    print('Done preparing testdata')

    checkpoint = ModelCheckpoint(modelh5, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ptc)
    callbacks_list = [checkpoint, es]
    model.fit(X_train_reshaped, y_train_reshaped, epochs=eps, batch_size=bs, validation_data=(X_test_reshaped, y_test_reshaped), callbacks=callbacks_list, verbose=vb)
    loss, accuracy = model.evaluate(X_test_reshaped, y_test_reshaped, verbose=vb)


def output_biLSTM(Xtrain, Ytrain, Xtest, Ytest, tknzr, modelh5, cv, c):
    if cv:
        tknzr = tknzr.rstrip('.pickle') + '_CV_' + str(c) + '.pickle'
        modelh5 = modelh5.rstrip('.mh5') + '_CV_' + str(c) + '.mh5'

    print('Loading tokenizer...')
    with open(tknzr, 'rb') as handle:
        t = pickle.load(handle)

    print('Tokenizer loaded! Loading model...')
    model = load_model(modelh5, custom_objects={'AttentionWithContext': AttentionWithContext})
    print('Model loaded! Processing data...')

    datalist_reshaped = t.texts_to_sequences(Xtest)
    try:
        max_length = max([len(s) for s in Xtest])
        datalist_reshaped = pad_sequences(datalist_reshaped, maxlen=max_length, padding='post')
        score = model.predict(datalist_reshaped)
    except ValueError:
        try:
            max_length = max([len(s) for s in Xtrain + Xtest])
            datalist_reshaped = pad_sequences(datalist_reshaped, maxlen=max_length, padding='post')
            score = model.predict(datalist_reshaped)
        except ValueError:
            try:
                max_length = max([len(s) for s in Xtrain])
                datalist_reshaped = pad_sequences(datalist_reshaped, maxlen=max_length, padding='post')
                score = model.predict(datalist_reshaped)
            except ValueError:
                try:
                    datalist_reshaped = pad_sequences(datalist_reshaped, maxlen=851, padding='post')
                    score = model.predict(datalist_reshaped)
                except ValueError:
                    try:
                        datalist_reshaped = pad_sequences(datalist_reshaped, maxlen=2337, padding='post')
                        score = model.predict(datalist_reshaped)
                    except ValueError:
                        try:
                            datalist_reshaped = pad_sequences(datalist_reshaped, maxlen=580, padding='post')
                            score = model.predict(datalist_reshaped)
                        except ValueError:
                            try:
                                datalist_reshaped = pad_sequences(datalist_reshaped, maxlen=392, padding='post')
                                score = model.predict(datalist_reshaped)
                            except ValueError:
                                datalist_reshaped = pad_sequences(datalist_reshaped, maxlen=214, padding='post')
                                score = model.predict(datalist_reshaped)
            except Exception:
                print('Failed..')

    print('Data processed! Predicting values...')
    yguess = np.argmax(score, axis=1)

    yguess = [str(item) for item in yguess]
    Ytest = [str(item) for item in Ytest]
    print(set(Ytest))
    print(set(yguess))

    print('Predictions made! returning output')
    # accuracy = accuracy_score(Ytest, yguess)
    # precision, recall, f1score, support = precision_recall_fscore_support(Ytest, yguess, average='macro')
    report = classification_report(Ytest, yguess)
    print(report)

    if cv:
        reportlist = report.strip().split('\n')
        newlist = []
        for i in reportlist:
            if i != '':
                i = i.split()[-4:-1]
                newlist.append(i)
        newlist = newlist[1:-1]
        NOT_prec = float(newlist[0][0])
        NOT_rec = float(newlist[0][1])
        NOT_f1 = float(newlist[0][2])

        OFF_prec = float(newlist[1][0])
        OFF_rec = float(newlist[1][1])
        OFF_f1 = float(newlist[1][2])

        return NOT_prec, NOT_rec, NOT_f1, OFF_prec, OFF_rec, OFF_f1
    else:
        return Ytest, yguess


def biLSTM(Xtrain, Ytrain, Xtest, Ytest, training, output, embeddings_index, tknzr, modelh5, cv, eps, ptc, ds, vb, bs, prob, pte, nrange):
    if training:
        training_biLSTM(Xtrain, Ytrain, Xtest, Ytest, embeddings_index, tknzr, modelh5, eps, ptc, vb, bs, cv, 0, pte, nrange)
        print('Done training')

    if cv:
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        c = 0

        NOT_prec = 0
        NOT_rec = 0
        NOT_f1 = 0
        OFF_prec = 0
        OFF_rec = 0
        OFF_f1 = 0
        for train, test in kfold.split(Xtrain, Ytrain):
            c += 1

            np_Xtrain = np.array(Xtrain)
            np_Ytrain = np.array(Ytrain)
            X_train, X_test = np_Xtrain[train], np_Xtrain[test]
            Y_train, Y_test = np_Ytrain[train], np_Ytrain[test]

            print('X_train {}, X_test {}'.format(len(X_train), len(X_test)))
            print('Y_train {}, Y_test {}'.format(len(Y_train), len(Y_test)))

            training_biLSTM(X_train.tolist(), Y_train.tolist(), X_test.tolist(), Y_test.tolist(), embeddings_index, tknzr, modelh5, eps, ptc, vb, bs, cv, c, pte, nrange)
            print('CV: {} - Done training'.format(str(c)))

            notp, notr, notf1, offp, offr, off1 = output_biLSTM(X_train.tolist(), Y_train.tolist(), X_test.tolist(), Y_test.tolist(), tknzr, modelh5, cv, c)

            NOT_prec += notp
            NOT_rec += notr
            NOT_f1 += notf1
            OFF_prec += offp
            OFF_rec += offr
            OFF_f1 += off1

        print("NOTprec  {}  NOTrec  {}  NOTf1  {}".format(NOT_prec/10, NOT_rec/10, NOT_f1/10))
        print("OFFprec  {}  OFFrec  {}  OFFf1  {}".format(OFF_prec/10, OFF_rec/10, OFF_f1/10))
        print("ALLf1  {}".format((NOT_f1+OFF_f1)/20))
        exit()

    if not output:
        exit()

    if output:
        Ytest, yguess = output_biLSTM(Xtrain, Ytrain, Xtest, Ytest, tknzr, modelh5, cv, 0) # c=0 cuz no CV, shape=851 placeholder
        return Ytest, yguess

    return True
