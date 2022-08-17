# -*- coding: utf-8 -*-
#!pip install sentence-transformers

from sentence_transformers import SentenceTransformer
from tensorflow.keras.layers import Dense, Input, concatenate, multiply, average, subtract, add, dot, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_probability as tfp
import numpy as np
from lime.lime_text import LimeTextExplainer
import ktrain
from ktrain import text


class Transformer():
    def __init__(self,model_name):

        models = {'bert_base_cased':'bert-base-multilingual-cased',
        'bert_pt_base_cased':'neuralmind/bert-base-portuguese-cased',
        'bert_pt_large_cased':'neuralmind/bert-large-portuguese-cased'}

        if model_name in models.values():
            self.MODEL_NAME = model_name
        else:
            try:
                self.MODEL_NAME = models[model_name]
            except:
                raise TypeError('Not recognized model. Please pick a valid model name. See https://huggingface.co/models for more models options.')

    def train(self,x_train, y_train,batch_size=32,learning_rate=5e-5,epochs=2):

        BATCH_SIZE = batch_size
        LEARNING_RATE = learning_rate
        EPOCH = epochs

        t = text.Transformer(MODEL_NAME, maxlen=128)
        trn = t.preprocess_train(x_train, y_train)
        model = t.get_classifier()
        learner = ktrain.get_learner(model, train_data=trn, batch_size=BATCH_SIZE) # lower bs if OOM occurs
        learner.fit_onecycle(LEARNING_RATE, EPOCH)

        predictor = ktrain.get_predictor(learner.model, t)
        return predictor

    def val(self,x_train, y_train, x_val, y_val,batch_size=32,learning_rate=5e-5,epochs=2):

        BATCH_SIZE = batch_size
        LEARNING_RATE = learning_rate
        EPOCH = epochs

        t = text.Transformer(MODEL_NAME, maxlen=128)
        trn = t.preprocess_train(x_train, y_train)
        val = t.preprocess_test(x_val, y_val)
        model = t.get_classifier()
        learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=BATCH_SIZE) # lower bs if OOM occurs
        learner.fit_onecycle(LEARNING_RATE, 5)

        predictor = ktrain.get_predictor(learner.model, t)
        return predictor


class LimeExplain():

  def __init__(self,model_predictor):
    # Lime setup specific for metaembedding predictor.
    self.explainer = LimeTextExplainer(mode='regression')
    self.model_predictor = model_predictor

  def explain_instance(self,pair,num_features=40,num_samples=60):
    self.explanation = self.explainer.explain_instance(pair,num_features=num_features,num_samples=num_samples,classifier_fn=self.model_predictor,multiple_texts=True)
    return self.explanation

  def explain_as_list(self,pair,num_features=40,num_samples=60):
    # Shortcut
    return self.explain_instance(pair,num_features=num_features,num_samples=num_samples).as_list()

  def explain_in_notebook(self,pair,num_features=40,num_samples=60):
    return self.explain_instance(pair,num_features=num_features,num_samples=num_samples).show_in_notebook()


class MetaEmbedding():

  def __init__(self,embedding_models,vectorizer=False,fusion_dim=512):

    self.lime = LimeExplain(self.predict_model)
    self.emb_models = embedding_models
    self.pairs_example = [['This is an example','This is an accepted input format.']]

    # Option for custumize vectorization (future)
    if not vectorizer:
      vectorizer = self.get_embedding

    merges = list()
    self.inputs = list()
    sents_X_train = list()

    # Inputs based on embedding models size
    for model_name in self.emb_models:

      text1_emb, text2_emb = self.get_embedding(model_name,self.pairs_example[:1])

      inp1 = Input(shape=(text1_emb.shape[1])) # size of embedding
      inp2 = Input(shape=(text2_emb.shape[1]))
      self.inputs.append(inp1)
      self.inputs.append(inp2)
      print(model_name, text1_emb.shape[1])

      ci = multiply([inp1,inp2])

      merge_i = Dense(fusion_dim,activation='linear')(ci)
      merges.append(merge_i)

    # Layers
    merge_layer = concatenate(merges)

    meta_embedding_layer_sentence1 = Dense(fusion_dim, activation='linear')(merge_layer)
    meta_embedding_layer_sentence2 = Dense(fusion_dim, activation='linear')(merge_layer)

    operation_layer = multiply([meta_embedding_layer_sentence1,meta_embedding_layer_sentence2])

    self.output_meta = Dense(1,activation='sigmoid')(operation_layer)

    model = Model(inputs=self.inputs, outputs=self.output_meta)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae',self.tf_pearson])
    self.model = model

  def fit(self):
    return self.model.fit()

  def score(self):
    return self.model.score()

  def save(self, path):
    return self.model.save(path)

  def summary(self):
    return self.model.summary()

  def save_weights(self,path):
    return self.model.save_weights(path)

  def load_weights(self,path):
    return self.model.load_weights(path)



  def get_embedding(self,model_name,pairs):
    # data aqui é lista de pares de str [['eu e voce','você é legal'],['eu também','ele também']]
    data_s1 = list(zip(*pairs))[0]
    data_s2 = list(zip(*pairs))[1]

    vectorizer = SentenceTransformer(model_name) # Defaut
    s1_vector = vectorizer.encode(data_s1)
    s2_vector = vectorizer.encode(data_s2)

    return s1_vector, s2_vector

  def tf_pearson(self,y_true, y_pred):
    return tfp.stats.correlation(y_pred, y_true, sample_axis=0, event_axis=None)

  def get_models_emb(self,X_train):

    X_train_emb = list()
    for model_name in self.emb_models:
      print(model_name)

      text1_emb, text2_emb = self.get_embedding(model_name,X_train)

      X_train_emb.append(text1_emb)
      X_train_emb.append(text2_emb)

    return X_train_emb

  def train_run(self, X_train,Y_train, x_val=False, y_val=False, epochs=1, shuffle=True):

    callback = EarlyStopping(monitor='loss', patience=2)
    x_train_emb = self.get_models_emb(X_train)

    if x_val is False and y_val is False:
      self.model.fit(x_train_emb, Y_train, epochs=epochs, shuffle=shuffle,callbacks=[callback])
    elif x_val is not False and y_val is not False:
      x_val_emb = self.get_models_emb(x_val)
      #X_train, x_val, Y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15, shuffle=True)
      self.model.fit(x_train_emb, Y_train, epochs=epochs, shuffle=shuffle,validation_data=(x_val_emb, y_val),callbacks=[callback])
    else:
        raise TypeError("Validation input error. Missing x_val or y_val.")
    return self.model

  def predict_model(self,X_test):
    X_test_emb = list()

    for model_name in self.emb_models:
      text1_emb, text2_emb = self.get_embedding(model_name,X_test)

      X_test_emb.append(text1_emb)
      X_test_emb.append(text2_emb)

    return self.model.predict(X_test_emb)
