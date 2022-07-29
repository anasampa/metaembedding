# -*- coding: utf-8 -*-
#!pip install sentence-transformers

from sentence_transformers import SentenceTransformer
from tensorflow.keras.layers import Dense, Input, concatenate, multiply, average, subtract, add, dot, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_probability as tfp
import numpy as np
from lime.lime_text import LimeTextExplainer

class MetaEmbedding():

  def __init__(self,embedding_models,vectorizer=False,fusion_dim=512):

    self.emb_models = embedding_models
    if not vectorizer:
      vectorizer = self.get_embedding

    self.pairs_example = [['This is an example','This is an accepted input format.']]

    #def make_model(self):
    merges = list()
    self.inputs = list()
    sents_X_train = list()

    # Inputs based on embedding models size
    for model_name in self.emb_models:

      text1_emb, text2_emb = self.get_embedding(model_name,self.pairs_example[:1])

      inp1 = Input(shape=(text1_emb.shape[1])) # tamanho da embedding
      inp2 = Input(shape=(text2_emb.shape[1]))
      self.inputs.append(inp1)
      self.inputs.append(inp2)
      print(model_name, text1_emb.shape[1])

      ci =multiply([inp1,inp2])

      merge_i = Dense(fusion_dim,activation='linear')(ci)
      merges.append(merge_i) #guardando todos os modelos

    # Layers
    merge_layer = concatenate(merges)

    meta_embedding_layer_sentence1 = Dense(fusion_dim, activation='linear')(merge_layer)
    meta_embedding_layer_sentence2 = Dense(fusion_dim, activation='linear')(merge_layer)

    operation_layer = multiply([meta_embedding_layer_sentence1,meta_embedding_layer_sentence2])

    self.output_meta = Dense(1,activation='linear')(operation_layer)

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

    #print(model_name)

    vectorizer = SentenceTransformer(model_name)
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

  def val_run(self, X_train,Y_train, x_val=False, y_val=False, epoch=1, shuffle=True):

    if not x_val:
      X_train, x_val, Y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15, shuffle=True)

    x_train_emb = self.get_models_emb(X_train)
    x_val_emb = self.get_models_emb(x_val)

    callback = EarlyStopping(monitor='loss', patience=2)
    #model_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min') # colocar no callbacks quando quiser salvar
    self.model.fit(x_train_emb, Y_train, epochs=2, shuffle=shuffle,validation_data=(x_val_emb, y_val),callbacks=[callback])
    return self.model

  def train_run(self, X_train,Y_train, epoch=1, shuffle=True):

    x_train_emb = self.get_models_emb(X_train)

    callback = EarlyStopping(monitor='loss', patience=2)
    #model_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min') # colocar no callbacks quando quiser salvar
    self.model.fit(x_train_emb, Y_train, epochs=2, shuffle=shuffle,callbacks=[callback])
    return self.model


  def predict_model(self,X_test):
    X_test_emb = list()

    for model_name in self.emb_models:
      text1_emb, text2_emb = self.get_embedding(model_name,X_test)

      X_test_emb.append(text1_emb)
      X_test_emb.append(text2_emb)

    return self.model.predict(X_test_emb)
