# Metaembedding

### Install
!pip install https://github.com/anasampa/metaembedding/archive/plusmetatool.zip

## 1. Model

### 1.1 Import
```
from metaembedding.metaembedding import MetaEmbedding
```


### 1.2 Choose pre-trained models 

List of possible models: https://www.sbert.net/docs/pretrained_models.html

```
model1 = 'paraphrase-multilingual-mpnet-base-v2' 
model2 = 'paraphrase-multilingual-MiniLM-L12-v2' 
model3 = 'distiluse-base-multilingual-cased-v1' 

embedding_models = [model1, model2, model3]
```

### 1.3 Metaembedding model
```
model = MetaEmbedding(embedding_models)
model.summary()
```

### 1.4 Train
```
model.train_run(X_train,Y_train, epochs=2)
```

### 1.5 Predict
```
model.predict_model([['I am a sentence','I am another sentence']])
```

### 1.5 Save weights from trained model
```
model.save_weights('name_of_file')
```

### 1.6 Load saved weights from trained model 

```
weight = 'name_of_file'

embedding_models = [model1, model2, model3]
model = MetaEmbedding(embedding_models)
model.load_weights(weight)
```

## 2. Explain model using LIME

Original lime: https://github.com/marcotcr/lime

### 2.1 Show explanation in notebook
```
s1 = 'I am a sentence'
s2 = 'I am another sentence'
pair = [s1,s2]

model.lime.explain_in_notebook(pair,num_features=30,num_samples=50)
```

### 2.2 Access values as a list
```
model.lime.explain_as_list(pair,num_features=30,num_samples=50)
```

