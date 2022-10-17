# Metaembedding

### Install
!pip install https://github.com/anasampa/metaembedding/archive/plusmetatool.zip

## 1. Model

### 1.1 Import
```
from metaembedding.metaembedding import MetaEmbedding
```


### 1.2 Choose pre-trained models 
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
model.predict_model([['Eu comi banana','Eu comi abacate']])
```

### 1.5 Load saved weights from trained model 

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
model.lime.explain_in_notebook(pair,num_features=50,num_samples=60)
```

### 2.2 Access values as a list
```
model.lime.explain_as_list(pair,num_features=25,num_samples=50)
```

