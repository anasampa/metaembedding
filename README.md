# Metaembedding

Sentence similarity prediction tool based on the combination of pre-trained Transformers based models.

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

### 1.3 Build the metaembedding model
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

## 2. Explain the prediction of intances using LIME

Generally, similarity tasks in Natural Language Processing is a regression multi-text task.   
The original LIME text module is only for classification tasks with single text inputs, which prevents its direct use for similarity comparison. 
We extended the original LIME to apply it in models that take multiple texts as inputs, such as pairs of sentences, and also for accepting regression models with text entrances. 

More about the sentence similarity task and this LIME extension can be found in the "Sentence Similarity Recognition in Portuguese from Multiple Embedding Models" (citation at the end of the readme).

Original LIME (without the extension): https://github.com/marcotcr/lime

### 2.1 Show explanation in notebook
```
s1 = 'I am a sentence'
s2 = 'I am another sentence'
pair = [s1,s2]

model.lime.explain_in_notebook(pair,num_features=30,num_samples=50)
```
ps: The token [SEP] is diplayed to indicate sentence separation. It is not computed in the predictions of the model.  

### 2.2 Access values as a list
```
model.lime.explain_as_list(pair,num_features=30,num_samples=50)
```

## Cite

### .bib
<pre><code>@inproceedings{rodrigues2022sentence,
  title={Sentence Similarity Recognition in Portuguese from Multiple Embedding Models},
  author={Rodrigues, Ana Carolina and Marcacini, Ricardo M.},
  booktitle={2022 21th IEEE international conference on machine learning and applications (ICMLA)},
  pages={154--159},
  year={2022},
  organization={IEEE}
}</code></pre>

