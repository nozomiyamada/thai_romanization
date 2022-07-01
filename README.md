# Romanization of Thai Name

developing Seq2Seq model that can romanize Thai person name 

mainly focusing **End-to-End Model** that does not need other data and linguistic information except for input/output data

## Data

> original data
 
- **3,806** distinct names with romanized annotations
- RTGS(Royal Thai General System of Transcription) and other variations
- gender f/m

> cleaned data

- **3,806** distinct names, **3,847** entries
- some names are ambiguous e.g. **เอกรัฐ** : 'ekkarat' or 'ekrat'
- use **3,766** non-ambiguous names
- train: **3,066** names (20% validation for preliminary train)
- test : **700** names

> additional data

- dictionary of Thai **26,828** words (exclude mono-syllabic words)
- e.g. `นายตรวจ,naitruat` `คนรัก,khonrak`

## Evaluation Metrics

- **WER(word error rate)** : to check whole correspondense of each name
- **CER macro** : the average of character error rate of each name 
- **CER micro** : the weighted average of character error rate of each name 

## Model

0. `tltk.nlp.th2roman()` - **benchmark**

1. LSTM without attention

encoder: BiLSTM, decoder: LSTM

![model_train](https://user-images.githubusercontent.com/44984892/174532893-8ff54723-457a-4a33-a12c-c437d9e78934.png)

2. LSTM with attention

![model_attention](https://user-images.githubusercontent.com/44984892/175194689-5ed0d2ec-ce10-4067-a47d-7b6edaaea24c.png)

3. Transformer

## Inference

1. Greedy Search
2. Beam Search : keep 3 most probable candidates

## Result

preliminary train with validation - no overfitting

![loss](https://user-images.githubusercontent.com/44984892/174543126-0d9923db-9dd9-4c58-bcb0-92e152c2b7b7.png)

> Result of Greedy Search

||tltk|LSTM|LSTM attention|LSTM + dict|LSTM attention + dict|
|:-:|:-:|:-:|:-:|:-:|:-:|
|WER|**0.090782**|0.259259|0.156790|0.122905|**0.090782**|
|CER macro|**0.012128**|0.075115|0.038668|0.035918|0.024459|
|CER micro|**0.012049**|0.075955|0.038693|0.036534|0.024111|

> Result of Beam Search (tltk has no beam search)

||tltk|LSTM|LSTM attention|LSTM + dict|LSTM attention + dict|
|:-:|:-:|:-:|:-:|:-:|:-:|
|WER|**0.090782**|0.259259|0.156790|0.117318|**0.090782**|
|CER macro|**0.012128**|0.075115|0.038668|0.033874|0.023728|
|CER micro|**0.012049**|0.075955|0.038693|0.034186|0.023680|
