# Romanization of Thai Name

developing Seq2Seq model that can romanize Thai person name 

mainly focusing **End-to-End Model** that does not need other data and linguistic information except for input/output data

## Data

> original data
 
- **3,801** distinct names with romanized annotations
- RTGS(Royal Thai General System of Transcription) and other variations
- gender f/m

> cleaned data

- exclude ambiguous names e.g. **เอกรัฐ** : 'ekkarat' or 'ekrat'
- **3,556** distinct names
- train: **2,856** names (20% validation for preliminary train)
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

1. greedy search
2. beam search : keep 3 most probable candidates

## Result

preliminary train with validation - no overfitting

![loss](https://user-images.githubusercontent.com/44984892/174543126-0d9923db-9dd9-4c58-bcb0-92e152c2b7b7.png)

> Result of name data only

||tltk|LSTM w/o|LSTM attention|LSTM w/o + dict|LSTM attention + dict|
|:-:|:-:|:-:|:-:|:-:|:-:|
|WER|0.101235|0.259259|0.156790|0.172840|0.108642|
|CER macro|0.013554|0.075115|0.038668|0.043355|0.026617|
|CER micro|0.013101|0.075955|0.038693|0.044962|0.028573|
