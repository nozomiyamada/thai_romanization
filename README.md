# Romanization of Thai Name

developing Seq2Seq model that can romanize Thai person name 

mainly focusing **End-to-End Model** that does not need other data and linguistic information except for input/output data

## Existing Tools

- `tltk.nlp.th2roman()`
- `pythainlp.romanize()`
- https://www.thaicorpus.tk/g2p - my website (it just looks up in dictionary OR use `tltk` so far)

## Data

> original data
 
- **3,806** distinct names with annotations; IPA, RTGS, and other variations
- RTGS = Royal Thai General System of Transcription
- gender f/m
- e.g. `ก้อง,m,kON2,kong,gong,`

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

## Model Variations

Possible models are factorial. **Model Architecture** (3) x **Output** (3) x **Additional Data** (3) x **Prediction Method** (2) = **54** petterns

`tltk.nlp.th2roman()` - is used as benchmark

> #### Model Architecture

1. **LSTM w/o attention** : encoder-BiLSTM, decoder-LSTM
2. **LSTM with attention**
3. **Transformers** : self-attention encoder/decoder

> #### Output (target)

1. **roman** - End-to-End conversion from Thai to Roman
e.g. สมใจ -> `['s', 'o', 'm', 'j', 'a', 'i']`
2. **syllable token** - one syllable is treated one token (vocab) 
e.g. สมใจ -> `["som5", "jai1"]`
3. **phoneme** - one syllable consists of 4 phonological features `(onset, vowel, coda, tone)`
e.g. สมใจ -> `[('s', 'o', 'm', 5, 'B'), ('j', 'a', 'i', 1, 'E')]`
decoder has 5 outputs (4 features + BIEO tagging)


> #### Additional Data

1. **w/o dictionary** : only 3,806 Thai names
2. **with dictionary** : 3,806 Thai names + 26,828 words
3. **Davor's data** : 100K+ names (no IPA annotations), must be filtered

> #### Predction Method

1. **Greedy Search**
2. **Beam Search** : keep 3 most probable candidates

### model example

> LSTM w/o attention

![model_train](https://user-images.githubusercontent.com/44984892/174532893-8ff54723-457a-4a33-a12c-c437d9e78934.png)

> LSTM with attention

![model_attention](https://user-images.githubusercontent.com/44984892/175194689-5ed0d2ec-ce10-4067-a47d-7b6edaaea24c.png)


## Result

preliminary train with validation - no overfitting

![loss](https://user-images.githubusercontent.com/44984892/174543126-0d9923db-9dd9-4c58-bcb0-92e152c2b7b7.png)

> Result of Greedy Search

||tltk|LSTM|LSTM attention|LSTM + dict|LSTM attention + dict|LSTM attention, syl token|Transformers(5)|Transformers(5) + dict|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|WER|**0.090782**|0.259259|0.156790|0.122905|**0.090782**|0.215|0.123|0.051|
|CER macro|**0.012128**|0.075115|0.038668|0.035918|0.024459|0.073|0.034|0.018|
|CER micro|**0.012049**|0.075955|0.038693|0.036534|0.024111|0.073|0.034|0.018|

> Result of Beam Search (tltk has no beam search)

||tltk|LSTM|LSTM attention|LSTM + dict|LSTM attention + dict|
|:-:|:-:|:-:|:-:|:-:|:-:|
|WER|**0.090782**|0.259259|0.156790|0.117318|**0.090782**|
|CER macro|**0.012128**|0.075115|0.038668|0.033874|0.023728|
|CER micro|**0.012049**|0.075955|0.038693|0.034186|0.023680|
