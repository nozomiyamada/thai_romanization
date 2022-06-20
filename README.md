# Romanization of Thai Name

developing Seq2Seq model that can romanize Thai person name 

## Data

- 4049 names with romanized annotations
- RTGS(Royal Thai General System of Transcription) and other variations
- gender f/m

## Evaluation Criteria

- accuracy: to check whole correspondense of each name
- CER macro: the average of character error rate of each name 
- CER micro: the weighted average of character error rate of each name 

## Model

0. `tltk.nlp.th2roman()` - **benchmark**
1. LSTM without attention

![model_train](https://user-images.githubusercontent.com/44984892/174532893-8ff54723-457a-4a33-a12c-c437d9e78934.png)

2. LSTM with attention
3. Transformer

## Result
### LSTM w/o attention 
train: 3239 names (20% validation for preliminary train)
test :  810 names
epoch: 100

preliminary train with validation

![loss](https://user-images.githubusercontent.com/44984892/174543126-0d9923db-9dd9-4c58-bcb0-92e152c2b7b7.png)

result 

||`tltk`|LSTM w/o|
|:-:|:-:|:-:|
|accuracy|0.190123|0.243210|
|CER macro|0.058586|0.071396|
|CER micro|0.064011|0.071626|
