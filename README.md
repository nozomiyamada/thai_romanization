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

0. `tltk.nlp.th2roman()` - benchmark
1. LSTM without attention
2. LSTM with attention
3. Transformer

# Result

