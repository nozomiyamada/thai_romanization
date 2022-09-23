# Data Dictionary

- `name_romanization_original_data.csv` : original data of annotated Thai names by Prim
- `name_romanization.csv` : cleaned data for training
  - `name` : Thai names
  - `gender` : `f` or `m`
  - `ipa` : **encoded IPA**, each syllable consists of (onset, vowel, coda, tone) e.g. `KwAj1`
  - `romanize1` : romanization according to RTGS
  - `romanize2`, `romanize3` : other variations of transcription

~~~
name,gender,ipa,romanize1,romanize2,romanize3
กุลสตรี,f,kun1 la-1 sat2 trI-1,kunlasattri,kullasattri,
กุสุมา,f,ku-2 su-2 mA-1,kusuma,gusuma,
กุสุมาลย์,f,ku-2 su-2 mAn1,kusuman,kusumal,gusumal
กุหลาบ,f,ku-1 lAp2,kulap,kulab,kularb
ก่อเกียรติ,m,kX-2 kJt2,kokiat,korkiat,gorkiat
~~~

- `g2p_dict` : additional dictionary data including only **words with at least 2 syllables**
  - **26,828** words
  - `grapheme` : Thai transcription
  - `phoneme` : **encoded IPA**
  - `rtgs` : RTGS transcription

~~~
grapheme,phoneme,rtgs
กกกอด,kok2 kXt2,kokkot
กกช้าง,kok2 CAN4,kokchang
กกธูป,kok2 TUp3,kokthup
กกหู,kok2 hU-5,kokhu
กกุธภัณฑ์,ka-2 kut2 Ta-4 Pan1,kakutthaphan
~~~ 

- `thai2phone.csv` : original data of `g2p_dict`, with many missing values

- `number2phone.csv` : numbers and their encoded IPA transcriptions

- `train_x.txt`, `train_y.txt`, `train_y_ipa.txt`, `test_x.txt`, `test_y.txt`,`test_y_ipa.txt` : split data of `name_romanization.csv` 

# encoded IPA in data

### Onset

|phoneme in Thai|encoded|IPA|RTGS|Haas system|
|:-:|:-:|:-:|:-:|:-:|
|/บ/|b|b|b|b|
|/ป/|p|p|p|p|
|/พ/|P|pʰ|ph|ph|
|/ม/|m|m|m|m|
|/ฟ/|f|f|f|f|
|/ด/|d|d|d|d|
|/ต/|t|t|t|t|
|/ท/|T|tʰ|th|th|
|/น/|n|n|n|n|
|/ส/|s|s|s|s|
|/ร/|r|r|r|r|
|/ล/|l|l|l|l|
|/จ/|c|tɕ|ch|c|
|/ช/|C|tɕʰ|ch|ch|
|/ก/|k|k|k|k|
|/ค/|K|kʰ|kh|kh|
|/ง/|N|ŋ|ng|ŋ|
|/ว/|w|w|w|w|
|/ย/|j|j|y|y|
|/ห/|h|h|h|h|
|/อ/|?|ʔ|-|ʔ|

### Vowel

|phoneme in Thai|encoded|IPA|RTGS|Haas|
|:-:|:-:|:-:|:-:|:-:|
|/อะ/|a|a|a|a|
|/อา/|A|aː|a|aa|
|/อิ/|i|i|i|i|
|/อี/|I|iː|i|ii|
|/อุ/|u|u|u|u|
|/อู/|U|uː|u|uu|
|/อึ/|v|ɯ|ue|ɯ|
|/อือ/|V|ɯː|ue|ɯɯ|
|/เอะ/|e|e|e|e|
|/เอ/|E|eː|e|ee|
|/แอะ/|y|ɛ|ae|ɛ|
|/แอ/|Y|ɛː|ae|ɛɛ|
|/โอะ/|o|o|o|o|
|/โอ/|O|oː|o|oo|
|/เอาะ/|x|ɔ|o|ɔ|
|/ออ/|X|ɔː|o|ɔɔ|
|/เออะ/|z|ə|oe|ə|
|/เออ/|Z|əː|oe|əə|
|/เอีย/|J|iə|ia|ia|
|/เอือ/|W|ɯə|uea|ɯa|
|/อัว/|R|uə|ua|ua|
