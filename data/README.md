## data dictionary

- `name_romanization_original_data.csv` : original data of annotated Thai names
- `name_romanization.csv` : cleaned data for training
  - `name` : Thai names
  - `gender` : `f` or `m`
  - `ipa` : **encoded** IPA, each syllable consists of (onset, vowel, coda, tone) e.g. `KwAj1`
  - `romanize1` : romanization according to RTGS
  - `romanize2`, `romanize3` : other variations


## encoded IPA in data

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
