import re

### function for cleaning data
def clean(text):
    text = re.sub('เเ', 'แ', text)
    text = re.sub(r'ํา','ำ', text) # o + า -> ำ
    text = re.sub(r'\u0E33([\u0E48\u0E49\u0E4A\u0E4B])', r'\1'+'\u0E33', text) # am + tone -> tone + am
    return text

### THAI CHARACTERS <> ID
THAI2INDEX = {'<PAD>':0, '^':1, '$':2}
for i in range(3585, 3674): # ก=3585->3, ๙=3673
    THAI2INDEX[chr(i)] = i - 3582
INDEX2THAI = {v:k for k,v in THAI2INDEX.items()}

### ROMAN CHARACTERS <> ID
ROMAN2INDEX = {'<PAD>':0, '^':1, '$':2, '-':3}
for i in range(97, 122): # 'a'=97->4, 'z'=122
    ROMAN2INDEX[chr(i)] = i - 93
INDEX2ROMAN = {v:k for k,v in ROMAN2INDEX.items()}

### NUMBER OF IDs
NUM_INPUT_INDEX = len(THAI2INDEX) # num of Thai characters
NUM_OUTPUT_INDEX = len(ROMAN2INDEX) # num of Roman characters
