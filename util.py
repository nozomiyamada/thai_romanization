import re

### function for cleaning data
def clean(text):
    text = re.sub('เเ', 'แ', text)
    text = re.sub(r'ํา','ำ', text) # o + า -> ำ
    text = re.sub(r'\u0E33([\u0E48\u0E49\u0E4A\u0E4B])', r'\1'+'\u0E33', text) # am + tone -> tone + am
    return text

### THAI CHARACTERS <> ID
thai2id = {'<PAD>':0, '^':1, '$':2}
for i in range(3585, 3674): # ก=3585->3, ๙=3673
    thai2id[chr(i)] = i - 3582
id2thai = {v:k for k,v in thai2id.items()}

### ROMAN CHARACTERS <> ID
roman2id = {'<PAD>':0, '^':1, '$':2, '-':3}
for i in range(97, 122): # a=97->4, z=122
    roman2id[chr(i)] = i - 93
id2roman = {v:k for k,v in roman2id.items()}

### NUMBER OF IDs
num_input_id = len(thai2id) # num of Thai characters
num_output_id = len(roman2id) # num of Roman characters
