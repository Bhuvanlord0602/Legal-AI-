import re

class KannadaSTTConfig:
    special_tokens = ['<blank>']
    swaras = ['ಅ','ಆ','ಇ','ಈ','ಉ','ಊ','ಋ','ಎ','ಏ','ಐ','ಒ','ಓ','ಔ']
    vyanjanas = [
        'ಕ','ಖ','ಗ','ಘ','ಙ','ಚ','ಛ','ಜ','ಝ','ಞ',
        'ಟ','ಠ','ಡ','ಢ','ಣ','ತ','ಥ','ದ','ಧ','ನ',
        'ಪ','ಫ','ಬ','ಭ','ಮ','ಯ','ರ','ಲ','ವ','ಶ','ಷ','ಸ','ಹ','ಳ'
    ]
    matras = ['ಾ','ಿ','ೀ','ು','ೂ','ೃ','ೆ','ೇ','ೈ','ೊ','ೋ','ೌ']
    modifiers = ['ಂ','ಃ','್']
    numbers = ['೦','೧','೨','೩','೪','೫','೬','೭','೮','೯']
    punctuation = [' ', '.', ',', '-', '!', '?', '(', ')', '/', "'", '"']

    @classmethod
    def get_full_vocab(cls):
        return (cls.special_tokens + cls.swaras + cls.vyanjanas +
                cls.matras + cls.modifiers + cls.numbers + cls.punctuation)

class KannadaSTTTokenizer:
    def __init__(self):
        self.vocab = KannadaSTTConfig.get_full_vocab()
        self.char_to_id = {char: i for i, char in enumerate(self.vocab)}
        self.id_to_char = {i: char for i, char in enumerate(self.vocab)}

        self.digit_map = str.maketrans("0123456789", "".join(KannadaSTTConfig.numbers))
        self.token_pattern = re.compile(r'[\u0C80-\u0CFF]|[ .,!?()/\'"-]')

    def clean_text(self, text):
        text = text.replace('\u200c', '').replace('\u200d', '')
        text = text.translate(self.digit_map)
        return " ".join(text.split())

    def tokenize(self, text):
        return self.token_pattern.findall(text)

    def encode(self, text):
        return [self.char_to_id[t] for t in self.tokenize(text) if t in self.char_to_id]