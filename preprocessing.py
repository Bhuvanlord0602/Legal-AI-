from tokenizer import KannadaSTTTokenizer

tokenizer = KannadaSTTTokenizer()

def preprocess_text(text):
    return tokenizer.clean_text(text)

def tokenize_text(text):
    return tokenizer.tokenize(text)

def encode_text(text):
    return tokenizer.encode(text)