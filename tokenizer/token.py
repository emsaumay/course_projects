import re
import json
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, vocab_size=1000, special_tokens=None):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.bpe_merges = []
        self.token2id = {}
        self.id2token = {}
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>']

    def byte_encode(self, text):
        # Converts text to a list of unicode bytes
        return [chr(b) for b in text.encode('utf-8')]

    def get_stats(self, corpus):
        pairs = defaultdict(int)
        for word in corpus:
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += 1
        return pairs

    def build_vocab(self, text):
        # Byte-level encoding for each word
        corpus = [' '.join(self.byte_encode(word)) + ' </w>' for word in text.split()]
        vocab = Counter(corpus)
        self.vocab = dict(vocab)
        for _ in range(self.vocab_size):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.bpe_merges.append(best)
            self.vocab = self.merge_vocab(best, self.vocab)
        # Build token2id and id2token
        tokens = set()
        for word in self.vocab:
            tokens.update(word.split())
        tokens = list(tokens)
        # Add special tokens first
        tokens = self.special_tokens + [t for t in tokens if t not in self.special_tokens]
        self.token2id = {t: i for i, t in enumerate(tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}

    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in vocab:
            w_out = pattern.sub(''.join(pair), word)
            new_vocab[w_out] = vocab[word]
        return new_vocab

    def encode(self, text, add_special_tokens=True):
        tokens = []
        if add_special_tokens:
            tokens.append('<BOS>')
        for word in text.split():
            chars = self.byte_encode(word) + ['</w>']
            while True:
                pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
                pair_to_merge = None
                for merge in self.bpe_merges:
                    if merge in pairs:
                        pair_to_merge = merge
                        break
                if not pair_to_merge:
                    break
                i = 0
                new_chars = []
                while i < len(chars):
                    if i < len(chars)-1 and (chars[i], chars[i+1]) == pair_to_merge:
                        new_chars.append(''.join(pair_to_merge))
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            tokens.extend(chars)
        if add_special_tokens:
            tokens.append('<EOS>')
        # Convert to ids, use <UNK> if not found
        return [self.token2id.get(t, self.token2id['<UNK>']) for t in tokens]

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [self.id2token.get(i, '<UNK>') for i in token_ids]
        words = []
        word = []
        for token in tokens:
            if skip_special_tokens and token in self.special_tokens:
                continue
            if token == '</w>':
                words.append(''.join(word))
                word = []
            else:
                word.append(token)
        if word:
            words.append(''.join(word))
        return ' '.join(words)

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump({
                'token2id': self.token2id,
                'id2token': self.id2token,
                'bpe_merges': self.bpe_merges,
                'special_tokens': self.special_tokens
            }, f)

    def load_vocab(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.token2id = data['token2id']
            self.id2token = {int(k): v for k, v in data['id2token'].items()}
            self.bpe_merges = [tuple(m) for m in data['bpe_merges']]
            self.special_tokens = data['special_tokens']

# Example usage:
if __name__ == "__main__":
    text = "hello 😊 world! GPT tokenizer test"
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.build_vocab(text)
    encoded = tokenizer.encode("hello world! 😊")
    print("Encoded IDs:", encoded)
    print("Decoded:", tokenizer.decode(encoded))
    tokenizer.save_vocab("vocab.json")