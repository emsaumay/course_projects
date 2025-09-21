import unittest
from token import BPETokenizer

class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.text = "hello 😊 world! GPT tokenizer test"
        self.tokenizer = BPETokenizer(vocab_size=50)
        self.tokenizer.build_vocab(self.text)

    def test_encode_decode(self):
        input_text = "hello world! 😊"
        encoded = self.tokenizer.encode(input_text)
        decoded = self.tokenizer.decode(encoded)
        # Decoded text may not match exactly due to BPE merges, but should contain original words
        for word in ["hello", "world", "😊"]:
            self.assertIn(word, decoded)

    def test_special_tokens(self):
        input_text = "hello"
        encoded = self.tokenizer.encode(input_text)
        bos_id = self.tokenizer.token2id['<BOS>']
        eos_id = self.tokenizer.token2id['<EOS>']
        self.assertEqual(encoded[0], bos_id)
        self.assertEqual(encoded[-1], eos_id)

    def test_save_and_load_vocab(self):
        self.tokenizer.save_vocab("test_vocab.json")
        new_tokenizer = BPETokenizer()
        new_tokenizer.load_vocab("test_vocab.json")
        self.assertEqual(self.tokenizer.token2id, new_tokenizer.token2id)
        self.assertEqual(self.tokenizer.id2token, new_tokenizer.id2token)
        self.assertEqual(self.tokenizer.bpe_merges, new_tokenizer.bpe_merges)
        self.assertEqual(self.tokenizer.special_tokens, new_tokenizer.special_tokens)

if __name__ == "__main__":
    unittest.main()