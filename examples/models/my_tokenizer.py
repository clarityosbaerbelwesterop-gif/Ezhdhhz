"""Your own tokenizer: `--tokenizer examples/models/my_tokenizer.py:WordPieceLite`.

Any class (or factory) with encode(str) -> list[int], decode(list[int]) -> str and vocab_size works.
This one maps lower-case letters, digits and a few symbols to ids and everything else to <unk>.
"""


class WordPieceLite:
    ALPHABET = " abcdefghijklmnopqrstuvwxyz0123456789.,;:!?'\"-\n()"

    def __init__(self):
        self.index = {c: i + 1 for i, c in enumerate(self.ALPHABET)}

    @property
    def vocab_size(self):
        return len(self.ALPHABET) + 1          # id 0 = <unk>

    def encode(self, text):
        return [self.index.get(c, 0) for c in text.lower()]

    def decode(self, ids):
        return "".join(self.ALPHABET[i - 1] if i else "?" for i in ids)
