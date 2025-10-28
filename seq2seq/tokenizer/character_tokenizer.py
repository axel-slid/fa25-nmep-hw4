from .tokenizer import Tokenizer

import torch


class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        If verbose is True, prints out the vocabulary.

        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
        """
        super().__init__()

        self.vocab = {}

        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = """aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}’•–í€óá«»… º◦©ö°äµ—ø­·òãñ―½¼γ®⇒²▪−√¥£¤ß´úª¾є™，ﬁõ  �►□′″¨³‑¯≈ˆ§‰●ﬂ⇑➘①②„≤±†✜✔➪✖◗¢ไทยếệεληνικαåşıруский 한국어汉语ž¹¿šćþ‚‛─÷〈¸⎯×←→∑δ■ʹ‐≥τ;∆℡ƒð¬¡¦βϕ▼⁄ρσ⋅≡∂≠π⎛⎜⎞ω∗"""
        
        for i, char in enumerate(self.vocab):
            self.vocab[char] = i

        if verbose:
            print("Vocabulary:", self.vocab)

        # raise NotImplementedError("Need to implement vocab initialization")

    def encode(self, text: str) -> torch.Tensor:

        tokens = []

        for c in text.lower():
            if c in self.vocab:
                tokens.append(c)
            else:
                raise ValueError("you have a char that is not in the valid chars")
            
        return torch.tensor(tokens)


        #raise NotImplementedError(
        #    "Need to implement encoder that converts text to tensor of tokens."
        #)

    def decode(self, tokens: torch.Tensor) -> str:

        diht = {v: k for k, v in self.vocab.items()}

        decoded = []

        for t in tokens:
            if t.item() in diht:
                decoded.append(diht[t.item()])
            else:
                raise ValueError("you have a char that is not in the valid chars")



        #raise NotImplementedError(
        #    "Need to implement decoder that converts tensor of tokens to text."
        #)
