from demo_bot import SpyBot
import random
import gensim

class GloomyBot(SpyBot):

    def __init__(self, vocab, game_board, p_id):
        self.vocab = set(vocab)
        model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)

    def update(self, is_my_turn, clue_word, clue_num_guesses, guesses):
        pass

    def getClue(self, invalid_words):
        return random.choice(list(self.vocab.difference(invalid_words))), 2