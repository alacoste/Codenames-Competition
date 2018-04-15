from demo_bot import SpyBot
import random
import gensim

WORD2VEC_EMBEDDING = './data/GoogleNews-trimmed-word2vec-negative300.bin'
SEMANTRIS_EMBEDDING = './data/semantris.bin'

class GloomyBot(SpyBot):

    def __init__(self, vocab, game_board, p_id):
        self.game_board = game_board
        self.p_id = p_id
        self.game_state = {word: False for word in self.game_board}
        
        # Load trimmed word2vec model.
        print('Loading word2vec model...')
        model = gensim.models.KeyedVectors.load_word2vec_format(SEMANTRIS_EMBEDDING, binary=True)
        self.vocab = list(set(vocab) & set(model.vocab))
        print('Model loaded: I am playing with %d potential clue words!' % len(self.vocab))
        
        # Compute similarity of all legal clue words we have an embedding for to all the board words.
        print('Pre-computing the similarity of all legal clue words to all board words.')
        self.similarities = {clue_word:{board_word:model.similarity(clue_word, board_word) for board_word in self.game_board} for clue_word in self.vocab}
        print('Pre-computed word similarities!')
        pass
        
    def updateWithGuesses(self, guesses):
        for word in guesses:
            self.game_state[word] = True
        pass
        
    def update(self, is_my_turn, clue_word, clue_num_guesses, guesses):
        updateWithGuesses(guesses)
        pass

    def getUncoveredWords(self):
        return [(word, type) for word, type in self.game_board.items() if not self.game_state[word]]
        
    def getMyWords(self):
        return [word for word, type in self.getUncoveredWords() if type == self.p_id]

    def getOpponentWords(self):
        return [word for word, type in self.getUncoveredWords() if type == (1 - self.p_id)]

    def getNeutralWords(self):
        NEUTRAL = 2
        return [word for word, type in self.getUncoveredWords() if type == NEUTRAL]

    def getAssasinWord(self):
        SPY = 3
        assasins = [word for word, type in self.game_board.items() if type == SPY]
        assert len(assasins) == 1
        return assasins[0]
        
    def getSimilarity(self, clue_word, board_word):
        return self.similarities[clue_word][board_word]
    
    # Basic clue evaluation logic, how many of my words are top-similar to the clue with enough margin against other words.
    def evaluateClue(self, clue_word):
        NEUTRAL_MARGIN = 0.08
        OPPONENT_MARGIN = 0.1
        ASSASIN_MARGIN = 0.15
    
        my_similarities = [self.getSimilarity(clue_word, word) for word in self.getMyWords()]
        opponent_similarities = [self.getSimilarity(clue_word, word) for word in self.getOpponentWords()]
        neutral_similarities = [self.getSimilarity(clue_word, word) for word in self.getNeutralWords()]
        assasin_similarity = self.getSimilarity(clue_word, self.getAssasinWord())
        min_similarity = max(max(opponent_similarities) + OPPONENT_MARGIN, max(neutral_similarities) + NEUTRAL_MARGIN, assasin_similarity + ASSASIN_MARGIN)
        
        my_top_similarities = [s for s in my_similarities if s > min_similarity]
        last_word_margin = (min(my_top_similarities) - min_similarity) if len(my_top_similarities) > 0 else 0
        return len(my_top_similarities), last_word_margin    
    
    def describeSimilarities(self, clue_word, board_words):
        board_words_similarities = {word : self.getSimilarity(clue_word, word) for word in board_words}
        board_words_by_descending_similarity = [(word, board_words_similarities[word]) for word in sorted(board_words_similarities, key=board_words_similarities.get, reverse=True)]
        for word, similarity in board_words_by_descending_similarity:
            print('%s: %.3f' % (word, similarity))
        pass
    
    def describeClue(self, clue_word):
        num_words, last_word_margin = self.evaluateClue(clue_word)
        print('\nDescribing the clue [%s], which top-associates with %d of my words and has a last-word-margin of %.3f' % (clue_word, num_words, last_word_margin))
        print('\nMy words:')
        self.describeSimilarities(clue_word, self.getMyWords())
        print('\nOpponent words:')
        self.describeSimilarities(clue_word, self.getOpponentWords())
        print('\nNeutral words:')
        self.describeSimilarities(clue_word, self.getNeutralWords())
        print('\nAssasin word:' )
        self.describeSimilarities(clue_word, [self.getAssasinWord()])
        pass
    
    def getClue(self, invalid_words):
        legal_clue_words = list(set(self.vocab).difference(invalid_words))
        print('Looking for the best clue out of %d legal clue words...' % len(legal_clue_words))
        
        best_clue = 'N/A'
        best_num_words = 0
        best_last_word_margin = 0
        for clue_word in legal_clue_words:
            num_words, last_word_margin = self.evaluateClue(clue_word)
            if (num_words > best_num_words or (num_words == best_num_words and last_word_margin > best_last_word_margin)):
                best_clue = clue_word
                best_num_words = num_words
                best_last_word_margin = last_word_margin
        
        self.describeClue(best_clue)
        return best_clue, best_num_words