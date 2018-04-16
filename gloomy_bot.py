import gensim
import heapq
from demo_bot import SpyBot
from operator import mul
from functools import reduce

class Strategy():
    MAX_WORDS_AND_MAX_LAST_WORD_MARGIN = 1
    MAX_TOTAL_DISCOUNTED_SIMILARITY = 2

class GloomyBot(SpyBot):

    def loadWord2VecTrimmedModel(self):
        print('Loading trimmed word2vec model...')
        self.model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-trimmed-word2vec-negative300.bin', binary=True)
        pass
    
    def loadSemantrisModel(self):
        print('Loading semantris model...')
        self.model = gensim.models.KeyedVectors.load('./data/semantris.model')
        pass

    def __init__(self, vocab, game_board, p_id):
        self.game_board = game_board
        self.p_id = p_id
        self.game_state = {word: False for word in self.game_board}
        self.vocab = vocab  # Will be filtered down later to the subset of words that the model supports.
        self.strategy = Strategy.MAX_TOTAL_DISCOUNTED_SIMILARITY
        
        # Load word vector model.
        self.loadSemantrisModel()
        assert len(set(self.game_board).difference(set(self.model.vocab))) == 0  # All game board words need to be in the model.
        self.vocab = list(set(self.vocab) & set(self.model.vocab))  # The "bot vocabulary" may be smaller than the game vocabulary if some words are not in the model.
        print('Model loaded: I am playing with %d potential clue words!' % len(self.vocab))
        
        # Compute similarity of all legal clue words we have an embedding for to all the board words.
        print('Pre-computing the similarity of all legal clue words to all board words.')
        self.similarities = {clue_word : {board_word : self.model.similarity(clue_word, board_word) for board_word in self.game_board} for clue_word in self.vocab}
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
    
    def describeSimilarities(self, clue_word, board_words):
        board_words_similarities = {word : self.getSimilarity(clue_word, word) for word in board_words}
        board_words_by_descending_similarity = [(word, board_words_similarities[word]) for word in sorted(board_words_similarities, key=board_words_similarities.get, reverse=True)]
        for word, similarity in board_words_by_descending_similarity:
            print('%30s: %.3f' % (word, similarity))
        pass
        
    def describeAllSimilarities(self, clue_word):
        print('\nDescribing all similarities for clue [%s]' % clue_word)
        print('\nMy words:')
        self.describeSimilarities(clue_word, self.getMyWords())
        print('\nOpponent words:')
        self.describeSimilarities(clue_word, self.getOpponentWords())
        print('\nNeutral words:')
        self.describeSimilarities(clue_word, self.getNeutralWords())
        print('\nAssasin word:' )
        self.describeSimilarities(clue_word, [self.getAssasinWord()])
        print('')
        pass
        
    def printClueDebug(self, clue_word, num_words, score, debug_dict):
        print('\nClue %s for %d words with score %s' % (clue_word, num_words, tuple(['%.3f' % x for x in score])))
        print()
        for key, value in debug_dict.items():
            print('%30s: %s' % (key, value))
        self.describeAllSimilarities(clue_word)
    
    def getClue(self, invalid_words):
        legal_clue_words = list(set(self.vocab).difference(invalid_words))
        print('Looking for the best clue out of %d legal clue words...' % len(legal_clue_words))
        
        NUM_TOP_CLUES = 10
        top_clues_heap = []
        for clue_word in legal_clue_words:
            num_words, score, debug_dict = self.evaluateClue(clue_word)
            # Push on a heap, sorted by score.
            heapq.heappush(top_clues_heap, (score, (clue_word, num_words, score, debug_dict)))
            while len(top_clues_heap) > NUM_TOP_CLUES:
                heapq.heappop(top_clues_heap)
        
        # Put the top-K best clues in a list, best first.
        top_clues = [0] * NUM_TOP_CLUES
        while len(top_clues_heap) > 0:
            unused, clue = heapq.heappop(top_clues_heap)
            top_clues[len(top_clues_heap)] = clue
            
        for clue_word, num_words, score, debug_dict in top_clues:
            print('%20s for %d words, with score %s' % (clue_word, num_words, tuple(['%.3f' % x for x in score])))
        
        # Debugging for best and next-best clues.
        for clue_word, num_words, score, debug_dict in top_clues:
            self.printClueDebug(clue_word, num_words, score, debug_dict)
            print_next_clue_debug = ''
            while print_next_clue_debug != 'y' and print_next_clue_debug != 'n':
                print_next_clue_debug = input('Print debug for next best clue? [y/n]')
            if print_next_clue_debug == 'n':
                break
        
        best_clue_word, best_num_words, best_score, best_debug_dict = top_clues[0]
        return best_clue_word, best_num_words
        
    # This should return a tuple (num_words, score, debug_dict) where:
    # 'num_words' will be the number given with the clue if it is selected.
    # 'debug_dict' contains information that can be printed for debugging.
    # 'score' is a tuple such that if clue X is better than clue Y, then (x1, x2, ...) > (y1, y2, ...).
    def evaluateClue(self, clue_word):
        if self.strategy == 1:
            return self.evaluateClueS1(clue_word)
        elif self.strategy == 2:
            return self.evaluateClueS2(clue_word)
        else:
            assert self.strategy > 0 and self.strategy <= 2
        
    # ================================================================
    # STRATEGY: MAX_WORDS_AND_MAX_LAST_WORD_MARGIN
    
    def evaluateClueS1(self, clue_word):
        NEUTRAL_MARGIN = 0.08
        OPPONENT_MARGIN = 0.1
        ASSASIN_MARGIN = 0.2
    
        my_similarities = [self.getSimilarity(clue_word, word) for word in self.getMyWords()]
        opponent_similarities = [self.getSimilarity(clue_word, word) for word in self.getOpponentWords()]
        neutral_similarities = [self.getSimilarity(clue_word, word) for word in self.getNeutralWords()]
        assasin_similarity = self.getSimilarity(clue_word, self.getAssasinWord())
        min_similarity = max(max(opponent_similarities) + OPPONENT_MARGIN, max(neutral_similarities) + NEUTRAL_MARGIN, assasin_similarity + ASSASIN_MARGIN)
        
        my_top_similarities = [s for s in my_similarities if s > min_similarity]
        num_words = len(my_top_similarities)
        last_word_margin = (min(my_top_similarities) - min_similarity) if len(my_top_similarities) > 0 else 0
        score = (num_words, last_word_margin)
        debug_dict = {'similarity_threshold': min_similarity, 'last_word_similarity': min(my_top_similarities)}
        
        return (num_words, score, debug_dict)
        
    # ================================================================
    # STRATEGY: MAX_TOTAL_DISCOUNTED_SIMILARITY
    
    def evaluateClueS2(self, clue_word):
        def penaltyFunction(similarity, bad_word_similarity, full_margin, zero_margin):
            assert full_margin >= zero_margin
            if similarity > bad_word_similarity + full_margin:
                return 1  # No penality
            elif similarity < bad_word_similarity + zero_margin:
                return 0
            else:
                return (similarity - (bad_word_similarity + zero_margin)) / (full_margin - zero_margin)
        
        NEUTRAL_FULL_MARGIN = 0.1
        NEUTRAL_ZERO_MARGIN = -0.1
        OPPONENT_FULL_MARGIN = 0.2
        OPPONENT_ZERO_MARGIN = 0.0
        ASSASIN_FULL_MARGIN = 0.3
        ASSASIN_ZERO_MARGIN = 0.1
        def getScore(similarity, opponent_similarities, neutral_similarities, assasin_similarity):
            # If a word has less than these margins, it gets penalized.
            neutral_penalty = reduce(mul, [penaltyFunction(similarity, bad_word_similarity, NEUTRAL_FULL_MARGIN, NEUTRAL_ZERO_MARGIN) for bad_word_similarity in neutral_similarities], 1)
            opponents_penalty = reduce(mul, [penaltyFunction(similarity, bad_word_similarity, OPPONENT_FULL_MARGIN, OPPONENT_ZERO_MARGIN) for bad_word_similarity in opponent_similarities], 1)
            assasin_penalty = penaltyFunction(similarity, assasin_similarity, ASSASIN_FULL_MARGIN, ASSASIN_ZERO_MARGIN)
            return similarity * neutral_penalty * opponents_penalty * assasin_penalty
    
        my_similarities = [self.getSimilarity(clue_word, word) for word in self.getMyWords()]
        opponent_similarities = [self.getSimilarity(clue_word, word) for word in self.getOpponentWords()]
        neutral_similarities = [self.getSimilarity(clue_word, word) for word in self.getNeutralWords()]
        assasin_similarity = self.getSimilarity(clue_word, self.getAssasinWord())
        
        # Get discounted similarity scores.
        my_similarity_scores = [getScore(similarity, opponent_similarities, neutral_similarities,assasin_similarity) for similarity in my_similarities]
        score = sum(my_similarity_scores)
        
        # Get the number of words to clue for.
        MIN_SIMILARITY_SCORE = 0.3
        my_similarity_scores_over_threshold = [score for score in my_similarity_scores if score > MIN_SIMILARITY_SCORE]
        num_words = len(my_similarity_scores_over_threshold)
        
        # Debug info:
        debug_dict = {}
        debug_dict['score'] = '%.3f' % score
        debug_dict['score_from_unclued'] = '%.3f' % (score - sum(my_similarity_scores_over_threshold))
        debug_dict['neutral_threshold'] = 'full: %.3f, zero: %.3f' % (max(neutral_similarities) + NEUTRAL_FULL_MARGIN, max(neutral_similarities) + NEUTRAL_ZERO_MARGIN)
        debug_dict['opponent_threshold'] = 'full: %.3f, zero: %.3f' % (max(opponent_similarities) + OPPONENT_FULL_MARGIN, max(opponent_similarities) + OPPONENT_ZERO_MARGIN)
        debug_dict['assasin_threshold'] = 'full: %.3f, zero: %.3f' % (assasin_similarity + ASSASIN_FULL_MARGIN, assasin_similarity + ASSASIN_ZERO_MARGIN)
        zipped_and_sorted = sorted(zip(my_similarities, self.getMyWords(), my_similarity_scores), reverse=True)
        my_similarities, my_words, my_similarity_scores = map(list, zip(*zipped_and_sorted))
        for i in range(0, len(my_similarities)):
            if my_similarity_scores[i] > 0:
                debug_dict[my_words[i]] = '%.3f (score: %.3f)' % (my_similarities[i], my_similarity_scores[i])
        
        return (num_words, (score,), debug_dict)
    