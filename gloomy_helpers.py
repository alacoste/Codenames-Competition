def words_from_file(filename):
    with open(filename) as f:
        return [line.strip() for line in f]

def get_game_vocabulary():
    vocab = words_from_file('./data/vocab.txt')
    game_words = words_from_file('./data/game_words.txt')
    game_vocab = list(set(vocab) | set(game_words))
    return game_vocab, game_words