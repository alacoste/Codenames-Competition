import gensim
import numpy

def read_vocabulary():
    with open('./data/vocab.txt') as f:
        return [line.strip() for line in f]
        
def get_trimmed_model(game_vocab, model_vocab, model_vectors, trim_len_target = 0):
    # By default, our target will be to keep all of the game_vocab.
    if (trim_len_target <= 0):
        trim_len_target = len(game_vocab)
    trimmed_model_vocab = {}
    trimmed_model_vectors = numpy.zeros(shape=(trim_len_target, model_vectors.shape[1]))
    
    num_model_words_considered = 0
    num_kept_words = 0
    done_trimming = False
    for word, value in model_vocab.items():
        kept_word = False
        if (word in game_vocab):
            kept_word = True
            trimmed_model_vocab[word] = value
            trimmed_model_vocab[word].index = num_kept_words
            trimmed_model_vectors[num_kept_words,:] = model_vectors[num_model_words_considered,:]
            done_trimming = (num_kept_words >= trim_len_target)
            num_kept_words = num_kept_words + 1
        num_model_words_considered = num_model_words_considered + 1
        if ((kept_word and num_kept_words % 1000 == 0) or num_model_words_considered % 100000 == 0 or done_trimming):
            print('Kept ' + str(len(trimmed_model_vocab)) + ' model words out of ' + str(num_model_words_considered))
        if done_trimming:
            break
    
    # If we didn't manage to keep as many words as we'd like, reshape trimmed_model_vectors.
    if (num_kept_words < trim_len_target):
        trimmed_model_vectors = trimmed_model_vectors[0:num_kept_words,:]
        
    return trimmed_model_vocab, trimmed_model_vectors

if __name__ == '__main__':
    print('Reading game vocabulary...')
    game_vocab = read_vocabulary()
    print('Done reading game vocabulary! [' + str(len(game_vocab)) + ' words]')

    # On windows, you will need to be running 64-bit python to be able to load that large a model.
    print("Loading word2vec model...")
    model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    print('Done loading word2vec model! [' + str(len(model.vocab)) + ' words]')
    
    print('Trimming model down to game vocabulary...')
    trimmed_model = model
    trimmed_model.vocab, trimmed_model.vectors = get_trimmed_model(game_vocab, model.vocab,  model.vectors, trim_len_target = int(1.0 * len(game_vocab)))
    print('Done trimming model! [' + str(len(trimmed_model.vocab)) + ' words]')
    
    print('Saving trimmed model...')
    trimmed_model.save_word2vec_format('./data/GoogleNews-trimmed-word2vec-negative300.bin', binary=True)
    print('Saved trimmed model!')
    