import gensim
import numpy

def read_vocabulary():
    with open('./data/vocab.txt') as f:
        return [line.strip() for line in f]
        
def get_trimmed_model(game_vocab, model_vocab, model_vectors):
    trimmed_model_vocab = {}
    trimmed_model_vectors = numpy.zeros(shape=(len(game_vocab), model_vectors.shape[1]))
    index = 0
    num_words_missing_from_model = 0
    model_vocab_words_list = list(model_vocab)
    for word in game_vocab:
        if ((index + num_words_missing_from_model) % 1000 == 0):
            print('Kept ' + str(index) + ' words out of ' + str(index + num_words_missing_from_model))
        if (not word in model_vocab):
            num_words_missing_from_model = num_words_missing_from_model + 1
            continue
        trimmed_model_vocab[word] = model_vocab[word]
        trimmed_model_vocab[word].index = index
        trimmed_model_vectors[index,:] = model_vectors[model_vocab_words_list.index(word),:]
        index = index + 1
    print(str(num_words_missing_from_model) + ' words from the game vocabulary could not be found in the model')
    
    # If we didn't manage to keep as many words as we'd like, reshape trimmed_model_vectors.
    if (index < trimmed_model_vectors.shape[0]):
        trimmed_model_vectors = trimmed_model_vectors[0:index,:]
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
    trimmed_model.vocab, trimmed_model.vectors = get_trimmed_model(game_vocab, model.vocab,  model.vectors)
    print('Done trimming model! [' + str(len(trimmed_model.vocab)) + ' words]')
    
    print('Saving trimmed model...')
    trimmed_model.save_word2vec_format('./data/GoogleNews-trimmed-word2vec-negative300.bin', binary=True)
    print('Saved trimmed model!')
    