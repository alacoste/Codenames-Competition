import gensim
import tensorflow as tf
import tensorflow_hub as hub
import numpy
import gloomy_helpers

def similarity(model, word1, word2):
    print('Similarity of %s and %s: %.3f' % (word1, word2, model.similarity(word1, word2)))

if __name__ == '__main__':
    print('Reading game vocabulary...')
    game_vocab, game_words = gloomy_helpers.get_game_vocabulary()
    print('Done reading game vocabulary! [%d words]' % len(game_vocab))
    
    print('Loading tensorflow \'universal sequence encoder\' hub module...')
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
    print('Loaded module.')
    
    print('Vectorizing the game vocabulary...')
    tf.logging.set_verbosity(tf.logging.ERROR)  # Reduce logging output.
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(embed(game_vocab))
        print('Done vectorizing, got an array of shape: (%d, %d)' % embeddings.shape)
        num_vectors = embeddings.shape[0]
        vector_size = embeddings.shape[1]
        assert num_vectors == len(game_vocab)
        
        print('Saving vectorization in a gensim word2vec model...')
        model = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size)
        model.vectors = embeddings
        for i in range(0, num_vectors):
            model.vocab[game_vocab[i]] = gensim.models.keyedvectors.Vocab()
            model.vocab[game_vocab[i]].index = i
        print('Checking some similarities:')
        similarity(model, 'australia', 'america')
        similarity(model, 'australia', 'kangaroo')
        similarity(model, 'america', 'kangaroo')
        model.save_word2vec_format('./data/semantris.bin', binary=True)
        print('Saved and exported model!')