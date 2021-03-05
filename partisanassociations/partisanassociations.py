from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

def word_embeddings_spaces(descriptions, total_passes=10, n_dim=150, epochs=100, window=10, alpha=0.03, min_alpha=0.03, workers=2, min_count=8, dm=0, dbow_words=1, save_by_iteration=True):
    '''
    descriptions: the descriptions as a TaggedDocument
    total_passes: number of times to fit the word embedding space
    n_dim: size of the word and document vectors
    epochs: number of passes through the data
    window: window size
    alpha: initial training rate
    min_alpha: lowest training rate; learning rate will linearly drop to min_alpha
    workers: worker threads
    min_count: ignores all words with total frequency lower than this
    dm: if dm=1, then distributed memory (DM) is used; otherwise, distributed bag-of-words (DBOW) is used
    dbow_words: if set to 1, will simultaneously train word vectors (in a skip-gram fashion) simultaneous with DBOW
    save_by_iteration: if True, saves the doc2vec object each time
    '''
    iterations = {}
    for i in range(0,total_passes):
        desc_d2v = Doc2Vec(documents=descriptions, vector_size=n_dim, window=window, alpha=alpha, min_alpha=min_alpha, min_count=min_count, dm=dm, dbow_words=dbow_words, epochs=epochs, workers=workers)
        iterations[f'iter{i}'] = desc_d2v
        if save_by_iteration:
            file_name = f'iter{i}_d2v.pkl'
            desc_d2v.save(file_name)
    return iterations
