from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd

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
        print("Pass" + str(i))
        desc_d2v = Doc2Vec(documents=descriptions, vector_size=n_dim, window=window, alpha=alpha, min_alpha=min_alpha, min_count=min_count, dm=dm, dbow_words=dbow_words, epochs=epochs, workers=workers)
        iterations[f'iter{i}'] = desc_d2v
        if save_by_iteration:
            file_name = f'iter{i}_d2v.pkl'
            desc_d2v.save(file_name)
    return iterations

def produce_subspace_vector(positive, negative, model, to_normalize=False):
    psum = np.zeros(model[0].shape[0])
    nsum = np.zeros(model[0].shape[0])

    for p in positive:
        if to_normalize:
            psum += normalize(np.expand_dims(model.wv[p], axis=0)).squeeze(0)
        else:
            psum += model.wv[p]
    if len(positive) > 1:
        psum /= len(positive)

    for n in negative:
        if to_normalize:
            nsum += normalize(np.expand_dims(model.wv[n], axis=0)).squeeze(0)
        else:
            nsum += model.wv[n]
    if len(negative) > 1:
        nsum /= len(negative)

    return psum - nsum

def partisan_associations(iterations, pairs_words, ids, to_normalize=False):
    """
    iterations: a dictionary of iterations
    pairs_words: a list of lists; each individual list is of length 2 and consists of a list of positive words and a list of negative words. Either of the lists can be of length 0.
    ids: a list of document ids, used to construct the TaggedDocument
    """
    PA_scores = []

    for i in range(len(iterations)):
        iteration_name = f'iter{i}'
        mod = iterations[iteration_name]
        docvecs = [mod.dv[x] for x in ids]
        df_iteration = pd.DataFrame(ids, columns=['id'])

        for pair in pairs_words:
            subspace = produce_subspace_vector(positive=pair[0], negative=pair[1], model=mod, to_normalize=to_normalize)
            subspace = np.expand_dims(subspace, axis=0)
            cs = cosine_similarity(docvecs, subspace)
            variable_name_pos = ''.join(pair[0])
            variable_name_neg = ''.join(pair[1])
            variable_name = '_'.join([variable_name_pos,variable_name_neg])
            df_iteration[variable_name] = cs.squeeze(axis=1)

        PA_scores.append(df_iteration)

    return PA_scores

def averaging_pa_scores(PA_scores):
    PA_scores_concat = pd.concat(PA_scores)
    byid = PA_scores_concat.groupby('id')
    PA_scores_mean = byid.mean()
    PA_scores_mean = PA_scores_mean.reset_index()

    return PA_scores_mean

def obtain_pa_scores(descriptions, pairs_words, total_passes=10, n_dim=150, epochs=100, window=10, alpha=0.03, min_alpha=0.03, workers=2, min_count=8, dm=0, dbow_words=1, save_by_iteration=True):
    iterations = word_embeddings_spaces(descriptions=descriptions, total_passes=total_passes, n_dim=n_dim, epochs=epochs, window=window, alpha=alpha, min_alpha=min_alpha, workers=workers, min_count=min_count, dm=dm, dbow_words=dbow_words, save_by_iteration=save_by_iteration)
    ids = []
    for i in range(len(descriptions)):
        ids.append(descriptions[i].tags[0])
    pa_scores = partisan_associations(iterations=iterations, pairs_words=pairs_words, ids=ids)
    final_pa_scores = averaging_pa_scores(PA_scores=pa_scores)
    return final_pa_scores
