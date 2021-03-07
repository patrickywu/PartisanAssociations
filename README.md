# Partisan Associations

This Python package aims to estimate Twitter users' partisan associations given partisan keywords and Twitter users' bios. The idea of this approach is to learn the non-partisan words that are in the contextual neighborhoods of explicitly partisan words. Even if someone does not explicitly use partisan expressions in their bio, he or she may describe themselves with words that the descriptions that feature explicit partisan expressions tend to contain. This idea resonates with research that studies the associations between partisan sentiments and seemingly non-partisan identities, activities, hobbies, spending habits, and interests.

This is accomplished using word2vec, a word embeddings method that maps words to fixed-dimensional continuous vectors. Word vectors close to each other indicate closer meanings. Documents can be mapped in a similar fashion to vectors in the same space as word vectors. Specifically, we map user bios to document embeddings using doc2vec and we map individual words to word embeddings using word2vec. We then take the cosine similarity between these document embeddings and specific partisan subspaces defined using partisan keywords that refer to presidential campaigns, candidates, parties, and slogans to calculate partisan associations.

To use the functions in this package, install the package first. To do this, navigate to the directory you would like to install the files in. Then,

```
git clone https://github.com/patrickywu/PartisanAssociations
cd PartisanAssociations
pip install -r requirements.txt
python setup.py develop
```

From this, you can import the functions in the package. 
