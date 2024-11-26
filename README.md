Text Mining project



Topic classification notes:


Word2Vec vectorization:

MultiOutput(GaussianNB)            #f1_score: 0.34
MultiOutput(LogisticRegression)  #f1_score: 0.45 on reviews_rows / f1_score: 0.40
MultiOutput(RandomForestClassifier)  #f1_score: 0.34

Doc2vec vectorization:

most macro f1_scores below 0.20
so we should not use doc2vec

TODO:
sprawdzic lematyzacje meali, lowercase, distinct values meals
sprawdzic tfidf input the matrix from 09-TM class

