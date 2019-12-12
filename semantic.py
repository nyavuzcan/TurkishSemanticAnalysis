import pandas as pd
import numpy as np
import glob
import re
import os
from snowballstemmer import TurkishStemmer
from stop_words import get_stop_words
from sklearn.model_selection import train_test_split
import turkishnlp
from turkishnlp import detector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


obj = detector.TurkishNLP()  # TurkishNLP dosyalarının  indirilmesi
# obj.download()
obj.create_word_set()

turkStem = TurkishStemmer()
stop_words = get_stop_words('turkish')  # stop_words array


def read_txt(folder_name):  # read ve data preprocessing işlemleri
    file_list = glob.glob(os.path.join(os.getcwd(), str(folder_name), "*.txt"))

    cumle_list = []  # 35, 35,35 dosyanın içeriklerini tutan array
    tip_list = []  # p,n,t tutan array 105 tane

    for file_path in file_list:
        with open(file_path) as f_input:

            word_list = f_input.read().split()  # tüm txt dosyalarının içindeki cümlelerin kelimelerini boşluklara ayırarak listede tutuyor

            sentence_list = []
            for i in word_list:
                if len(i) != 0:  # missing values???

                    a = obj.auto_correct(obj.list_words(i))  # yazım hatalarını düzeltip a'ya atıyoruz (typos)
                    if len(a) != 0:

                        a = ''.join([i for i in a if i.isalpha()])  # alfabetik olmayan sembolleri siliyor a'ya atıyor

                        a = turkStem.stemWord(a)  # çekim eklerini atıp köklerine ayırarak a'ya atıyoruz

                        # print(a)
                        if a not in stop_words:
                            if len(a) > 2:
                                sentence_list.append(a)  # kelimelerin yazım hataları ve ekleri kaldırarak sentence_list'te tutuyoruz. Daha sonra tekrar cümleye çevrilecek
                    # print("f_input:",f_input.read())
                    # print("sentence_list:",sentence_list)
            x = ' '.join(word for word in sentence_list)  # x= kelimelerin arasına boşluk bırakarak tekrar cümleyen çeviriyor
            cumle_list.append(x)  # data olarak kullanılacak cümleler var

            if folder_name == "negatif":
                tip_list.append(int(-1))
            elif folder_name == "pozitif":
                tip_list.append(int(1))
            elif folder_name == "tarafsiz":
                tip_list.append(int(0))

    corpus = {  # dictionary. key-value(örnek: 'cumle' ,cumle_list)
        'cumle': cumle_list,
        "tip": tip_list
    }

    return pd.DataFrame.from_dict(corpus)  # DataFrame çeviriyoruz


negatif = read_txt("negatif")
pozitif = read_txt("pozitif")
tarafsiz = read_txt("tarafsiz")

all_data = pd.concat([negatif, pozitif, tarafsiz],ignore_index=True)  # 3farklı dataframeyi tek dataframe yapıyor concat ile

X_train, X_test, y_train, y_test = train_test_split(all_data['cumle'], all_data['tip'], test_size=0.33, random_state=42)
# print(X_train.head())
# print(X_train.head())
# print('\n\nX_train shape: ', X_train.shape)





from sklearn.metrics import mean_squared_error

# from sklearn.feature_extraction.text import CountVectorizer
# vect = CountVectorizer(encoding ='iso-8859-9').fit(X_train)
# X_train_vectorized = vect.transform(X_train)



from sklearn.feature_extraction.text import TfidfVectorizer  # tfidf ile kelimelri sasyısallaştırıyor. modellerde kullanabşlmek için.

vect = TfidfVectorizer(min_df=6).fit(X_train)
X_train_vectorized = vect.transform(X_train)

# vect = CountVectorizer(min_df = 2, ngram_range = (1,2)).fit(X_train)
# X_train_vectorized = vect.transform(X_train)



# ------------LOGISTIC REGRESSION------------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

model_log = LogisticRegression()
model_log.fit(X_train_vectorized, y_train)
predictions_log = model_log.predict(vect.transform(X_test))
print("Logistic Regression Cross Validation Score:", cross_val_score(estimator=model_log, X=X_train_vectorized, y=y_train, cv=10, n_jobs=1).mean())
print('Logistic Regression Accuarcy: ', accuracy_score(y_test, predictions_log))
print("Logistic Regression RMS:", mean_squared_error(y_test, predictions_log))
print("Logistic Regression Precission Score:", precision_score(y_test, predictions_log, average=None).mean())
print("Logistic Regression Recall Score:", recall_score(y_test, predictions_log, average=None).mean())
print("Logistic Regression F1-Score:",f1_score(y_test, predictions_log, average=None).mean())

# -------------DecisionTree--------------
from sklearn import tree
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

clf = tree.DecisionTreeClassifier()
clf.fit(X_train_vectorized, y_train)
predictions_dec = clf.predict(vect.transform(X_test))
print("Decision Tree Classifier Validation Score:", cross_val_score(estimator=clf, X=X_train_vectorized, y=y_train, cv=10, n_jobs=1).mean())
print('Decision Tree Classifier Accuarcy: ', accuracy_score(y_test, predictions_dec))
print("Decision Tree Classifier RMS:", mean_squared_error(y_test, predictions_dec))
print("Decision Tree Classifier Precission Score:", precision_score(y_test, predictions_dec, average=None).mean())
print("Decision Tree Classifier Recall Score:", recall_score(y_test, predictions_dec, average=None).mean())
print("Decision Tree Classifier F1-Score:",f1_score(y_test, predictions_dec, average=None).mean())


# --------RandomForest-----------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

clf_random = RandomForestClassifier(random_state=0)
clf_random.fit(X_train_vectorized, y_train)
predictions_ran = clf_random.predict(vect.transform(X_test))
print("Random Forest Classifier Cross Validation Score:", cross_val_score(estimator=clf_random, X=X_train_vectorized, y=y_train, cv=10, n_jobs=1).mean())
print('Random Forest Classifier Accuarcy: ', accuracy_score(y_test, predictions_ran))
print("Random Forest Classifier RMS:", mean_squared_error(y_test, predictions_ran))
print("Random Forest Classifier Precision Score:", precision_score(y_test, predictions_ran, average=None).mean())
print("Random Forest Classifier Recall Score:", recall_score(y_test, predictions_ran, average=None).mean())
print("Random Forest Classifier F1-Score Score:", f1_score(y_test, predictions_ran, average=None).mean())


# ---------XGB----------
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

model = XGBClassifier()
model.fit(X_train_vectorized, y_train)
predictions_xg = model.predict(vect.transform(X_test))
print("XGBoost Classifier Cross Validation Score:", cross_val_score(estimator=model, X=X_train_vectorized, y=y_train, cv=10, n_jobs=1).mean())
print('XGBoost Classifier Accuarcy: ', accuracy_score(y_test, predictions_xg))
print("XGBoost Classifier RMS:", mean_squared_error(y_test, predictions_xg))
print("XGBoost Classifier Precision Score:", precision_score(y_test, predictions_xg, average=None).mean())
print("XGBoost Classifier Recall Score:", recall_score(y_test, predictions_xg, average=None).mean())
print("XGBoost Classifier F1-Score Score:", f1_score(y_test, predictions_xg, average=None).mean())


# ------------K-NN----------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

clf_knn = KNeighborsClassifier()
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train_vectorized, y_train)
predictions_knn = clf_knn.predict(vect.transform(X_test))
print("K-NN Classifier Cross Validation Score:", cross_val_score(estimator=clf_knn, X=X_train_vectorized, y=y_train, cv=10, n_jobs=1).mean())
print('K-NN Classifier Accuarcy: ', accuracy_score(y_test, predictions_knn))
print("K-NN Classifier RMS:", mean_squared_error(y_test, predictions_knn))
print("K-NN Classifier Precision Score:", precision_score(y_test, predictions_knn, average=None).mean())
print("K-NN Classifier Recall Score:", recall_score(y_test, predictions_knn, average='weighted'))
print("K-NN Classifier F1-Score Score:", f1_score(y_test, predictions_knn, average=None).mean())


# ---------SVM----------
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

clf_svm = SVC(kernel='linear', gamma='scale')
clf_svm.fit(X_train_vectorized, y_train)
predictions_svm = clf_svm.predict(vect.transform(X_test))
print("SVM Classifier Cross Validation Score:", cross_val_score(estimator=clf_svm, X=X_train_vectorized, y=y_train, cv=10, n_jobs=1).mean())
print('SVM Classifier Classifier Accuarcy Score: ', accuracy_score(y_test, predictions_svm))
print("SVM Classifier RMS:", mean_squared_error(y_test, predictions_svm))
print("SVM Classifier Precision Score:", precision_score(y_test, predictions_svm, average=None).mean())
print("SVM Classifier Recall Score:", recall_score(y_test, predictions_svm, average=None).mean())
print("SVM Classifier F1-Score Score:", f1_score(y_test, predictions_svm, average=None).mean())


# ------NAIVE BAYES---------
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


clf_bern_bayes = BernoulliNB(binarize=True)
clf_bern_bayes.fit(X_train_vectorized, y_train)
predictions_bayes_bern = clf_bern_bayes.predict(vect.transform(X_test))
print("Bernoulli NB Cross Validation Score:", cross_val_score(estimator=clf_bern_bayes, X=X_train_vectorized, y=y_train, cv=10, n_jobs=1).mean())
print('Bernoulli NB Classifier Accuarcy: ', accuracy_score(y_test, predictions_bayes_bern))
print("Bernoulli NB RMS:", mean_squared_error(y_test, predictions_bayes_bern))
print("Bernoulli NB Precision Score:", precision_score(y_test, predictions_bayes_bern, average=None).mean())
print("Bernoulli NB Recall Score:", recall_score(y_test, predictions_bayes_bern, average=None).mean())
print("Bernoulli NB F1-Score Score:", f1_score(y_test, predictions_bayes_bern, average=None).mean())


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


clf_multi_bayes = MultinomialNB()
clf_multi_bayes.fit(X_train_vectorized, y_train)
predictions_bayes_multi = clf_multi_bayes.predict(vect.transform(X_test))
print("Multinomial NB Cross Validation Score:", cross_val_score(estimator=clf_multi_bayes, X=X_train_vectorized, y=y_train, cv=10, n_jobs=1).mean())
print('Multinomial NB Classifier ACC: ', accuracy_score(y_test, predictions_bayes_multi))
print("Multinomial NB RMS:", mean_squared_error(y_test, predictions_bayes_multi))
print("Multinomial NB Precision Score:", precision_score(y_test, predictions_bayes_multi, average=None).mean())
print("Multinomial NB Recall Score:", recall_score(y_test, predictions_bayes_multi, average=None).mean())
print("Multinomial NB F1-Score Score:", f1_score(y_test, predictions_bayes_multi, average=None).mean())



# gaussian bayes?????????


#-------K-MEANS--------------

from sklearn.cluster import KMeans
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


clf_kmeans = KMeans(n_clusters=2, random_state=0)
clf_kmeans.fit(X_train_vectorized, y_train)
predictions_kmeans = clf_kmeans.predict(vect.transform(X_test))
print("K-MEANS Cross Validation Score:", cross_val_score(estimator=clf_kmeans, X=X_train_vectorized, y=y_train, cv=10, n_jobs=1).mean())
print('K-MEANS Classifier Accuarcy: ', accuracy_score(y_test, predictions_kmeans))
print("K-MEANS RMS:", mean_squared_error(y_test, predictions_kmeans))
print("K-MEANS Precision Score:", precision_score(y_test, predictions_kmeans, average=None).mean())
print("K-MEANS Recall Score:", recall_score(y_test, predictions_kmeans, average=None).mean())
print("K-MEANS F1-Score Score:", f1_score(y_test, predictions_kmeans, average=None).mean())




def kontrol(string):
    data = np.array([str(string)])
    ser = pd.Series(data)
    return clf.predict(vect.transform(ser))