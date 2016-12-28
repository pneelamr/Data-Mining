# Author: Padma Neelamraju <pneelamr@gmail.com>
# Sample taken from http://scikit-learn.org

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt



from sklearn.model_selection import GridSearchCV
import logging
import warnings

warnings.filterwarnings("ignore")

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

def __trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 200 else s[:197] + "..."

def train(X_train, y_train, X_test, y_test, feature_names, filesuffix, opts, args):
    # Start benchmark definition
    def __benchmark(clf, c = 0):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)

        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test,pred, average = 'weighted')

        print("precision:   %0.3f" % precision)
        print("recall:   %0.3f" % recall)
        print("fscore:   %0.3f" % fscore)
        #print("support:   " % support)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d %d" % clf.coef_.shape)
            print("density: %f" % density(clf.coef_))

            if opts.print_top10 and feature_names is not None:
                print("top 10 keywords per class:")
                for i, label in enumerate(target_names):
                    try:
                        top10 = np.argsort(clf.coef_[i])[-10:]
                        print(__trim("%s: %s" % (label, " ".join(feature_names[top10]))))
                        #print()
                    except:
                        abc=1
                        #print ("error in label :" + label)
                        #print ("Unexpected error:", sys.exc_info()[0])
                        #raise

        if opts.print_report:
            print("classification report:")
            print(metrics.classification_report(y_test, pred))
            #,                                            target_names=target_names))




        print()
        clf_descr = str(clf).split('(')[0]

        if opts.print_cm:
            if clf_descr=='LinearSVC':

                print("confusion matrix:")
                cmlsvcl2 = metrics.confusion_matrix(y_test, pred)
                print(cmlsvcl2)
                if c == 1:
                    plotCM(cmlsvcl2,filesuffix)
                c+=1


        return clf_descr, score, train_time, test_time, precision, recall, fscore, pred

    #### End benchmark definition

    target_names = ["0","1"]

    if opts.select_chi2:
        print("Extracting %d best features by a chi-squared test" %opts.select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        if feature_names:
            # keep selected feature names
            feature_names = [feature_names[i] for i
                             in ch2.get_support(indices=True)]
        print("done in %fs" % (time() - t0))
        print()

    if feature_names:
        feature_names = np.asarray(feature_names)

    results = []
    benchmark = []
    preds = []


    # PARAMETERS
    #NB
    # defauls
    palpha = 0.01
    c = 1.0
    fitint = True
    nn = 10
    ne = 100
    maxfeat = 100
    #options override

    if opts.alpha:
        palpha = opts.alpha
    if opts.c:
        c = opts.c
    if opts.fitint:
        fitint = opts.fitint
    if opts.nn:
        nn = opts.nn
    if opts.ne:
        ne = opts.ne
    if opts.numfeatures:
        import math
        maxfeat = int(math.sqrt(opts.numfeatures))
    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes - MultinomialNB")
    benchmark = __benchmark(MultinomialNB(alpha=palpha))
    results.append(benchmark[0:7])
    preds.append(benchmark[7])

    print("Naive Bayes - BernoulliNB")
    benchmark = __benchmark(BernoulliNB(alpha=palpha))
    results.append(benchmark[0:7])
    preds.append(benchmark[7])

    print ("Linear SVC - L1 penalty")
    print('=' * 80)
    # Train Liblinear model
    benchmark =__benchmark(LinearSVC(C = c, loss='l2', penalty='l1',dual=False, tol=1e-3, fit_intercept = fitint))
    results.append(benchmark[0:7])
    preds.append(benchmark[7])

    print ("Linear SVC - L2 penalty")
    benchmark =__benchmark(LinearSVC(C = c, loss='l2', penalty='l2',dual=False, tol=1e-3, fit_intercept = fitint),c=1)
    results.append(benchmark[0:7])
    preds.append(benchmark[7])

    for clf, name in (
            #(RidgeClassifier(tol=0.001, solver="auto", fit_intercept =False), "Ridge Classifier"), # lsqr => auto, fit intercept added
            #(Perceptron(n_iter=50), "Perceptron"),
            #(PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=nn), "kNN"),
            (RandomForestClassifier(n_estimators=ne, max_features = maxfeat), "Random forest")):
        print('=' * 80)
        print(name)
        benchmark = __benchmark(clf)
        print ("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print (benchmark[0:7])
        print ("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        results.append(benchmark[0:7])
        preds.append(benchmark[7])



    # Train NearestCentroid without threshold
    print('=' * 80)
    print("NearestCentroid (aka Rocchio classifier)")
    benchmark = __benchmark(NearestCentroid())
    results.append(benchmark[0:7])
    preds.append(benchmark[7])



    '''print('=' * 80)
    print("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    benchmark = __benchmark(Pipeline([('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
                                ('classification', LinearSVC())]))
    results.append(benchmark[0:7])
    preds.append(benchmark[7])'''
    #('MultinomialNB', 'BernoulliNB', 'LinearSVC(l1)','LinearSVC(l2)', 'SGD(l1)', 'SGD(l2)', 'SGD(en)', 'Ridge', 'Perceptron',
     #'PassiveAggressive', 'KNN', 'NearestCentroid'),

    ## FOR REPORT TAKE THE CORRECT NAMES
    #print(results)
    return results, preds

    #plotall(results,filesuffix)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        #print (height)
        plt.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%0.03f' % height,
                ha='center', va='bottom')

def plotall (results, filesuffix):
    print ("Plotting metrics")
    indices = np.arange(len(results))
    print (len(results))
    results = [[x[i] for x in results] for i in range(7)]
    #clf_names, score, training_time, test_time = results
    #training_time = np.array(training_time) / np.max(training_time)
    #test_time = np.array(test_time) / np.max(test_time)
    clf_names, score,training_time, test_time , precision, recall, fscore = results
    #print (clf_names)
    fig = plt.subplot(221)
    width = 0.35
    rects1 = plt.bar(indices, score,  width= width, label="Accuracy Score", color='Gray') # coral
    autolabel(rects1)
    plt.legend(loc='center right')
    plt.xticks(indices,

     rotation = 90, ha ='left', va='top')

    fig2 = plt.subplot(223)
    rects2 = plt.bar(indices, precision,  width= width, label="Avg Precision", color='DarkSeaGreen') # dark green
    plt.legend(loc='center right')
    autolabel(rects2)
    plt.xticks(indices,
    ('MultinomialNB', 'BernoulliNB', 'LinearSVC(l1)','LinearSVC(l2)', 'PassiveAggressive', 'KNN', 'RandomForest','NearestCentroid'), rotation = 90, ha ='left', va='top')

    fig3 = plt.subplot(224)
    rects3 = plt.bar(indices, recall,  width= width, label="Avg Recall", color='SteelBlue')#dark turq
    plt.legend(loc='center right')
    autolabel(rects3)
    plt.xticks(())

    fig4 = plt.subplot(222)
    rects4 = plt.bar(indices, fscore,  width= width, label="F1-Score", color='Tan')#hotpin
    plt.legend(loc='center right')
    autolabel(rects4)
    plt.xticks(())

    plt.subplots_adjust(wspace = 0.2, hspace = 0)
    plt.savefig('all_metrics_'+filesuffix+'.png')
    #plt.show()

def plotF (results, filesuffix):
    print ("Plotting Fscore")
    for x in results:
        print (x[0])


    indices = np.arange(len(results))
    f_score = [x[6] for x in results]
    fig, ax = plt.subplots()

    plt.figure(figsize=(10,8))
    plt.title("Algorithm Performance on Cross Validation Set - Unbalanced")


    width = 0.35
    rects1 = plt.bar(indices, f_score,  width= width, label="F1 Score", color='LightSeaGreen') # coral
    autolabel(rects1)
    plt.legend(loc='center right')
    plt.xticks(indices,
    ('MultinomialNB', 'BernoulliNB', 'LinearSVC(l1)','LinearSVC(l2)', 'KNN', 'RandomForest','NearestCentroid'),
     rotation = -45, ha ='left', va='top')
    #plt.text("Cross Validation Data",  size=14, color = 'white')
    plt.tight_layout()
    #plt.subplots_adjust(top = 2)
    plt.savefig('FScore_'+filesuffix+'.png')

def plotCM(cm,filename):
    import itertools

    print ("Plotting confusion matrix")
    cmap=plt.cm.Greens
    normalize=False
    title='Confusion matrix'
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = ["0","1"]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    plt.savefig("confusion_matrix.png")
