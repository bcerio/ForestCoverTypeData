#!/usr/bin/env python

import sys
import os
import random
from operator import itemgetter

reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import numpy as np

from sklearn import cross_validation
from sklearn import grid_search
from sklearn import ensemble
from sklearn import metrics

sys.path.append(os.path.join(os.environ['ROOTDIR'],'shared_tools'))
from logging_tools import Logger

logger = Logger()

def main():

    X_fields = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
    X_fields += map(lambda x: 'Wilderness_Area_%s' % x,range(4))
    X_fields += map(lambda x: 'Soil_Type_%s' % x,range(40))
    
    y_fields = ['Cover_Type']

    headers = X_fields + y_fields

    df = pd.read_csv('covtype.data',names=headers,header=None)

    X = df[X_fields].values
    y = np.ravel(df[y_fields].values)

    # split sample into random subsets for training and testing
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.55,random_state=42)
    #################################

    # use *exactly* the same training and test samples as https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info
    X_train = X[0:11340+3780]
    y_train = y[0:11340+3780]

    X_test = X[11340+3780:]
    y_test = y[11340+3780:]
    ################################

    test_fold = [-1]*11340+[0]*3780
    ps = cross_validation.PredefinedSplit(test_fold)

    logger.log('Shape of training sample: %s' % str(X_train.shape))
    logger.log('Shape of test sample: %s' % str(X_test.shape))

    classifier = ensemble.RandomForestClassifier(verbose=0)
    
    parameters = {'n_estimators':[25,50,100,250,500],'max_depth':[2,5,None],'criterion':['gini','entropy']}
    opt_classifier = grid_search.GridSearchCV(classifier,parameters,verbose=1,n_jobs=8,cv=ps)

    opt_classifier.fit(X_train,y_train)

    logger.log('Optimal hyperparameters: %s' % opt_classifier.best_params_)
    logger.log('Best score: %s' % opt_classifier.best_score_)
    logger.log('Score on hold-out: %s' % opt_classifier.score(X_test,y_test))
    logger.log('Accuracy score on hold-out: %s' % metrics.accuracy_score(y_test,opt_classifier.best_estimator_.predict(X_test)))    

    for i in range(10):
        rand_indy = random.randint(0,X_test.shape[0]-1)
        logger.log('Predicted class: %s' % opt_classifier.predict([X_test[rand_indy]]))
        logger.log('Actual class: %s' % y_test[rand_indy])

    logger.log('Feature importance')
    logger.log(' ')
    fi = sorted([(headers[i],imp) for i,imp in enumerate(opt_classifier.best_estimator_.feature_importances_)],key=itemgetter(1),reverse=True)
    for feat_imp in fi:
        logger.log('%s: %s' % feat_imp)
        
    return 1

if __name__ == '__main__':
    main()

