#!/usr/bin/python

import sys
import pickle
# sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### possible features:  ( from print my_dataset[my_dataset.keys()[0]].keys() )
###   salary,  deferral_payments, total_payments, exercised_stock_options, bonus, restricted_stock, director_fees, long_term_incentive,
###   restricted_stock_deferred, total_stock_value, expenses, loan_advances, deferred_income,
###   to_messages, from_messages, other, from_this_person_to_poi, poi, email_address, from_poi_to_this_person, shared_receipt_with_poi
features_list = ['poi','salary', 'bonus', 'total_stock_value','long_term_incentive', 'director_fees',\
'restricted_stock_deferred', 'expenses', 'loan_advances', 'deferred_income',\
'deferral_payments',  'exercised_stock_options', 'bonus', 'restricted_stock']#,'restricted_stock_options', 'bonus','total_payments', 'exercised_stock_options'] # You will need to use more features



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# removing unnecessary data; 'THE TRAVEL AGENCY IN THE PARK', 'TOTAL' and all associated values
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


####################### feature exploration #######################
###################################################################
### print how many of the datapoints have useful information (non-zero, non-NaN)
import numpy as np
# complete list of monetary features included in dict
keys = ['salary', 'bonus', 'total_stock_value','long_term_incentive', 'director_fees',\
'restricted_stock_deferred', 'expenses', 'loan_advances', 'deferred_income',\
'deferral_payments',  'exercised_stock_options', 'bonus', 'restricted_stock']
# 'to_messages', 'from_messages', 'other','from_this_person_to_poi','poi',\
# 'email_address', 'from_poi_to_this_person','shared_receipt_with_poi']

total_persons = float(len(my_dataset.keys()))
print "\nnumber of entries present out of:", total_persons

total_poi = sum([1 for person in my_dataset.keys() if my_dataset[person]['poi']])
print 'total_poi:', total_poi


def get_metrics(info_values):
    info_q1, info_med, info_q3, info_max = np.percentile(info_values, [25, 50, 75, 100])
    return "\t\tfor {0}:\n\t\t\t[q1, q3]: {1}\n\t\t\tmedian value: {2}\n\t\t\tmaximum value: {3}".format( \
            key, [info_q1, info_q3], info_med, info_max )

# loop through all features to look for interesting/complete ones
for key in keys:
    # list comprehension; key value of each person who has an informational (non-zero, non-NaN) value of key
    # info_values = [my_dataset[person][key] for person in my_dataset.keys() \
    #     if (my_dataset[person][key] != 0) and (my_dataset[person][key] != 'NaN')]
    
    poi_counter = 0
    all_info_values, other_info_values, poi_info_values = [], [], []
    for person in my_dataset.keys():
        if (my_dataset[person][key] != 0) and (my_dataset[person][key] != 'NaN'):
            all_info_values.append(my_dataset[person][key])
            if my_dataset[person]['poi']:
                poi_info_values.append(my_dataset[person][key])
                poi_counter += 1
            else:
                other_info_values.append(my_dataset[person][key])

    num_info = len(all_info_values)   # keep track of how mnay informational values there are
    percent_info = round(num_info/total_persons*100,1)   # calculate percent of values are informational
    print "\t{0}: {1} == {2}%".format(key, num_info, percent_info)
    print "\t\tnumber of POIs present:", poi_counter

    # print key's quartiles if at least X% of the keys are informational
    if percent_info > 65:
        for info in [('ALL', all_info_values), ('OTHER', other_info_values), ('POI', poi_info_values)]:
            print "\t\t%s" % info[0]
            print get_metrics(info[1])

print "\n"

"""The most important attributes we look for in our features are the inclusion of
informational values for the the majority of the POIs and others from the data.
From looking at the features with informational values (non-zero,non-Nan), and
comparing the POI to other metrics, it appears that the most divisive features are
salary, exercised_stock_options, and total_stock_value (log?)."""
#################### feature exploration end ######################




### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


##################### feature creation ############################
###################################################################
"""here I have started using PCA to build another feature, it unforunately
only got 85 (next 88, ) percent score after training on the practice set (no test).
additionally, it probably trimms too many features before applying it, as well
as will need to be optimized for these array shapes. goodnight."""
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
# from sklearn.grid_search import GridSearchCV
"""running a PCA on the monetary features seems to get pretty good results (need
to test with KFold to get some consistency)"""

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=.3)
pca = PCA(n_components=5).fit(features_train)
pca_features_train = pca.transform(features_train)
pca_features_test = pca.transform(features_test)
print pca.explained_variance_ratio_

"""I am getting vastly different score, precision and recall scores for my clf,
and so I am going to use KFolds to run multiple iteration and then average them.
I will next try and find supervised PCA to implement.
Unfortunately, it seems that this method is only achieving high accuracy when """
kf = KFold( n_splits=3, shuffle=True)
kf_accuracy, kf_precision, kf_recall = [], [], []
for train_indices, test_indices in kf.split(features):
    # set training and testing features and labels randomly using kfolds
    features_train = [features[ii] for ii in train_indices]
    features_test = [features[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]

    # set PCA fit to training features, and reduce dementionality of test features
    pca = PCA(n_components=5)
    pca_features_train = pca.fit_transform(features_train)
    pca_features_test = pca.transform(features_test)

    # use a decision tree to classify features
    clf = DecisionTreeClassifier()
    clf.fit(pca_features_train, labels_train)

    # predict test features and then assess accuracy, precision, and score
    pred = clf.predict(pca_features_test)
    kf_acc = round( clf.score(pca_features_test, labels_test) , 3)
    kf_pre = round( precision_score(labels_test, pred) , 2)   # sometimes precision will be 0 by the
    kf_rec = round( recall_score(labels_test, pred) , 2)
    print "percentage poi in lables_test", round( 100 * sum(labels_test)/float(len(labels_test)) , 2)
    print "\taccuracy:", kf_acc
    print "\tprecision:", kf_pre, "  recall: ", kf_rec

    # append metric values for each run into a list and then average values after loop
    kf_accuracy.append(kf_acc)
    if kf_pre != 0:  # sometimes precision will be 0 by the KFolds separations
        kf_precision.append(kf_pre)
        kf_recall.append(kf_rec)

print "after KFolds, overall accuracy, precision, and recall are:\n\t%0.3f, %0.2f, %0.2f."\
     % ( np.average(kf_accuracy), np.average(kf_precision), np.average(kf_recall) )
##################### feature creation end ########################



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.predict

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split   # import train_test_split to separate data into training and testing groups

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
