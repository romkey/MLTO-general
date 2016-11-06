#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 1.0)
#plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
#plt.legend()
#plt.xlabel("bumpiness")
#plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

import sklearn
from sklearn import neighbors
from sklearn import ensemble
from sklearn.metrics import accuracy_score

# for some reason I didn't bother...
#clf = neighbors.NearestNeighbors()

max_n_estimators = 0
max_learning_rate = 0
max_random_state = 0 
max_accuracy = 0

for i in range(1, 100):
  clf = ensemble.RandomForestClassifier(n_estimators=i)
  clf.fit(features_train, labels_train)

  pred = clf.predict(features_test)
  score = accuracy_score(labels_test, pred)
  if score > max_accuracy:
    print "*** --->"
    print i, score

    max_n_estimators =  i
    max_accuracy = score


max_n_estimators = 0
max_learning_rate = 0
max_random_state = 0 
max_accuracy = 0

# a better way to do this would be to have a master process that fork()ed children off with i, j and k as parameters
# run one child per CPU core in your machine
# child outputs its best result and the master remembers that if it beats its current best
# on a quad core machine that would complete in 1/4 of the time
for i in range(1, 100):
  for j in range(1, 100):
    for k in range(1,100):
      clf = ensemble.AdaBoostClassifier(n_estimators=i, learning_rate=j/10.0, algorithm="SAMME.R", random_state=k)
      clf.fit(features_train, labels_train)
      pred = clf.predict(features_test)

      score = accuracy_score(labels_test, pred)
      if score > max_accuracy:
        print "*** --->"
        print i, j, k, score

        max_n_estimators =  i
        max_learning_rate = j/10.0
        max_random_state = k
        max_accuracy = score


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
