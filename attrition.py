from __future__ import print_function

import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular

feature_names = ['Age','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EducationField','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','Over18','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
data = np.genfromtxt('C:\\Users\\prakh\\Documents\\SMS Innvent 2018\\Innvent 2018\\HR.csv', delimiter=',', dtype=str)

arr=[21,'Travel_Frequently',251,'Research & Development',10,2,'Life Sciences',1,'Female',45,2,1,'Laboratory Technician',3,'Single',2625,25308,1,'Y','No',20,4,3,80,2,2,1,2,2,2,2,0]
sample_test_row = np.array([arr,arr])
sample_test_row = sample_test_row[:,:-1]
labels = data[:,31]
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:,:-1]

categorical_features = [1,3,6,8,12,14,18,19]
categorical_names = {}
le = None
# for feature in categorical_features:
#     le = sklearn.preprocessing.LabelEncoder()
#     le.fit(data[1:, feature])
#     data[1:, feature] = le.transform(data[1:, feature])
#     sample_test_row[1:, feature] = le.transform(sample_test_row[1:, feature])
#     categorical_names[feature] = le.classes_

data = data.astype(float)
sample_test_row = sample_test_row.astype(float)

encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)

np.random.seed(1)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)

encoder.fit(data)
encoded_train = encoder.transform(train)

import xgboost
gbtree = xgboost.XGBClassifier(objective="binary:logistic", n_estimators=300, max_depth=5, learning_rate=0.001)
gbtree.fit(encoded_train, labels_train)
sample = encoder.transform(sample_test_row)
accuracy = sklearn.metrics.accuracy_score(labels_test, gbtree.predict(encoder.transform(test)))
print(accuracy)
prediction = gbtree.predict(encoder.transform(test))
prediction1= gbtree.predict_proba(sample)
print('Turn Over Prediction: {:.3f}'.format(prediction1[0][1]))
predict_fn = lambda x: gbtree.predict_proba(encoder.transform(x)).astype(float)

explainer = lime.lime_tabular.LimeTabularExplainer(train ,feature_names = feature_names,class_names=class_names,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3)
np.random.seed(1)
i = 200
exp = explainer.explain_instance(sample_test_row[0], predict_fn, num_features=10)
exp.show_in_notebook(show_all=True)





