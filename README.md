# Pima-Indians-Diabetes-Iris-Classification-Decision-Tree-Multi-Layer-Perceptron

## Overview:


## Language, Classification Method, Tool:

- Python

- Decision Tree

- Multi Layer Perceptron (MLP)

## Dataset:

- Iris

- Pima Indians Diabetes Database: https://www.kaggle.com/uciml/pima-indians-diabetes-database

### Context

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

### Content

The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

## Overview:

### Decision Trees vs. MLPs

What are the main differences between nonlinear models MLPs and Decision Trees?

There are 2 main differences: 

- 1) Decision boundary: arbitrary curves for MLPs but *segments* parallel with axes for Decision Trees. For regression surface: arbitrary for MLPs but *block-wise* for Decision Trees 

- 2) Explainability 

### Decision Trees' predicted output

- How outputs are calculated? How to measure its quality?

- For regression: average of data subset at leaf nodes; smaller spreading (standard deviation) is better. 

- For classification: majority label of data subset at leaf nodes; smaller entropy is better. 

## Analysis:

### Iris Classification

#### Iris Classes Distribution

<img src="https://user-images.githubusercontent.com/70437668/141033065-8aa1a0b1-e33e-46f7-b5e9-c392138d3f91.jpg" width=50% height=50%>

#### Get the best_model after tuning and evaluating on the Train Set and Test Set
```
from sklearn.model_selection import GridSearchCV

params = {
    'criterion': ['entropy','gini'],
    'max_leaf_nodes': list(range(2, 30)),
    'max_depth': list(range(2, 10)),
    'min_samples_split': np.linspace(0.1, 1.0, 10), 
    'max_features': [1,2],
}

model = DecisionTreeClassifier( random_state=1612)

grid = GridSearchCV(model, params, cv=5)
grid.fit(X_train, y_train)
grid.best_params_
```
```
Accuracy on Train Set 0.9666666666666667
Accuracy on Test Set 0.95
```

#### Iris Tuned model boundary on Train Set

<img src="https://user-images.githubusercontent.com/70437668/141033098-c3b4b865-a00d-4b1e-bd85-268aa88dc731.jpg" width=50% height=50%>

#### Iris Tuned model boundary on Test Set

<img src="https://user-images.githubusercontent.com/70437668/141033117-9e428f79-9c49-4b94-82ec-e7e04d3ce65e.jpg" width=50% height=50%>

#### Iris Best Decision Tree model
```
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

# draw tree
def drawTree(model):
  dot_data = StringIO()
  export_graphviz(model, out_file=dot_data,  
                  filled=True, rounded=True,
                  special_characters=True,
                  feature_names = iris.feature_names[2:],
                  class_names= iris.target_names
                  )
  graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
  return graph.create_png()
```
<img src="https://user-images.githubusercontent.com/70437668/141033171-9af369df-1cfa-42c1-9423-9f267a94a903.jpg" width=50% height=50%>


#### Conclusion

If the petal length is <= 2.42, it will be the class setosa. If not, in the case of pedal with <= 1.7, the class will be versicolor. Otherwise, it will be virginica.

### Pima Indians Diabetes Dataase

#### Pair Plot

![Pima Pair Plot](https://user-images.githubusercontent.com/70437668/141033248-2973d4d0-d272-4579-b7ee-6a08cc58cde7.jpg)

![Pima Pair Plot 2](https://user-images.githubusercontent.com/70437668/141033256-2794daeb-5070-4f8d-aa4e-ad3b15803650.jpg)

#### Anomalies

Features **`Insulin, SkinThickness, BloodPressure, BMI, Glucose`** has many values equal to 0. With these features, value = 0 means that those data points are null (nan / na).

#### Create and Train the Decision Tree and Random Forest

```
# 1. Create model for Decision Tree and RandomForest
# 2. Fit both models on Train Set
# 3. Use score() function on Train Set and Test Set

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 

model_decisiontree = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=1612)
model_randomforest = RandomForestClassifier(criterion='entropy', max_depth=4, random_state=1613, n_estimators=300)

model_decisiontree.fit(X_train, y_train)
model_randomforest.fit(X_train, y_train)

print('Decision Tree score on Train Set: ' + str(model_decisiontree.score(X_train, y_train)))
print('Decision Tree score on Test Set: ' + str(model_decisiontree.score(X_test, y_test)))

print('Random Forest score on Train Set: ' + str(model_decisiontree.score(X_train, y_train)))
print('Random Forest score on Test Set: ' + str(model_decisiontree.score(X_test, y_test)))
```
```
Decision Tree score on Train Set: 0.808695652173913
Decision Tree score on Test Set: 0.7435064935064936
Random Forest score on Train Set: 0.808695652173913
Random Forest score on Test Set: 0.7435064935064936
```
