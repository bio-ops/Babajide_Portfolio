# Babajide Ogunbanwo
## Data Analytics Portfolio

# Model Improvement & Implementation

Model Improvement techniques such as cross validation, ensemble method, and parameter tuning utilized to optimize a classification model:

* Data was taken from a Credit Dataset in .csv format
* Label encoding and data splitting performed on dataset
* Decision Tree model created and evaluated using ROC metric
* Cross validation technique performed to improve model score
* Bagging, Boosting, and random forest ensemble methods performed to compare resulting model scores
* Grid search using GridSearchCV from sklearn module leveraged to tune model hyperparameters
* Model performance score calculated using optimal model hyperparameters and best model identified

# Supervised Learning

Classification analysis using a decision tree model was performed with the goal of building a model used to predict likelihood of a potential customer accepting a loan:

* Data was taken from an Universal Bank Dataset in .csv format
* Label encoding and data split into training/test set performed
* DecisionTreeClassifier method from scikitlearn library leveraged to build decision tree model
* Confusion matrix and AUC score used to evaluate model performance

Classification analysis using a logistic regression model was performed with the goal of building a model used to predict likelihood of a potential customer accepting a loan:

* Data was taken from a Customer Churn Dataset in .csv format
* Label encoding and data split into training/test set performed
* LogisticRegression method from scikitlearn library leveraged to build logistic regression model
* Confusion matrix and AUC score used to evaluate model performance

Regression analysis on an insurance cost dataset was performed with the goal of predicting the cost of insurance based on a set of predictors: 

* Data was taken from an Insurance Dataset in .csv format
* Correlation heatmap created using seaborn and matplotlib
* Label encoding and division of data into training/test set performed using SciKitLearn
* Regression model created using statsmodels.api
* Regression model evaluated using test data and based on RMSE/Coefficient of Determination

# Unsupervised Learning

K-cluster analysis on a pharmaceutical company dataset was performed with the goal of creating a basket of pharmaceutical companiess with similar stock attributes:

* Data was taken from a Pharmaceutical Stock Dataset in .csv format
* Standardization of numerical features was performed using the StandardScaler module from the SciKit learn preprocessing module
* KMeans Cluster Analysis performed. Optimal K determined utilizing the elbow method
* Parallel Coordinates created to visualize the centroids of each cluster
* Scatterplot used to visualize clusters across unique attribute dimensions

Hierarchical cluster analysis on an Automobile dealership dataset was performed with the goal of clustering similar car models together: 

* Data was taken from an Automobile Dataset in .csv format
* Data was preprocessed using the SciKitLearn Preprocessing module
* A dendogram was created and plotted using the average linkage and ward linkage method

Association Rule analysis on a grocery basket data set  was performed with the goal of identify grocery items that are typically bought together and thus, associated with each other:

* Data was taken from a Grocery Dataset in .csv format
* Data preprocessed using the TransactionEncoder method from the mlextend preprocessing library
* Frequency analysis of items bought performed and visualized using a bar graph
* Association rules generated using association_rules method from mlextend frequent_patterns library
* Dataframe with association rules for each item set created for further analysis

# Exploratory Data Analysis

In this project, an exploratory analysis on the salaries of SanFrancisco residents was performed:

* Data was taken from a SF Salary Dataset in .csv format
* Descriptive statistics analysis performed using the .describe() method
* Information extracted from dataset based on employee name, job title, etc
* Box plots and histogram of variables of importance generated
* Libraries used: Pandas, Seaborn


