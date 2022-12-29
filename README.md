# Babajide Ogunbanwo
## Data Analytics Portfolio

**For best viewing of images in this portfolio, please switch to a light theme**
# Model Improvement & Implementation 

[Model Improvement Repository](https://github.com/bio-ops/Model-Improvement) 

Model improvement techniques such as cross validation, ensemble method, and parameter tuning utilized to optimize a classification model:

* Data was taken from a Credit Dataset in .csv format
* Label encoding and data splitting performed on dataset
* Decision Tree model created and evaluated using ROC metric
* Cross validation technique performed to improve model score
* Bagging, Boosting, and random forest ensemble methods performed to compare resulting model scores
* Grid search using GridSearchCV from sklearn module leveraged to tune model hyperparameters
* Model performance score calculated using optimal model hyperparameters and best model identified

**The model with the best resulting AUC was the Random Forest model as displayed below:** 

![](https://github.com/bio-ops/Portfolio-Images/blob/main/RandomForest_ROC.png)


# Supervised Learning

[Supervised Learning Repository](https://github.com/bio-ops/Supervised-Learning)

Classification analysis using a decision tree model was performed with the goal of building a model used to predict likelihood of a potential customer accepting a loan:

* Data was taken from an Universal Bank Dataset in .csv format
* Label encoding and data split into training/test set performed
* DecisionTreeClassifier method from scikitlearn library leveraged to build decision tree model
* Confusion matrix and AUC score used to evaluate model performance

**Resulting AUC:** 

![](https://github.com/bio-ops/Portfolio-Images/blob/main/UniversalBankModel_AUC.png)

Classification analysis using a logistic regression model was performed with the goal of building a model used to predict likelihood of a potential customer accepting a loan:

* Data was taken from a Customer Churn Dataset in .csv format
* Label encoding and data split into training/test set performed
* LogisticRegression method from scikitlearn library leveraged to build logistic regression model
* Confusion matrix and AUC score used to evaluate model performance


**Resulting Decision Tree Confusion Matrix for Customer Churn:** 

![](https://github.com/bio-ops/Portfolio-Images/blob/main/Decision%20Tree_Confusion_Telco.png)

Regression analysis on a laptop sale dataset was performed utilizing neural network modeling with the goal of predicting retail sale price: 

* Data was taken from an Laptop Sales Dataset in .csv format
* Predictor variables and Target variable identified
* Data split into training and test set
* Keras library used to create a neural network model 
* Model performance evaluated


Regression analysis on an insurance cost dataset was performed with the goal of predicting the cost of insurance based on a set of predictors: 

* Data was taken from an Insurance Dataset in .csv format
* Correlation heatmap created using seaborn and matplotlib
* Label encoding and division of data into training/test set performed using SciKitLearn
* Regression model created using statsmodels.api
* Regression model evaluated using test data and based on RMSE/Coefficient of Determination

![](https://github.com/bio-ops/Portfolio-Images/blob/main/Insurance%20Cost%20_%20LR_%20Heatmap.png)

# Unsupervised Learning

[Unspervised Learning Repository](https://github.com/bio-ops/Unsupervised-Learning)

k-cluster analysis on a mall customer dataset performed with the goal of identifying unique customer segements for marketing purposes: 

* Data was taken from a Mall Customer Dataset in .csv format
* Data read and preprocessed using scikitlearn
* KMeans Cluster Analysis performed. Optimal K determined utilizing the elbow method
* Parallel Coordinates created to visualize the centroids of each cluster
* Scatterplot used to visualize clusters across unique attribute dimensions

**Resulting cluster map across annual income and spending score:**

![](https://github.com/bio-ops/Portfolio-Images/blob/main/Mall%20Customers_Cluster%20Graph.png)

K-cluster analysis on a pharmaceutical company dataset was performed with the goal of creating a basket of pharmaceutical companiess with similar stock attributes:

* Data was taken from a Pharmaceutical Stock Dataset in .csv format
* Standardization of numerical features was performed using the StandardScaler module from the SciKit learn preprocessing module
* KMeans Cluster Analysis performed. Optimal K determined utilizing the elbow method
* Parallel Coordinates created to visualize the centroids of each cluster
* Scatterplot used to visualize clusters across unique attribute dimensions

![](https://github.com/bio-ops/Portfolio-Images/blob/main/Pharmaceutical%20Data%20Clustering_Elbow.png)

![](https://github.com/bio-ops/Portfolio-Images/blob/main/Pharmaceutical%20Data_parallel%20coord.png) 

Hierarchical cluster analysis on an Automobile dealership dataset was performed with the goal of clustering similar car models together: 

* Data was taken from an Automobile Dataset in .csv format
* Data was preprocessed using the SciKitLearn Preprocessing module
* A dendogram was created and plotted using the average linkage and ward linkage method

**Ward Linkage Dendogram:** 

![](https://github.com/bio-ops/Portfolio-Images/blob/main/Hierchical_Ward%20Linkage.png) 

Association Rule analysis on a grocery basket data set  was performed with the goal of identify grocery items that are typically bought together and thus, associated with each other:

* Data was taken from a Grocery Dataset in .csv format
* Data preprocessed using the TransactionEncoder method from the mlextend preprocessing library
* Frequency analysis of items bought performed and visualized using a bar graph
* Association rules generated using association_rules method from mlextend frequent_patterns library
* Dataframe with association rules for each item set created for further analysis

**Frequency Analysis:** 

![](https://github.com/bio-ops/Portfolio-Images/blob/main/ItemFreq_Groceries_AR.png)

# Exploratory Data Analysis

[Exploratory Analysis Repository](https://github.com/bio-ops/Exploratory-Data-Analysis)

In this project, an exploratory analysis on the salaries of SanFrancisco residents was performed:

* Data was taken from a SF Salary Dataset in .csv format
* Descriptive statistics analysis performed using the .describe() method
* Information extracted from dataset based on employee name, job title, etc
* Box plots and histogram of variables of importance generated
* Libraries used: Pandas, Seaborn

![](https://github.com/bio-ops/Portfolio-Images/blob/main/Boston_BoxPlot.png)

![](https://github.com/bio-ops/Portfolio-Images/blob/main/SF%20Salary_Pay%20Distribution.png)
