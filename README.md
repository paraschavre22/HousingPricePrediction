# HousingPricePrediction

This Project provides end-to-end implementation of a machine learning pipeline for housing price prediction using the Boston Housing Dataset. This dataset contains various attributes of houses and their corresponding median values (MEDV). Here's an overview and description of the different sections of the code:

Data Loading and Exploration:

The code begins by loading the housing dataset from a CSV file named "Mini Project Data.csv" using the Pandas library.
It then prints the first few rows of the dataset using housing.head() to give you an initial view of the data.
The housing.info() function is used to display information about the dataset, including data types, non-null counts, and memory usage.
Value counts are computed for specific columns using housing['CHAS'].value_counts() and housing['INDUS'].value_counts() to get a sense of the distribution of values in categorical attributes.
Descriptive statistics are calculated using housing.describe() to provide insights into the central tendency, spread, and other statistics of the numerical attributes.
Histograms of the dataset's attributes are visualized using housing.hist() to understand the distribution of each attribute.
Data Splitting:

The code demonstrates two methods of splitting the dataset into training and testing sets:
The first method, implemented as split_train_test, randomly splits the data into training and testing sets based on a given test ratio.
The second method uses the train_test_split function from sklearn.model_selection to perform a similar split. The resulting training and testing sets are printed.
Stratified Sampling:

The code uses StratifiedShuffleSplit from sklearn.model_selection to create a stratified split of the dataset. This is particularly useful when dealing with imbalanced categorical attributes. The code demonstrates how to maintain the same distribution of the 'CHAS' attribute in both the training and testing sets.
Correlation Analysis and Data Visualization:

Correlation coefficients between attributes and the target variable (MEDV) are calculated using housing.corr(). These coefficients are sorted to show the attributes with the highest positive and negative correlations.
The code uses scatter plots and correlation values to visualize relationships between specific attributes (e.g., 'RM', 'ZN', 'LSTAT') and the target variable ('MEDV').
Data Preprocessing:

The code handles missing values in the 'RM' attribute by filling them with the median value.
SimpleImputer from sklearn.impute is used to fill missing values in the dataset's numerical attributes with their median values.
Creating a Pipeline:

A data preprocessing pipeline is created using Pipeline from sklearn.pipeline. The pipeline consists of two steps: filling missing values using SimpleImputer and scaling the data using StandardScaler.
Model Selection and Training:

The code demonstrates the use of three regression models for predicting housing prices: LinearRegression, DecisionTreeRegressor, and RandomForestRegressor. The chosen model is the RandomForestRegressor.
The selected model is trained using the transformed training data (housing_num_tr) and housing price labels (housing_labels).
Model Evaluation:

The code evaluates the trained model's performance using the Mean Squared Error (MSE) on the training set.
Cross-validation is performed using cross_val_score to assess the model's performance on different folds of the training data. The calculated Root Mean Squared Error (RMSE) scores are printed.
Saving and Loading the Model:

The trained model is saved to a file named "MiniProject.joblib" using the dump function from joblib.
Model Testing:

The model is tested on the test set (X_test) to make predictions. The predictions are then compared to the actual prices (Y_test), and the RMSE on the test set is calculated and printed.
This code provides a comprehensive example of building a machine learning pipeline for housing price prediction, including data loading, exploration, preprocessing, model selection, training, evaluation, and testing. It showcases key steps and techniques used in machine learning projects, such as handling missing data, creating pipelines, selecting models, and evaluating performance.
	
