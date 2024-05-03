# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("C:/Users/akmik/OneDrive/Desktop/BFIN 491/HW/HW7/hw7.csv")

# Display the first few rows of the dataframe
print(df.head())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Check data types of each column
print(df.dtypes)

# Count of each category in categorical columns
categorical_columns = ['sex', 'education', 'marriage', 'payment_status_sep', 'payment_status_aug', 
                       'payment_status_jul', 'payment_status_jun', 'payment_status_may', 'payment_status_apr']
for column in categorical_columns:
    print(df[column].value_counts())

# Visualizations (e.g., histograms, box plots, etc.) to understand distributions and relationships

# Histogram of 'age'
plt.figure(figsize=(8, 6))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Boxplot of 'limit_bal' vs. 'default_payment_next_month'
plt.figure(figsize=(10, 6))
sns.boxplot(x='default_payment_next_month', y='limit_bal', data=df)
plt.title('Boxplot of Limit Balance by Default Payment Next Month')
plt.xlabel('Default Payment Next Month')
plt.ylabel('Limit Balance')
plt.show()

# Splitting data into features and target variable
X = df.drop('default_payment_next_month', axis=1)
y = df['default_payment_next_month']

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for categorical and numerical columns
categorical_features = ['sex', 'education', 'marriage', 'payment_status_sep', 'payment_status_aug', 
                        'payment_status_jul', 'payment_status_jun', 'payment_status_may', 'payment_status_apr']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = DecisionTreeClassifier(random_state=42)

# Create pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Fit the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Define hyperparameters grid
param_grid = {
    'model__max_depth': [3, 5, 7, 10],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Grid search using cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Refit the model using best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predictions
y_pred = best_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with tuned hyperparameters:", accuracy)