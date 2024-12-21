# InsightiML: Data Analysis and Machine Learning Web Application 🤖

**InsightiML** – an interactive web app built using **Streamlit** that empowers users with advanced machine learning capabilities for dataset analysis and insights. This app provides functionalities to upload, explore, clean, preprocess, and train machine learning models on datasets, all while leveraging the power of **PyCaret** for automated model selection and training.

## Features

### 1. **Upload Data** 📂
   - Users can upload datasets in `CSV`, `Excel`, or `JSON` formats.
   - The app automatically reads the uploaded file and loads it into the session for further processing.

### 2. **Exploratory Data Analysis (EDA)** 🔍
   - **Basic Stats**: View the dataset's shape (rows and columns) and column data types.
   - **Summary Statistics**: Get summary statistics such as mean, standard deviation, and percentiles for numerical columns.
   - **Missing Values**: Display missing values for each column.
   - **Pairplot**: Visualize relationships between numeric variables using Seaborn’s pairplot.

### 3. **Missing Value Handling** ❓
   - **Drop Missing Data**: Option to drop rows or columns with missing values.
   - **Replace Missing Data**: Fill missing numerical values with mean, median, or mode, and categorical values with mode or 'missing'.

### 4. **Data Preprocessing** 🔧
   - **Feature Selection**: Select relevant features for machine learning models.
   - **Target Variable**: Select the target variable (Y) and features (X).
   - **Encoding**: Perform categorical encoding using either One-Hot Encoding or Label Encoding.

### 5. **Machine Learning Model Training with PyCaret** 🤖
   - **Automatic Model Selection**: Using **PyCaret**, the app automatically analyzes the dataset and determines the appropriate machine learning model type (Classification or Regression) based on the target column's data type.
   - **Model Training**: PyCaret compares various models and selects the best-performing one for the dataset, streamlining the process of finding the most effective model.
   - **Model Prediction**: Once the best model is selected, users can make predictions and view the results instantly.
