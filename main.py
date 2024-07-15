from sklearn import preprocessing
import streamlit as st
import pandas as pd
import seaborn as sns
from MissingValueHandlerClass import MissingValueHandler

#******************************methods************************************
def performing_eda(data):
    """
    Perform Exploratory Data Analysis (EDA) on the given dataset.

    Parameters:
        data (DataFrame): The dataset for which EDA is to be performed.
    """
    st.header("Basic Exploratory Data Analysis 📈.")

    # Display shape
    st.subheader("DataFrame Shape")
    st.write(pd.DataFrame({"Rows": [data.shape[0]], "Columns": [data.shape[1]]}))
    
    # Display Column data types
    st.subheader("Column Data Types")
    col_info_df = pd.DataFrame({
    'Column Name': data.columns,
    'Data Type': [str(dtype) for dtype in data.dtypes]
})
    st.write(col_info_df)

    # Display summary statistics
    st.subheader('Summary Statistics')
    st.write(data.describe())

    # Display missing values
    st.subheader('Missing Values')
    st.write(data.isnull().sum()) 

    # Display Pairplot
    st.subheader("Pairplot")
    fig = sns.pairplot(data)
    st.pyplot(fig)
#*************************************************************************************************
def MissingValue(mdata):
    # In this section we Handle Missing Values in the data

    # Display a warning if no data is uploaded
    if st.session_state["data"].empty :
        st.warning("Please upload some data first.")
    else:
        st.title("Handle Missing Values🧼")
        handler = MissingValueHandler(st.session_state["data"])
        # Display NA count
        st.subheader("The NA count in each column of the data")
        na_counts_display = st.empty()  # Placeholder for displaying NA counts
        na_counts = st.session_state["data"].isnull().sum()
        na_counts_display.dataframe(na_counts)
    
        encoding_options = ['Drop all missing rows', 'Replace Missing rows']
        encoding_option = st.radio('Select a method to handle missing values:', encoding_options, index=None, key='na_option')

        
        if encoding_option == 'Drop all missing rows':
            
            st.session_state["data"] = handler.drop_missing()
            st.write("Preview of the dataset")
            st.dataframe(st.session_state["data"])
            # Update NA counts after dropping missing values
            na_counts_updated = st.session_state["data"].isnull().sum()
            na_counts_display.dataframe(na_counts_updated)
                            
        elif encoding_option == 'Replace Missing rows':
            st.session_state["data"] = mdata.copy()

            # Handling Numerical Missing Values
            st.subheader("Handle Numerical Missing Values")
            numerical_columns =  st.session_state["data"].select_dtypes(include=["int", "float"]).columns.tolist()
            numerical_col = st.multiselect("Select a column to fill missing values in:", numerical_columns, key="numerical_col_select")
            numerical_strategy = st.selectbox("Select a strategy to fill missing values:", ("mode", "mean", "median"), key="numerical_strategy_select", index=None, placeholder="Select an option...")
            if numerical_strategy:
                handler.fill_missing(numerical_col, numerical_strategy)
                st.write("Preview of the dataset after filling missing values:")
                st.session_state["data"] = handler.get_data()
                st.dataframe(st.session_state["data"][numerical_columns])
                na_counts_updated = st.session_state["data"].isnull().sum()
                na_counts_display.dataframe(na_counts_updated)

            # Handling Categorical Missing Values
            st.subheader("Handle Categorical Missing Values")
            categorical_columns =  st.session_state["data"].select_dtypes(include=["object", "category"]).columns.tolist()
            categorical_col = st.multiselect("Select a column to fill missing values in:", categorical_columns, key="categorical_col_select")
            categorical_strategy = st.selectbox("Select a strategy to fill missing values:", ("mode", "missing"), key="categorical_strategy_select", index=None, placeholder="Select an option...")
            if categorical_strategy:
                handler.fill_missing(categorical_col, categorical_strategy)
                st.write("Preview of the dataset after filling missing values:")
                st.session_state["data"] = handler.get_data()
                st.dataframe(st.session_state["data"][categorical_columns])
                na_counts_updated = st.session_state["data"].isnull().sum()
                na_counts_display.dataframe(na_counts_updated)


    return st.session_state["data"]
    

#****************************************************************************
def prediction_model(Y_col):
    """
    Determine if the column is categorical or discrete,
    and decide if it's suitable for classification or regression.
    """
    data_type = Y_col.dtype
    if data_type == 'object':
        return "Classification"
    elif data_type == 'int64' or data_type == 'float64':
        return "Regression"
#***********************end of the methods******************************* 

if "data" not in st.session_state:
    st.session_state["data"] = None

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Welcome to InsightiML 🔭")
st.write("Empowering users with advanced machine learning for dataset analysis and insights.")

uploaded_file = st.file_uploader("Upload file 📤", type=['csv', 'xlsx', 'json'])
if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1]
    if file_ext == "csv":
        st.session_state["data"] = pd.read_csv(uploaded_file)
    elif file_ext in ["xlsx", "xls"]:
        st.session_state["data"] = pd.read_excel(uploaded_file)
    elif file_ext == "json":
        st.session_state["data"] =  pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file format.")
         
else:
    st.info("Please upload a file.")
      
if uploaded_file:
    st.dataframe(st.session_state["data"])
drop_columns_option = st.checkbox("Do you want to drop any columns?")
if drop_columns_option:
    if st.session_state["data"] is  None:
        st.warning("No data found")
    else:
        columns_to_drop = st.multiselect("Select columns to drop:", st.session_state["data"].columns.tolist())
        if columns_to_drop:
            st.session_state["data"] = st.session_state["data"].drop(columns=columns_to_drop)
            st.write("Columns has been Dropped:")
            st.dataframe(st.session_state["data"])
        else:
            st.warning("Please select at least one column to drop.")

perform_eda = st.checkbox("Do you want to perform EDA?")
if perform_eda:
    if st.session_state["data"] is  None:
        st.warning("No data found")
    else:
        eda_columns = st.multiselect("Select columns to perform EDA:", st.session_state["data"].columns.tolist())
        if eda_columns:
            st.session_state["data"] =  st.session_state["data"][eda_columns]
            #method calling 

            performing_eda(st.session_state["data"])


hv = st.checkbox("Do you want to handle missing values?")
if hv :
    if st.session_state["data"] is  None:
        st.warning("No data found")
    else:
        st.session_state["data"]= MissingValue(st.session_state["data"]) #method calling

machine_learning = st.checkbox("Prepare for machine learnin?")
##----------------------------------------------
if machine_learning :
    if st.session_state["data"] is  None:
        st.warning("No data found")
    else:
        st.title("Model Prediction")
        columns = st.session_state["data"].columns.tolist()
        # Selecting features (X) and target (Y) columns
        X_columns = st.multiselect("Select the features (X) columns:", columns)
        Y_column = st.selectbox("Select the target (Y) column:", columns, index=None, placeholder="Select an option...")
        st.session_state.setdefault("model_type", "")

        # Check if the target column is selected
        if Y_column not in columns:
            st.warning("Please select a target (Y) column.")
        else:
            if Y_column in X_columns:
                st.warning("You cannot choose the same column for both X and Y.")
            else:
                st.session_state["X_column_data"] = st.session_state["data"][X_columns]
                st.session_state["Y_column_data"] = st.session_state["data"][Y_column]
                
                # Combine X and Y data for further processing
                st.session_state["combined_data"] = pd.concat([st.session_state["X_column_data"], pd.Series(st.session_state["Y_column_data"], name=Y_column)], axis=1)
                
                # Predict the model type
                model_type = prediction_model(st.session_state["Y_column_data"])
                st.subheader("Recommended Model type: " + model_type)
                
                # Label encode Y column if it's categorical
                if st.session_state["Y_column_data"].dtype == 'object':
                    if st.session_state["Y_column_data"].isnull().values.any():
                         st.warning("The categorical target variable (Y column) could not be label encoded.")
                         st.warning("Please address missing values in the dataset.")

                    else:
                        st.subheader("The categorical target variable (Y column) has been successfully label encoded.")
                        label_encoder = preprocessing.LabelEncoder()
                        st.session_state["Y_column_data"] = pd.Series(label_encoder.fit_transform(st.session_state["Y_column_data"]), name=Y_column)
                        st.write(st.session_state["Y_column_data"])
                
                # Categorical encoding for X columns if required
                if  X_columns is not None:
                    if st.checkbox("Do you need to perform categorical encoding on X"):

                        if st.session_state["X_column_data"].isnull().values.any():
                          st.warning("Please address missing values in the dataset.")
                        else:
                            encoding_option = st.radio('Select encoding method for the X columns:', ('One-Hot Encoding', 'Label Encoding'), key='encoding_option')

                            if encoding_option == 'One-Hot Encoding':
                                st.session_state["X_column_data"] = pd.get_dummies(st.session_state["X_column_data"])
                            else:
                                label_encoder = preprocessing.LabelEncoder()
                                categorical_columns = st.session_state["X_column_data"].select_dtypes(include=['object']).columns
                                for col in categorical_columns:
                                    st.session_state["X_column_data"][col] = label_encoder.fit_transform(st.session_state["X_column_data"][col])
                            
                st.session_state["combined_data"] = pd.concat([st.session_state["X_column_data"], pd.Series(st.session_state["Y_column_data"], name=Y_column)], axis=1)
                # Show the combined dataframe if requested
                if st.checkbox("Would you like to check out the dataframe?"):
                    if st.session_state["combined_data"] is None:
                        st.warning("No data found")
                    else:
                        st.dataframe(pd.concat([st.session_state["X_column_data"], pd.Series(st.session_state["Y_column_data"], name=Y_column)], axis=1))
                
                # Machine learning model training
                if st.checkbox("Would you like to begin the machine learning training process?"):
                    if st.session_state["combined_data"] is None:
                        st.warning("No data found")
                    elif st.session_state["X_column_data"].isnull().values.any():
                         st.warning("Please address missing values in the dataset.")

                    else:
                        if model_type == "Regression":
                            from pycaret.regression import setup, compare_models, predict_model
                        elif model_type == "Classification":
                            from pycaret.classification import setup, compare_models, predict_model
                        
                        st.title("Machine Learning Models Training for " + model_type)
                        setup_data = setup(data=st.session_state["combined_data"], target=Y_column)
                        best_model = compare_models()
                        st.table(setup_data.pull())
                        
                        st.title("The Best Model:")
                        predict_model(best_model)
                        st.table(setup_data.pull())
                