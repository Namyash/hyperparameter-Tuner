# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression
# from sklearn.metrics import mean_squared_error, accuracy_score

# # Title of the app
# st.title("Dataset Uploader and Analyzer")

# # Step 1: Upload a CSV file
# uploaded_file = st.file_uploader("Upload your dataset (CSV file only)", type=["csv"])

# # Check if a file is uploaded
# if uploaded_file is not None:
#     try:
#         # Load the dataset into a DataFrame
#         df = pd.read_csv(uploaded_file)
        
#         # Check if the dataset is empty
#         if df.empty:
#             st.error("The uploaded file is empty. Please upload a valid dataset.")
#         else:
#             # Step 2: Show the preview of the dataset
#             st.write("Here is a preview of your dataset:")
#             st.dataframe(df.head())

#             # Step 3: Check the column types (numerical/categorical)
#             categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
#             numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

#             st.write(f"Categorical Columns: {categorical_cols}")
#             st.write(f"Numerical Columns: {numerical_cols}")

#             # Step 4: Ask the user to select the target column
#             target_column = st.selectbox("Please select the target column", df.columns)
#             st.write(f"Selected target column: {target_column}")

#             # Ensure that the user selects the target column and proceed only if it's selected
#             if target_column not in df.columns:
#                 st.error("Please select a valid target column.")
#             else:
#                 # Step 8: Visualize Data
#                 if st.checkbox("Visualize Data Distribution"):
#                     viz_column = st.selectbox("Select a column to visualize", df.columns)
#                     viz_type = st.radio("Choose visualization type", ["Histogram", "Count Plot", "Box Plot"])

#                     plt.figure(figsize=(6, 4))
#                     if viz_type == "Histogram" and viz_column in numerical_cols:
#                         sns.histplot(df[viz_column], kde=True)
#                     elif viz_type == "Count Plot" and viz_column in categorical_cols:
#                         sns.countplot(x=df[viz_column])
#                     elif viz_type == "Box Plot" and viz_column in numerical_cols:
#                         sns.boxplot(x=df[viz_column])
#                     else:
#                         st.write("Invalid combination of column and visualization type.")
#                     st.pyplot(plt)

#                 # Step 5: Handle missing values with user choice
#                 if st.checkbox("Handle Missing Values"):
#                     st.write("Choose how to handle missing values:")
#                     handle_option = st.radio(
#                         "Select a method:",
#                         ["Remove rows with null values", "Fill numerical columns with Mean", 
#                          "Fill numerical columns with Median", "Fill numerical columns with Mode", 
#                          "Drop columns with > 50% missing values"]
#                     )

#                     if handle_option == "Remove rows with null values":
#                         df = df.dropna()
#                         st.write("Rows with null values removed.")

#                     elif handle_option == "Fill numerical columns with Mean":
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Missing values filled with mean for numerical columns and mode for categorical columns.")

#                     elif handle_option == "Fill numerical columns with Median":
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Missing values filled with median for numerical columns and mode for categorical columns.")

#                     elif handle_option == "Fill numerical columns with Mode":
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mode().iloc[0])
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Missing values filled with mode for both numerical and categorical columns.")

#                     elif handle_option == "Drop columns with > 50% missing values":
#                         df = df.loc[:, df.isnull().mean() < 0.5]
#                         st.write("Columns with more than 50% missing values dropped.")

#                 # Step 5.1: Handle Outliers
#                 if st.checkbox("Handle Outliers"):
#                     st.write("Choose how to handle outliers:")
#                     outlier_option = st.radio(
#                         "Select a method for handling outliers:",
#                         ["Remove rows with outliers", "Cap outliers to 1st and 99th percentiles"]
#                     )

#                     # Function to remove outliers
#                     def remove_outliers(df, numerical_cols):
#                         for col in numerical_cols:
#                             Q1 = df[col].quantile(0.25)
#                             Q3 = df[col].quantile(0.75)
#                             IQR = Q3 - Q1
#                             lower_bound = Q1 - 1.5 * IQR
#                             upper_bound = Q3 + 1.5 * IQR
#                             df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#                         return df

#                     # Function to cap outliers
#                     def cap_outliers(df, numerical_cols):
#                         for col in numerical_cols:
#                             Q1 = df[col].quantile(0.25)
#                             Q3 = df[col].quantile(0.75)
#                             IQR = Q3 - Q1
#                             lower_bound = Q1 - 1.5 * IQR
#                             upper_bound = Q3 + 1.5 * IQR
#                             df[col] = np.clip(df[col], lower_bound, upper_bound)
#                         return df

#                     if outlier_option == "Remove rows with outliers":
#                         df = remove_outliers(df, numerical_cols)
#                         st.write("Rows with outliers removed.")

#                     elif outlier_option == "Cap outliers to 1st and 99th percentiles":
#                         df = cap_outliers(df, numerical_cols)
#                         st.write("Outliers capped to 1st and 99th percentiles.")

#                 # Step 6: Standardize all numerical data except target column
#                 st.write("Standardizing all numerical columns except the target column...")
#                 scaler = StandardScaler()
#                 numerical_cols_without_target = [col for col in numerical_cols if col != target_column]
#                 df[numerical_cols_without_target] = scaler.fit_transform(df[numerical_cols_without_target])
#                 st.write(f"Standardized columns: {numerical_cols_without_target}")

#                 # Show the dataset after standardization
#                 st.write("Here is the dataset after standardization:")
#                 st.dataframe(df.head())

#                 # Step 7: Encode categorical columns
#                 if len(categorical_cols) > 0:
#                     encoder = LabelEncoder()
#                     for col in categorical_cols:
#                         df[col] = encoder.fit_transform(df[col])
#                     st.write("Categorical columns encoded successfully.")

#                 # Step 9: Split Data into Train and Test
#                 X = df.drop(columns=[target_column])
#                 y = df[target_column]

#                 # Check if the data is empty after splitting
#                 if X.empty or y.empty:
#                     st.error("The feature or target data is empty after splitting. Please check your dataset.")
#                 else:
#                     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#                     # Step 10: Automatically Detect Suitable Models
#                     st.write("Analyzing dataset to suggest suitable models...")

#                     if y.nunique() == 2:
#                         st.write("Suitable Models for Binary Classification: Logistic Regression, Perceptron, Random Forest, SVM")
#                         suggested_models = ["Logistic Regression", "Perceptron", "Random Forest", "SVM"]
#                     elif y.nunique() > 2:
#                         st.write("Suitable Models for Multi-class Classification: Random Forest, SVM, Gradient Boosting")
#                         suggested_models = ["Random Forest", "SVM", "Gradient Boosting"]
#                     else:
#                         st.write("Suitable Models for Regression: Linear Regression, Random Forest Regressor")
#                         suggested_models = ["Linear Regression", "Random Forest"]

#                     # Step 11: Model Selection and Hyperparameter Tuning
#                     selected_models = st.multiselect("Select models for hyperparameter tuning", suggested_models)

#                     if selected_models:
#                         param_grids = {
#                             "Random Forest": {
#                                 'n_estimators': [50, 100, 200],
#                                 'max_depth': [5, 10, 20, None],
#                                 'min_samples_split': [2, 5, 10],
#                                 'min_samples_leaf': [1, 2, 4]
#                             },
#                             "SVM": {
#                                 'C': [0.1, 1, 10],
#                                 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#                                 'gamma': ['scale', 'auto']
#                             },
#                             "Logistic Regression": {
#                                 'C': [0.1, 1, 10],
#                                 'penalty': ['l1', 'l2', 'elasticnet'],
#                                 'solver': ['liblinear', 'saga']
#                             },
#                             "Perceptron": {
#                                 'alpha': [0.0001, 0.001, 0.01],
#                                 'max_iter': [1000, 2000, 5000],
#                                 'tol': [1e-4, 1e-3]
#                             },
#                             "Linear Regression": {
#                                 # No hyperparameters for basic Linear Regression in sklearn
#                             },
#                             "Gradient Boosting": {
#                                 'n_estimators': [50, 100, 200],
#                                 'learning_rate': [0.01, 0.1, 0.2],
#                                 'max_depth': [3, 5, 7],
#                                 'subsample': [0.8, 1.0]
#                             }
#                         }

#                         best_models = {}
#                         for model_name in selected_models:
#                             st.write(f"Running GridSearch for {model_name}...")
#                             model = None
#                             if model_name == "Random Forest":
#                                 model = RandomForestClassifier()
#                             elif model_name == "SVM":
#                                 model = SVC()
#                             elif model_name == "Logistic Regression":
#                                 model = LogisticRegression()
#                             elif model_name == "Perceptron":
#                                 model = Perceptron()
#                             elif model_name == "Linear Regression":
#                                 model = LinearRegression()
#                             elif model_name == "Gradient Boosting":
#                                 model = GradientBoostingClassifier()

#                             try:
#                                 if param_grids[model_name]:
#                                     grid_search = GridSearchCV(model, param_grids[model_name], cv=5)
#                                     grid_search.fit(X_train, y_train)
#                                     best_models[model_name] = grid_search.best_estimator_
#                                     st.write(f"Best parameters for {model_name}: {grid_search.best_params_}")
#                                 else:
#                                     model.fit(X_train, y_train)
#                                     best_models[model_name] = model
#                                     st.write(f"{model_name} does not require hyperparameter tuning.")
#                             except Exception as e:
#                                 st.error(f"Error during GridSearch for {model_name}: {e}")

#                         # Evaluate models
#                         for model_name, model in best_models.items():
#                             if model_name == "Linear Regression":
#                                 predictions = model.predict(X_test)
#                                 mse = mean_squared_error(y_test, predictions)
#                                 st.write(f"{model_name} - Mean Squared Error: {mse:.2f}")
#                             else:
#                                 accuracy = accuracy_score(y_test, model.predict(X_test))
#                                 st.write(f"{model_name} - Accuracy: {accuracy:.2f}")
#     except Exception as e:
#         st.error(f"An error occurred while processing the file: {e}")
# else:
#     st.write("Please upload a CSV file to get started.")

















# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression
# from sklearn.metrics import mean_squared_error, accuracy_score

# # Title of the app
# st.title("Dataset Uploader and Analyzer")

# # Step 1: Upload a CSV file
# uploaded_file = st.file_uploader("Upload your dataset (CSV file only)", type=["csv"])

# # Check if a file is uploaded
# if uploaded_file is not None:
#     try:
#         # Load the dataset into a DataFrame
#         df = pd.read_csv(uploaded_file)
        
#         # Check if the dataset is empty
#         if df.empty:
#             st.error("The uploaded file is empty. Please upload a valid dataset.")
#         else:
#             # Step 2: Show the preview of the dataset
#             st.write("Here is a preview of your dataset:")
#             st.dataframe(df.head())

#             # Step 3: Check the column types (numerical/categorical)
#             categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
#             numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

#             st.write(f"Categorical Columns: {categorical_cols}")
#             st.write(f"Numerical Columns: {numerical_cols}")

#             # Step 4: Ask the user to select the target column
#             target_column = st.selectbox("Please select the target column", df.columns)
#             st.write(f"Selected target column: {target_column}")

#             # Ensure that the user selects the target column and proceed only if it's selected
#             if target_column not in df.columns:
#                 st.error("Please select a valid target column.")
#             else:
#                 # Step 8: Visualize Data
#                 if st.checkbox("Visualize Data Distribution"):
#                     viz_column = st.selectbox("Select a column to visualize", df.columns)
#                     viz_type = st.radio("Choose visualization type", ["Histogram", "Count Plot", "Box Plot"])

#                     plt.figure(figsize=(6, 4))
#                     if viz_type == "Histogram" and viz_column in numerical_cols:
#                         sns.histplot(df[viz_column], kde=True)
#                     elif viz_type == "Count Plot" and viz_column in categorical_cols:
#                         sns.countplot(x=df[viz_column])
#                     elif viz_type == "Box Plot" and viz_column in numerical_cols:
#                         sns.boxplot(x=df[viz_column])
#                     else:
#                         st.write("Invalid combination of column and visualization type.")
#                     st.pyplot(plt)

#                 # Step 8.1: Correlation Heatmap
#                 if st.checkbox("Show Correlation Heatmap"):
#                     st.write("Visualizing the correlation between numerical features:")
#                     if len(numerical_cols) > 1:
#                         corr_matrix = df[numerical_cols].corr()
#                         plt.figure(figsize=(8, 6))
#                         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
#                         st.pyplot(plt)
#                     else:
#                         st.warning("Not enough numerical columns to generate a correlation heatmap.")

#                 # Step 5: Handle missing values with user choice
#                 if st.checkbox("Handle Missing Values"):
#                     st.write("Choose how to handle missing values:")
#                     handle_option = st.radio(
#                         "Select a method:",
#                         ["Remove rows with null values", "Fill numerical columns with Mean", 
#                          "Fill numerical columns with Median", "Fill numerical columns with Mode", 
#                          "Drop columns with > 50% missing values"]
#                     )

#                     if handle_option == "Remove rows with null values":
#                         df = df.dropna()
#                         st.write("Rows with null values removed.")

#                     elif handle_option == "Fill numerical columns with Mean":
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Missing values filled with mean for numerical columns and mode for categorical columns.")

#                     elif handle_option == "Fill numerical columns with Median":
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Missing values filled with median for numerical columns and mode for categorical columns.")

#                     elif handle_option == "Fill numerical columns with Mode":
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mode().iloc[0])
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Missing values filled with mode for both numerical and categorical columns.")

#                     elif handle_option == "Drop columns with > 50% missing values":
#                         df = df.loc[:, df.isnull().mean() < 0.5]
#                         st.write("Columns with more than 50% missing values dropped.")

#                 # Step 5.1: Handle Outliers
#                 if st.checkbox("Handle Outliers"):
#                     st.write("Choose how to handle outliers:")
#                     outlier_option = st.radio(
#                         "Select a method for handling outliers:",
#                         ["Remove rows with outliers", "Cap outliers to 1st and 99th percentiles"]
#                     )

#                     # Function to remove outliers
#                     def remove_outliers(df, numerical_cols):
#                         for col in numerical_cols:
#                             Q1 = df[col].quantile(0.25)
#                             Q3 = df[col].quantile(0.75)
#                             IQR = Q3 - Q1
#                             lower_bound = Q1 - 1.5 * IQR
#                             upper_bound = Q3 + 1.5 * IQR
#                             df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#                         return df

#                     # Function to cap outliers
#                     def cap_outliers(df, numerical_cols):
#                         for col in numerical_cols:
#                             Q1 = df[col].quantile(0.25)
#                             Q3 = df[col].quantile(0.75)
#                             IQR = Q3 - Q1
#                             lower_bound = Q1 - 1.5 * IQR
#                             upper_bound = Q3 + 1.5 * IQR
#                             df[col] = np.clip(df[col], lower_bound, upper_bound)
#                         return df

#                     if outlier_option == "Remove rows with outliers":
#                         df = remove_outliers(df, numerical_cols)
#                         st.write("Rows with outliers removed.")

#                     elif outlier_option == "Cap outliers to 1st and 99th percentiles":
#                         df = cap_outliers(df, numerical_cols)
#                         st.write("Outliers capped to 1st and 99th percentiles.")

#                 # Step 6: Standardize all numerical data except target column
#                 st.write("Standardizing all numerical columns except the target column...")
#                 scaler = StandardScaler()
#                 numerical_cols_without_target = [col for col in numerical_cols if col != target_column]
#                 df[numerical_cols_without_target] = scaler.fit_transform(df[numerical_cols_without_target])
#                 st.write(f"Standardized columns: {numerical_cols_without_target}")

#                 # Show the dataset after standardization
#                 st.write("Here is the dataset after standardization:")
#                 st.dataframe(df.head())

#                 # Step 7: Feature Selection
#                 if st.checkbox("Perform Feature Selection"):
#                     st.write("Performing Feature Selection based on correlation with the target column...")

#                     # Calculate correlation with the target column for numerical features
#                     correlation_threshold = st.slider("Set correlation threshold", 0.0, 1.0, 0.1)
#                     correlations = df[numerical_cols_without_target].corrwith(df[target_column])
#                     selected_features = correlations[correlations.abs() >= correlation_threshold].index.tolist()

#                     st.write(f"Features selected based on correlation threshold ({correlation_threshold}): {selected_features}")

#                     if not selected_features:
#                         st.warning("No features meet the correlation threshold. Please adjust the threshold or check your dataset.")
#                     else:
#                         X = df[selected_features]
#                         st.write("Selected features dataset:")
#                         st.dataframe(X.head())

#                         y = df[target_column]

#                         # Split Data into Train and Test
#                         st.write("Splitting data into training and testing sets...")
#                         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#                         st.write("Data split successfully.")

#                         # Step 10: Automatically Detect Suitable Models
#                         st.write("Analyzing dataset to suggest suitable models...")

#                         if y.nunique() == 2:
#                             st.write("Suitable Models for Binary Classification: Logistic Regression, Perceptron, Random Forest, SVM")
#                             suggested_models = ["Logistic Regression", "Perceptron", "Random Forest", "SVM"]
#                         elif y.nunique() > 2:
#                             st.write("Suitable Models for Multi-class Classification: Random Forest, SVM, Gradient Boosting")
#                             suggested_models = ["Random Forest", "SVM", "Gradient Boosting"]
#                         else:
#                             st.write("Suitable Models for Regression: Linear Regression, Random Forest Regressor")
#                             suggested_models = ["Linear Regression", "Random Forest"]

#                         # Step 11: Model Selection and Hyperparameter Tuning
#                         selected_models = st.multiselect("Select models for hyperparameter tuning", suggested_models)

#                         if selected_models:
#                             param_grids = {
#                                 "Random Forest": {
#                                     'n_estimators': [50, 100, 200],
#                                     'max_depth': [5, 10, 20, None],
#                                     'min_samples_split': [2, 5, 10],
#                                     'min_samples_leaf': [1, 2, 4]
#                                 },
#                                 "SVM": {
#                                     'C': [0.1, 1, 10],
#                                     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#                                     'gamma': ['scale', 'auto']
#                                 },
#                                 "Logistic Regression": {
#                                     'C': [0.1, 1, 10],
#                                     'penalty': ['l1', 'l2', 'elasticnet'],
#                                     'solver': ['liblinear', 'saga']
#                                 },
#                                 "Perceptron": {
#                                     'alpha': [0.0001, 0.001, 0.01],
#                                     'max_iter': [1000, 2000, 5000],
#                                     'tol': [1e-4, 1e-3]
#                                 },
#                                 "Linear Regression": {
#                                     # No hyperparameters for basic Linear Regression in sklearn
#                                 },
#                                 "Gradient Boosting": {
#                                     'n_estimators': [50, 100, 200],
#                                     'learning_rate': [0.01, 0.1, 0.2],
#                                     'max_depth': [3, 5, 7],
#                                     'subsample': [0.8, 1.0]
#                                 }
#                             }

#                             best_models = {}
#                             for model_name in selected_models:
#                                 st.write(f"Running GridSearch for {model_name}...")
#                                 model = None
#                                 if model_name == "Random Forest":
#                                     model = RandomForestClassifier()
#                                 elif model_name == "SVM":
#                                     model = SVC()
#                                 elif model_name == "Logistic Regression":
#                                     model = LogisticRegression()
#                                 elif model_name == "Perceptron":
#                                     model = Perceptron()
#                                 elif model_name == "Linear Regression":
#                                     model = LinearRegression()
#                                 elif model_name == "Gradient Boosting":
#                                     model = GradientBoostingClassifier()

#                                 try:
#                                     if param_grids[model_name]:
#                                         grid_search = GridSearchCV(model, param_grids[model_name], cv=5)
#                                         grid_search.fit(X_train, y_train)
#                                         best_models[model_name] = grid_search.best_estimator_
#                                         st.write(f"Best parameters for {model_name}: {grid_search.best_params_}")
#                                     else:
#                                         model.fit(X_train, y_train)
#                                         best_models[model_name] = model
#                                         st.write(f"{model_name} does not require hyperparameter tuning.")
#                                 except Exception as e:
#                                     st.error(f"Error during GridSearch for {model_name}: {e}")

#                             # Evaluate models
#                             for model_name, model in best_models.items():
#                                 if model_name == "Linear Regression":
#                                     predictions = model.predict(X_test)
#                                     mse = mean_squared_error(y_test, predictions)
#                                     st.write(f"{model_name} - Mean Squared Error: {mse:.2f}")
#                                 else:
#                                     accuracy = accuracy_score(y_test, model.predict(X_test))
#                                     st.write(f"{model_name} - Accuracy: {accuracy:.2f}")
#     except Exception as e:
#         st.error(f"An error occurred while processing the file: {e}")
# else:
#     st.write("Please upload a CSV file to get started.")



















# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression
# from sklearn.metrics import mean_squared_error, accuracy_score

# # Title of the app
# st.title("Dataset Uploader and Analyzer")

# # Step 1: Upload a CSV file
# uploaded_file = st.file_uploader("Upload your dataset (CSV file only)", type=["csv"])

# # Check if a file is uploaded
# if uploaded_file is not None:
#     try:
#         # Load the dataset into a DataFrame
#         df = pd.read_csv(uploaded_file)
        
#         # Check if the dataset is empty
#         if df.empty:
#             st.error("The uploaded file is empty. Please upload a valid dataset.")
#         else:
#             # Step 2: Show the preview of the dataset
#             st.write("Here is a preview of your dataset:")
#             st.dataframe(df.head())

#             # Step 3: Check the column types (numerical/categorical)
#             categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
#             numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

#             st.write(f"Categorical Columns: {categorical_cols}")
#             st.write(f"Numerical Columns: {numerical_cols}")

#             # Step 4: Ask the user to select the target column
#             target_column = st.selectbox("Please select the target column", df.columns)
#             st.write(f"Selected target column: {target_column}")

#             # Ensure that the user selects the target column and proceed only if it's selected
#             if target_column not in df.columns:
#                 st.error("Please select a valid target column.")
#             else:
#                 # Encode categorical columns if required
#                 if st.checkbox("Encode Categorical Columns"):
#                     label_encoders = {}
#                     for col in categorical_cols:
#                         le = LabelEncoder()
#                         df[col] = le.fit_transform(df[col].astype(str))
#                         label_encoders[col] = le
#                     st.write("Categorical columns have been label encoded.")

#                 # Encode the target column if selected
#                 if st.checkbox("Encode Target Column"):
#                     if target_column in categorical_cols:
#                         le_target = LabelEncoder()
#                         df[target_column] = le_target.fit_transform(df[target_column].astype(str))
#                         st.write(f"Target column '{target_column}' has been label encoded.")

#                 # Step 5: Handle missing values with user choice
#                 if st.checkbox("Handle Missing Values"):
#                     st.write("Choose how to handle missing values:")
#                     handle_option = st.radio(
#                         "Select a method:",
#                         ["Remove rows with null values", "Fill numerical columns with Mean", 
#                          "Fill numerical columns with Median", "Fill numerical columns with Mode", 
#                          "Drop columns with > 50% missing values"]
#                     )

#                     if handle_option == "Remove rows with null values":
#                         df = df.dropna()
#                         st.write("Rows with null values removed.")

#                     elif handle_option == "Fill numerical columns with Mean":
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Missing values filled with mean for numerical columns and mode for categorical columns.")

#                     elif handle_option == "Fill numerical columns with Median":
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Missing values filled with median for numerical columns and mode for categorical columns.")

#                     elif handle_option == "Fill numerical columns with Mode":
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mode().iloc[0])
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Missing values filled with mode for both numerical and categorical columns.")

#                     elif handle_option == "Drop columns with > 50% missing values":
#                         df = df.loc[:, df.isnull().mean() < 0.5]
#                         st.write("Columns with more than 50% missing values dropped.")

#                 # Step 5.1: Handle Outliers
#                 if st.checkbox("Handle Outliers"):
#                     st.write("Choose how to handle outliers:")
#                     outlier_option = st.radio(
#                         "Select a method for handling outliers:",
#                         ["Remove rows with outliers", "Cap outliers to 1st and 99th percentiles"]
#                     )

#                     # Function to remove outliers
#                     def remove_outliers(df, numerical_cols):
#                         for col in numerical_cols:
#                             Q1 = df[col].quantile(0.25)
#                             Q3 = df[col].quantile(0.75)
#                             IQR = Q3 - Q1
#                             lower_bound = Q1 - 1.5 * IQR
#                             upper_bound = Q3 + 1.5 * IQR
#                             df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#                         return df

#                     # Function to cap outliers
#                     def cap_outliers(df, numerical_cols):
#                         for col in numerical_cols:
#                             Q1 = df[col].quantile(0.25)
#                             Q3 = df[col].quantile(0.75)
#                             IQR = Q3 - Q1
#                             lower_bound = Q1 - 1.5 * IQR
#                             upper_bound = Q3 + 1.5 * IQR
#                             df[col] = np.clip(df[col], lower_bound, upper_bound)
#                         return df

#                     if outlier_option == "Remove rows with outliers":
#                         df = remove_outliers(df, numerical_cols)
#                         st.write("Rows with outliers removed.")

#                     elif outlier_option == "Cap outliers to 1st and 99th percentiles":
#                         df = cap_outliers(df, numerical_cols)
#                         st.write("Outliers capped to 1st and 99th percentiles.")

#                 # Step 6: Standardize all numerical data except target column
#                 st.write("Standardizing all numerical columns except the target column...")
#                 scaler = StandardScaler()
#                 numerical_cols_without_target = [col for col in numerical_cols if col != target_column]
#                 df[numerical_cols_without_target] = scaler.fit_transform(df[numerical_cols_without_target])
#                 st.write(f"Standardized columns: {numerical_cols_without_target}")

#                 # Show the dataset after standardization
#                 st.write("Here is the dataset after standardization:")
#                 st.dataframe(df.head())

#                 # Step 8: Visualize Data
#                 if st.checkbox("Visualize Data Distribution"):
#                     viz_column = st.selectbox("Select a column to visualize", df.columns)
#                     viz_type = st.radio("Choose visualization type", ["Histogram", "Count Plot", "Box Plot"])

#                     plt.figure(figsize=(6, 4))
#                     if viz_type == "Histogram" and viz_column in numerical_cols:
#                         sns.histplot(df[viz_column], kde=True)
#                     elif viz_type == "Count Plot" and viz_column in categorical_cols:
#                         sns.countplot(x=df[viz_column])
#                     elif viz_type == "Box Plot" and viz_column in numerical_cols:
#                         sns.boxplot(x=df[viz_column])
#                     else:
#                         st.write("Invalid combination of column and visualization type.")
#                     st.pyplot(plt)

#                 # Step 8.1: Correlation Heatmap
#                 if st.checkbox("Show Correlation Heatmap"):
#                     st.write("Visualizing the correlation between numerical features:")
#                     if len(numerical_cols) > 1:
#                         corr_matrix = df[numerical_cols].corr()
#                         plt.figure(figsize=(8, 6))
#                         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
#                         st.pyplot(plt)
#                     else:
#                         st.warning("Not enough numerical columns to generate a correlation heatmap.")

#                 # Step 7: Feature Selection
#                 if st.checkbox("Perform Feature Selection"):
#                     st.write("Performing Feature Selection based on correlation with the target column...")

#                     # Calculate correlation with the target column for numerical features
#                     # correlation_threshold = st.slider("Set correlation threshold", 0.0, 1.0, 0.1)
#                     # correlations = df[numerical_cols_without_target].corrwith(df[target_column])
#                     # selected_features = correlations[correlations.abs() >= correlation_threshold].index.tolist()

#                     # st.write(f"Features selected based on correlation threshold ({correlation_threshold}): {selected_features}")
#                     correlation_threshold = st.slider("Set correlation threshold", -1.0, 1.0, 0.1)
#                     correlations = df[numerical_cols_without_target].corrwith(df[target_column])
#                     selected_features = correlations[correlations.abs() >= correlation_threshold].index.tolist()

#                     st.write(f"Features selected based on correlation threshold ({correlation_threshold}): {selected_features}")

#                 if not selected_features:
#                     st.warning("No features meet the correlation threshold. Please adjust the threshold or check your dataset.")
#                 else:
#                     st.write("Selected features dataset:")
#                     st.dataframe(df[selected_features].head())

#                     if not selected_features:
#                         st.warning("No features meet the correlation threshold. Please adjust the threshold or check your dataset.")
#                     else:
#                         X = df[selected_features]
#                         st.write("Selected features dataset:")
#                         st.dataframe(X.head())

#                         y = df[target_column]

#                         # Split Data into Train and Test
#                         st.write("Splitting data into training and testing sets...")
#                         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#                         st.write("Data split successfully.")

#                         # Step 10: Automatically Detect Suitable Models
#                         st.write("Analyzing dataset to suggest suitable models...")

#                         if y.nunique() == 2:
#                             st.write("Suitable Models for Binary Classification: Logistic Regression, Perceptron, Random Forest, SVM")
#                             suggested_models = ["Logistic Regression", "Perceptron", "Random Forest", "SVM"]
#                         elif y.nunique() > 2:
#                             st.write("Suitable Models for Multi-class Classification: Random Forest, SVM, Gradient Boosting")
#                             suggested_models = ["Random Forest", "SVM", "Gradient Boosting"]
#                         else:
#                             st.write("Suitable Models for Regression: Linear Regression, Random Forest Regressor")
#                             suggested_models = ["Linear Regression", "Random Forest"]

#                         # Step 11: Model Selection and Hyperparameter Tuning
#                         selected_models = st.multiselect("Select models for hyperparameter tuning", suggested_models)

#                         if selected_models:
#                             param_grids = {
#                                 "Random Forest": {
#                                     'n_estimators': [50, 100, 200],
#                                     'max_depth': [5, 10, 20, None],
#                                     'min_samples_split': [2, 5, 10],
#                                     'min_samples_leaf': [1, 2, 4]
#                                 },
#                                 "SVM": {
#                                     'C': [0.1, 1, 10],
#                                     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#                                     'gamma': ['scale', 'auto']
#                                 },
#                                 "Logistic Regression": {
#                                     'C': [0.1, 1, 10],
#                                     'penalty': ['l1', 'l2', 'elasticnet'],
#                                     'solver': ['liblinear', 'saga']
#                                 },
#                                 "Perceptron": {
#                                     'alpha': [0.0001, 0.001, 0.01],
#                                     'max_iter': [1000, 2000, 5000],
#                                     'tol': [1e-4, 1e-3]
#                                 },
#                                 "Linear Regression": {
#                                     # No hyperparameters for basic Linear Regression in sklearn
#                                 },
#                                 "Gradient Boosting": {
#                                     'n_estimators': [50, 100, 200],
#                                     'learning_rate': [0.01, 0.1, 0.2],
#                                     'max_depth': [3, 5, 7],
#                                     'subsample': [0.8, 1.0]
#                                 }
#                             }

#                             best_models = {}
#                             for model_name in selected_models:
#                                 st.write(f"Running GridSearch for {model_name}...")
#                                 model = None
#                                 if model_name == "Random Forest":
#                                     model = RandomForestClassifier()
#                                 elif model_name == "SVM":
#                                     model = SVC()
#                                 elif model_name == "Logistic Regression":
#                                     model = LogisticRegression()
#                                 elif model_name == "Perceptron":
#                                     model = Perceptron()
#                                 elif model_name == "Linear Regression":
#                                     model = LinearRegression()
#                                 elif model_name == "Gradient Boosting":
#                                     model = GradientBoostingClassifier()

#                                 try:
#                                     if param_grids[model_name]:
#                                         grid_search = GridSearchCV(model, param_grids[model_name], cv=5)
#                                         grid_search.fit(X_train, y_train)
#                                         best_models[model_name] = grid_search.best_estimator_
#                                         st.write(f"Best parameters for {model_name}: {grid_search.best_params_}")
#                                     else:
#                                         model.fit(X_train, y_train)
#                                         best_models[model_name] = model
#                                         st.write(f"{model_name} does not require hyperparameter tuning.")
#                                 except Exception as e:
#                                     st.error(f"Error during GridSearch for {model_name}: {e}")

#                             # Evaluate models
#                             for model_name, model in best_models.items():
#                                 if model_name == "Linear Regression":
#                                     predictions = model.predict(X_test)
#                                     mse = mean_squared_error(y_test, predictions)
#                                     st.write(f"{model_name} - Mean Squared Error: {mse:.2f}")
#                                 else:
#                                     accuracy = accuracy_score(y_test, model.predict(X_test))
#                                     st.write(f"{model_name} - Accuracy: {accuracy:.2f}")
#     except Exception as e:
#         st.error(f"An error occurred while processing the file: {e}")
# else:
#     st.write("Please upload a CSV file to get started.")


















# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression
# from sklearn.metrics import mean_squared_error, accuracy_score




# # Title of the app
# st.title("Dataset Uploader and Analyzer")

# # Step 1: Upload a CSV file
# uploaded_file = st.file_uploader("Upload your dataset (CSV file only)", type=["csv"])

# # Check if a file is uploaded
# if uploaded_file is not None:
#     try:
#         # Load the dataset into a DataFrame
#         df = pd.read_csv(uploaded_file)

#         # Check if the dataset is empty
#         if df.empty:
#             st.error("The uploaded file is empty. Please upload a valid dataset.")
#         else:
#             # Step 2: Show the preview of the dataset
#             st.write("Here is a preview of your dataset:")
#             st.dataframe(df.head())

#             # Step 3: Check the column types (numerical/categorical)
#             categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
#             numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

#             st.write(f"Categorical Columns: {categorical_cols}")
#             st.write(f"Numerical Columns: {numerical_cols}")

#             # Step 4: Ask the user to select the target column
#             target_column = st.selectbox("Please select the target column", df.columns)
#             st.write(f"Selected target column: {target_column}")

#             # Ensure that the user selects the target column and proceed only if it's selected
#             if target_column not in df.columns:
#                 st.error("Please select a valid target column.")
#             else:
#                 # Step 4.1: Encoding options for the target column
#                 target_encoding = st.radio(
#                     "How do you want to handle the target column encoding?",
#                     ("Do not encode", "Label Encode"),
#                     index=0
#                 )

#                 if target_encoding == "Label Encode" and df[target_column].dtype == 'object':
#                     le = LabelEncoder()
#                     df[target_column] = le.fit_transform(df[target_column])
#                     st.write("Target column encoded using Label Encoding.")

#                 # Option to encode categorical features
#                 if st.checkbox("Encode Categorical Features"):
#                     st.write("Encoding categorical features using Label Encoding...")
#                     for col in categorical_cols:
#                         le = LabelEncoder()
#                         df[col] = le.fit_transform(df[col])
#                     st.write("Categorical features encoded.")

#                 # Step 8: Visualize Data
#                 if st.checkbox("Visualize Data Distribution"):
#                     viz_column = st.selectbox("Select a column to visualize", df.columns)
#                     viz_type = st.radio("Choose visualization type", ["Histogram", "Count Plot", "Box Plot"])

#                     plt.figure(figsize=(6, 4))
#                     if viz_type == "Histogram" and viz_column in numerical_cols:
#                         sns.histplot(df[viz_column], kde=True)
#                     elif viz_type == "Count Plot" and viz_column in categorical_cols:
#                         sns.countplot(x=df[viz_column])
#                     elif viz_type == "Box Plot" and viz_column in numerical_cols:
#                         sns.boxplot(x=df[viz_column])
#                     else:
#                         st.write("Invalid combination of column and visualization type.")
#                     st.pyplot(plt)

#                 # Step 8.1: Correlation Heatmap
#                 if st.checkbox("Show Correlation Heatmap"):
#                     st.write("Visualizing the correlation between numerical features:")
#                     if len(numerical_cols) > 1:
#                         corr_matrix = df[numerical_cols].corr()
#                         plt.figure(figsize=(14, 12))
#                         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
#                         st.pyplot(plt)
#                     else:
#                         st.warning("Not enough numerical columns to generate a correlation heatmap.")

#                 # Step 5: Handle missing values with user choice
#                 if st.checkbox("Handle Missing Values"):
#                     st.write("Choose how to handle missing values:")
#                     handle_option = st.radio(
#                         "Select a method:",
#                         ["Remove rows with null values", "Fill numerical columns with Mean", 
#                          "Fill numerical columns with Median", "Fill numerical columns with Mode", 
#                          "Drop columns with > 50% missing values"]
#                     )

#                     if handle_option == "Remove rows with null values":
#                         df = df.dropna()
#                         st.write("Rows with null values removed.")

#                     elif handle_option == "Fill numerical columns with Mean":
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Missing values filled with mean for numerical columns and mode for categorical columns.")

#                     elif handle_option == "Fill numerical columns with Median":
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Missing values filled with median for numerical columns and mode for categorical columns.")

#                     elif handle_option == "Fill numerical columns with Mode":
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mode().iloc[0])
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Missing values filled with mode for both numerical and categorical columns.")

#                     elif handle_option == "Drop columns with > 50% missing values":
#                         df = df.loc[:, df.isnull().mean() < 0.5]
#                         st.write("Columns with more than 50% missing values dropped.")

#                 # Step 5.1: Handle Outliers
#                 if st.checkbox("Handle Outliers"):
#                     st.write("Choose how to handle outliers:")
#                     outlier_option = st.radio(
#                         "Select a method for handling outliers:",
#                         ["Remove rows with outliers", "Cap outliers to 1st and 99th percentiles"]
#                     )

#                     # Function to remove outliers
#                     def remove_outliers(df, numerical_cols):
#                         for col in numerical_cols:
#                             Q1 = df[col].quantile(0.25)
#                             Q3 = df[col].quantile(0.75)
#                             IQR = Q3 - Q1
#                             lower_bound = Q1 - 1.5 * IQR
#                             upper_bound = Q3 + 1.5 * IQR
#                             df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#                         return df

#                     # Function to cap outliers
#                     def cap_outliers(df, numerical_cols):
#                         for col in numerical_cols:
#                             Q1 = df[col].quantile(0.25)
#                             Q3 = df[col].quantile(0.75)
#                             IQR = Q3 - Q1
#                             lower_bound = Q1 - 1.5 * IQR
#                             upper_bound = Q3 + 1.5 * IQR
#                             df[col] = np.clip(df[col], lower_bound, upper_bound)
#                         return df

#                     if outlier_option == "Remove rows with outliers":
#                         df = remove_outliers(df, numerical_cols)
#                         st.write("Rows with outliers removed.")

#                     elif outlier_option == "Cap outliers to 1st and 99th percentiles":
#                         df = cap_outliers(df, numerical_cols)
#                         st.write("Outliers capped to 1st and 99th percentiles.")

#                 # Step 6: Standardize all numerical data except target column
#                 st.write("Standardizing all numerical columns except the target column...")
#                 scaler = StandardScaler()
#                 numerical_cols_without_target = [col for col in numerical_cols if col != target_column]
#                 df[numerical_cols_without_target] = scaler.fit_transform(df[numerical_cols_without_target])
#                 st.write(f"Standardized columns: {numerical_cols_without_target}")

#                 # Show the dataset after standardization
#                 st.write("Here is the dataset after standardization:")
#                 st.dataframe(df.head())

#                 # Step 7: Feature Selection (Optional)
#                 # Step 7: Feature Selection (Optional)
#                 selected_features = None
#                 if st.checkbox("Perform Feature Selection"):
#                     st.write("Performing Feature Selection based on correlation with the target column...")

#     # Calculate correlation with the target column for numerical features
#                 correlation_threshold = st.slider("Set correlation threshold", -1.0, 1.0, 0.1)
#                 correlations = df[numerical_cols_without_target].corrwith(df[target_column])
#                 selected_features = correlations[correlations.abs() >= correlation_threshold].index.tolist()

#                 st.write(f"Features selected based on correlation threshold ({correlation_threshold}): {selected_features}")

#                 if not selected_features:
#                     st.warning("No features meet the correlation threshold. Please adjust the threshold or check your dataset.")
#                 else:
#                     st.write("Selected features dataset:")
#                     st.dataframe(df[selected_features].head())


#                 # Define features (X) and target (y)
#                 if selected_features:
#                     X = df[selected_features]
#                 else:
#                     X = df[numerical_cols_without_target]  # Use all numerical columns if no feature selection
#                 y = df[target_column]

#                 # Split Data into Train and Test
#                 st.write("Splitting data into training and testing sets...")
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#                 st.write("Data split successfully.")

#                 # Step 10: Automatically Detect Suitable Models
#                 st.write("Analyzing dataset to suggest suitable models...")

#                 if y.nunique() == 2:
#                     st.write("Suitable Models for Binary Classification: Logistic Regression, Perceptron, Random Forest, SVM")
#                     suggested_models = ["Logistic Regression", "Perceptron", "Random Forest", "SVM"]
#                 elif y.nunique() > 2:
#                     st.write("Suitable Models for Multi-class Classification: Random Forest, SVM, Gradient Boosting")
#                     suggested_models = ["Random Forest", "SVM", "Gradient Boosting"]
#                 else:
#                     st.write("Suitable Models for Regression: Linear Regression, Random Forest Regressor")
#                     suggested_models = ["Linear Regression", "Random Forest"]

#                 # Step 11: Model Selection and Hyperparameter Tuning
#                 selected_models = st.multiselect("Select models for hyperparameter tuning", suggested_models)

#                 if selected_models:
#                     param_grids = {
#     "Random Forest": {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [5, 10, 20, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     },
#     "SVM": {
#         'C': [0.1, 1, 10],
#         'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#         'gamma': ['scale', 'auto']
#     },
#     "Logistic Regression": {
#         'C': [0.1, 1, 10],
#         'penalty': ['l1', 'l2'],  # Removed 'elasticnet' penalty
#         'solver': ['liblinear', 'saga']  # Use 'saga' only for 'l2' penalty if needed
#     },
#     "Perceptron": {
#         'alpha': [0.0001, 0.001, 0.01],
#         'max_iter': [1000, 2000, 5000],
#         'tol': [1e-4, 1e-3]
#     },
#     "Linear Regression": {
#         # No hyperparameters for basic Linear Regression in sklearn
#     },
#     "Gradient Boosting": {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_depth': [3, 5, 7],
#         'subsample': [0.8, 1.0]
#     }
# }


#                     best_models = {}
#                     for model_name in selected_models:
#                         st.write(f"Running GridSearch for {model_name}...")
#                         model = None
#                         if model_name == "Random Forest":
#                             model = RandomForestClassifier()
#                         elif model_name == "SVM":
#                             model = SVC()
#                         elif model_name == "Logistic Regression":
#                             model = LogisticRegression()
#                         elif model_name == "Perceptron":
#                             model = Perceptron()
#                         elif model_name == "Linear Regression":
#                             model = LinearRegression()
#                         elif model_name == "Gradient Boosting":
#                             model = GradientBoostingClassifier()

#                         if model_name != "Linear Regression":
#                             grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=3, scoring='accuracy')
#                             grid_search.fit(X_train, y_train)
#                             best_model = grid_search.best_estimator_
#                             best_score = grid_search.best_score_
#                             st.write(f"Best parameters for {model_name}: {grid_search.best_params_}")
#                             st.write(f"Best cross-validation accuracy for {model_name}: {best_score:.4f}")
#                             best_models[model_name] = best_model
#                         else:
#                             model.fit(X_train, y_train)
#                             score = model.score(X_test, y_test)
#                             st.write(f"Linear Regression score: {score:.4f}")
#                             best_models[model_name] = model

#                     st.write("Hyperparameter tuning completed. Best models:")
#                     for model_name, best_model in best_models.items():
#                         st.write(f"{model_name}: {best_model}")

#     except Exception as e:
#         st.error(f"An error occurred: {e}")












import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Title of the app
st.title("Dataset Uploader and Analyzer")

# Step 1: Upload a CSV file
uploaded_file = st.file_uploader("Upload your dataset (CSV file only)", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    try:
        # Load the dataset into a DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Check if the dataset is empty
        if df.empty:
            st.error("The uploaded file is empty. Please upload a valid dataset.")
        else:
            # Step 2: Show the preview of the dataset
            st.write("Here is a preview of your dataset:")
            st.dataframe(df.head())

            # Step 3: Check the column types (numerical/categorical)
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

            st.write(f"Categorical Columns: {categorical_cols}")
            st.write(f"Numerical Columns: {numerical_cols}")

            # Step 4: Ask the user to select the target column
            target_column = st.selectbox("Please select the target column", df.columns)
            st.write(f"Selected target column: {target_column}")

            # Ensure that the user selects the target column and proceed only if it's selected
            if target_column not in df.columns:
                st.error("Please select a valid target column.")
            else:
                # Encode categorical columns if required
                if st.checkbox("Encode Categorical Columns"):
                    label_encoders = {}
                    for col in categorical_cols:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        label_encoders[col] = le
                    st.write("Categorical columns have been label encoded.")

                # Encode the target column if selected
                if st.checkbox("Encode Target Column"):
                    if target_column in categorical_cols:
                        le_target = LabelEncoder()
                        df[target_column] = le_target.fit_transform(df[target_column].astype(str))
                        st.write(f"Target column '{target_column}' has been label encoded.")

                # Step 5: Handle missing values with user choice
                if st.checkbox("Handle Missing Values"):
                    st.write("Choose how to handle missing values:")
                    handle_option = st.radio(
                        "Select a method:",
                        ["Remove rows with null values", "Fill numerical columns with Mean", 
                         "Fill numerical columns with Median", "Fill numerical columns with Mode", 
                         "Drop columns with > 50% missing values"]
                    )

                    if handle_option == "Remove rows with null values":
                        df = df.dropna()
                        st.write("Rows with null values removed.")

                    elif handle_option == "Fill numerical columns with Mean":
                        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
                        for col in categorical_cols:
                            df[col].fillna(df[col].mode()[0], inplace=True)
                        st.write("Missing values filled with mean for numerical columns and mode for categorical columns.")

                    elif handle_option == "Fill numerical columns with Median":
                        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
                        for col in categorical_cols:
                            df[col].fillna(df[col].mode()[0], inplace=True)
                        st.write("Missing values filled with median for numerical columns and mode for categorical columns.")

                    elif handle_option == "Fill numerical columns with Mode":
                        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mode().iloc[0])
                        for col in categorical_cols:
                            df[col].fillna(df[col].mode()[0], inplace=True)
                        st.write("Missing values filled with mode for both numerical and categorical columns.")

                    elif handle_option == "Drop columns with > 50% missing values":
                        df = df.loc[:, df.isnull().mean() < 0.5]
                        st.write("Columns with more than 50% missing values dropped.")

                # Step 5.1: Handle Outliers
                if st.checkbox("Handle Outliers"):
                    st.write("Choose how to handle outliers:")
                    outlier_option = st.radio(
                        "Select a method for handling outliers:",
                        ["Remove rows with outliers", "Cap outliers to 1st and 99th percentiles"]
                    )

                    # Function to remove outliers
                    def remove_outliers(df, numerical_cols):
                        for col in numerical_cols:
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                        return df

                    # Function to cap outliers
                    def cap_outliers(df, numerical_cols):
                        for col in numerical_cols:
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            df[col] = np.clip(df[col], lower_bound, upper_bound)
                        return df

                    if outlier_option == "Remove rows with outliers":
                        df = remove_outliers(df, numerical_cols)
                        st.write("Rows with outliers removed.")

                    elif outlier_option == "Cap outliers to 1st and 99th percentiles":
                        df = cap_outliers(df, numerical_cols)
                        st.write("Outliers capped to 1st and 99th percentiles.")

                # Step 6: Standardize all numerical data except target column
                st.write("Standardizing all numerical columns except the target column...")
                scaler = StandardScaler()
                numerical_cols_without_target = [col for col in numerical_cols if col != target_column]
                df[numerical_cols_without_target] = scaler.fit_transform(df[numerical_cols_without_target])
                st.write(f"Standardized columns: {numerical_cols_without_target}")

                # Show the dataset after standardization
                st.write("Here is the dataset after standardization:")
                st.dataframe(df.head())

                # Step 8: Visualize Data
                if st.checkbox("Visualize Data Distribution"):
                    viz_column = st.selectbox("Select a column to visualize", df.columns)
                    viz_type = st.radio("Choose visualization type", ["Histogram", "Count Plot", "Box Plot", 
                                                                    "Pair Plot", "Heatmap", "Scatter Plot", "Line Plot", "Violin Plot"])

                    plt.figure(figsize=(6, 4))
                    if viz_type == "Histogram" and viz_column in numerical_cols:
                        sns.histplot(df[viz_column], kde=True)
                    elif viz_type == "Count Plot" and viz_column in categorical_cols:
                        sns.countplot(x=df[viz_column])
                    elif viz_type == "Box Plot" and viz_column in numerical_cols:
                        sns.boxplot(x=df[viz_column])
                    elif viz_type == "Pair Plot":
                        sns.pairplot(df[numerical_cols])
                    elif viz_type == "Heatmap" and len(numerical_cols) > 1:
                        corr_matrix = df[numerical_cols].corr()
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
                    elif viz_type == "Scatter Plot" and len(numerical_cols) > 1:
                        x_column = st.selectbox("Select x-axis column", numerical_cols)
                        y_column = st.selectbox("Select y-axis column", numerical_cols)
                        sns.scatterplot(x=df[x_column], y=df[y_column])
                    elif viz_type == "Line Plot" and len(numerical_cols) > 1:
                        x_column = st.selectbox("Select x-axis column", numerical_cols)
                        y_column = st.selectbox("Select y-axis column", numerical_cols)
                        sns.lineplot(x=df[x_column], y=df[y_column])
                    elif viz_type == "Violin Plot" and viz_column in numerical_cols:
                        sns.violinplot(x=df[viz_column])
                    else:
                        st.write("Invalid combination of column and visualization type.")
                    st.pyplot(plt)

                # Step 7: Feature Selection
                if st.checkbox("Perform Feature Selection"):
                    st.write("Performing Feature Selection based on correlation with the target column...")

                    # Calculate correlation with the target column for numerical features
                    correlation_threshold = st.slider("Set correlation threshold", -1.0, 1.0, 0.1)
                    correlations = df[numerical_cols_without_target].corrwith(df[target_column])
                    selected_features = correlations[correlations.abs() >= correlation_threshold].index.tolist()

                    st.write(f"Features selected based on correlation threshold ({correlation_threshold}): {selected_features}")

                if not selected_features:
                    st.warning("No features meet the correlation threshold. Please adjust the threshold or check your dataset.")
                else:
                    st.write("Selected features dataset:")
                    st.dataframe(df[selected_features].head())

                    if not selected_features:
                        st.warning("No features meet the correlation threshold. Please adjust the threshold or check your dataset.")
                    else:
                        X = df[selected_features]
                        y = df[target_column]
                        
                        # Train/Test Split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # Step 8: Model Selection
                        model_option = st.selectbox("Select a model", 
                                                     ["Logistic Regression", "Random Forest", "SVM", 
                                                      "Gradient Boosting", "Perceptron", "Linear Regression"])

                        # Train the model based on user selection
                        if model_option == "Logistic Regression":
                            model = LogisticRegression()
                        elif model_option == "Random Forest":
                            model = RandomForestClassifier()
                        elif model_option == "SVM":
                            model = SVC()
                        elif model_option == "Gradient Boosting":
                            model = GradientBoostingClassifier()
                        elif model_option == "Perceptron":
                            model = Perceptron()
                        elif model_option == "Linear Regression":
                            model = LinearRegression()

                        model.fit(X_train, y_train)

                        # Step 9: Hyperparameter Tuning (GridSearchCV)
                        param_grid = {
                            "Logistic Regression": {"C": [0.1, 1, 10]},
                            "Random Forest": {"n_estimators": [50, 100], "max_depth": [10, 20]},
                            "SVM": {"C": [0.1, 1], "kernel": ["linear", "rbf"]},
                            "Gradient Boosting": {"n_estimators": [50, 100]},
                            "Perceptron": {"alpha": [0.0001, 0.001]},
                            "Linear Regression": {"fit_intercept": [True, False]},
                        }

                        grid_search = GridSearchCV(model, param_grid[model_option], cv=5)
                        grid_search.fit(X_train, y_train)

                        st.write(f"Best Parameters: {grid_search.best_params_}")

                        # Step 10: Evaluate the model
                        if model_option == "Linear Regression":
                            y_pred = grid_search.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            st.write(f"Mean Squared Error: {mse}")
                        else:
                            y_pred = grid_search.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            st.write(f"Accuracy: {accuracy}")

    except Exception as e:
        st.error(f"Error: {e}")
