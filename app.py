from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import multiprocessing
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import ElasticNet

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from striprtf.striprtf import rtf_to_text
from IPython.display import display
import warnings
from sklearn.exceptions import ConvergenceWarning

import streamlit as st
import os
from striprtf.striprtf import rtf_to_text

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Function to parse RTF file and call hackathon_problem function
def rtf_parser(file_path):
    # Read the RTF file
    with open(file_path, 'r') as file:
        rtf_content = file.read()
     
    # Convert the RTF content to text
    text_content = rtf_to_text(rtf_content)
    st.write("rtf parser")
    
    # Call the hackathon_problem function with the text content
    hackathon_problem(text_content)
    

def hackathon_problem(text_content):
    # Read JSON file
    st.write("hackathon")
    json_data = json.loads(text_content)

    # Load Data
    data = pd.read_csv(json_data["design_state_data"]["session_info"]["dataset"])

    # Encode categorical columns
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            data[column] = label_encoders[column].fit_transform(data[column])

    # Define preprocessing steps based on JSON data
    preprocessing_steps = []
    for feature, details in json_data["design_state_data"]["feature_handling"].items():
        if "missing_values" in details and details["missing_values"] == "Impute":
            if details["impute_with"] == "Average of values":
                strategy = 'mean'
            else:
                strategy = 'median'
            preprocessing_steps.append((feature + '_imputer', SimpleImputer(strategy=strategy)))

    # Apply preprocessing steps
    for step in preprocessing_steps:
        feature_name, transformer = step
        data[feature_name] = transformer.fit_transform(data[[feature_name]])

    # Separate X and y
    selected_features = [feature for feature, details in json_data["design_state_data"]["feature_handling"].items() if details["is_selected"]]
    X = data[selected_features]
    Y = data[json_data["design_state_data"]["target"]["target"]]


    # Split Data
    train_ratio = json_data["design_state_data"]["train"]["train_ratio"]
    random_seed = json_data["design_state_data"]["train"]["random_seed"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - train_ratio, random_state=random_seed)

    
    
    # Get the number of available CPU cores for parallelism
    num_cores = multiprocessing.cpu_count()
    
    selected_algorithm = None
    for algorithm, details in json_data["design_state_data"]["algorithms"].items():
        if details["is_selected"]:
            selected_algorithm = algorithm
            break
    
    if selected_algorithm == "RandomForestClassifier":
        model = RandomForestClassifier()
        parameters = {
            "n_estimators": np.linspace(details["min_trees"], details["max_trees"], num=3, dtype=int),
            "max_depth": np.linspace(details["min_depth"], details["max_depth"], num=3, dtype=int),
            "min_samples_leaf": np.linspace(details["min_samples_per_leaf_min_value"], details["min_samples_per_leaf_max_value"], num=3, dtype=int)
        }
        # Modify GridSearchCV instantiation to use parallel processing
        grid_search = GridSearchCV(model, parameters, cv=5, n_jobs=num_cores)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_


    # Evaluate Models
        Y_pred = best_model.predict(X_test)


        confusion_mat = confusion_matrix(Y_test, Y_pred)

        # Create heatmap with seaborn
        sns.heatmap(confusion_mat, annot=True, cmap="viridis", fmt="d", cbar=False,
                    linewidths=0.5, linecolor='gray', square=True,
                    xticklabels=True, yticklabels=True, annot_kws={"size": 10})
        
        # Customize axis labels
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        # Show plot
        plt.savefig('RandomForestClassifier.png') 
        st.image('RandomForestClassifier.png')
        # plt.show()      
        # Assign value to classification_rep_df
        classification_rep_dict = classification_report(Y_test, Y_pred, output_dict=True)
        classification_rep_df = pd.DataFrame(classification_rep_dict)
        # Add some styling to the DataFrame
        classification_rep_styled = classification_rep_df.style.background_gradient(cmap='viridis')
        # Inside each block where you print the classification report, replace the print statement with the following:
        # Print the styled classification report
        st.write("Classification Report:")
        st.write(classification_rep_styled)

     
            
    if selected_algorithm == "RandomForestRegressor":
        # Your RandomForestRegressor code
        model = RandomForestRegressor()
        parameters = {
            "n_estimators": list(range(details["min_trees"], details["max_trees"] + 1)),
            "max_depth": list(range(details["min_depth"], details["max_depth"] + 1)),
            "min_samples_leaf": list(range(details["min_samples_per_leaf_min_value"], details["min_samples_per_leaf_max_value"] + 1))}
        # Modify GridSearchCV instantiation to use parallel processing
        grid_search = GridSearchCV(model, parameters, cv=5, n_jobs=num_cores)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
    
        # Predict on test data
        Y_pred = best_model.predict(X_test)
        
        # Calculate R-squared
        r_squared = best_model.score(X_test, Y_test)
        
        # Calculate adjusted R-squared
        n = len(Y_test)
        k = X_test.shape[1]  # Number of predictors
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        
        # Create DataFrame for metrics
        
        metrics_df = pd.DataFrame({
            'Metric': ['R-squared', 'Adjusted R-squared', 'Root Mean Squared Error (RMSE)'],
            'Value': [r_squared, adjusted_r_squared, rmse]
        })
    
        # Style DataFrame
        styled_metrics_df = (
            metrics_df.style
                .set_properties(**{'text-align': 'left'})  # Align text to the left
                .highlight_max(color='lightgreen')         # Highlight maximum value
                .set_caption('Model Evaluation Metrics')   # Add caption
        )
    
        # Display styled DataFrame
        st.write("metrics_df:")
        st.write(styled_metrics_df)
        
    if selected_algorithm == "LinearRegression":
        # Your LinearRegression code
        best_model = LinearRegression()
        best_model.fit(X_train, Y_train)
    
        # Predict on test data
        Y_pred = best_model.predict(X_test)
        
        # Calculate R-squared
        r_squared = best_model.score(X_test, Y_test)
        
        # Calculate adjusted R-squared
        n = len(Y_test)
        k = X_test.shape[1]  # Number of predictors
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

        # Create DataFrame for metrics
        metrics_df = pd.DataFrame({
            'Metric': ['R-squared', 'Adjusted R-squared', 'Root Mean Squared Error (RMSE)'],
            'Value': [r_squared, adjusted_r_squared, rmse]
        })
    
        # Style DataFrame
        styled_metrics_df = (
            metrics_df.style
                .set_properties(**{'text-align': 'left'})  # Align text to the left
                .highlight_max(color='lightgreen')         # Highlight maximum value
                .set_caption('Model Evaluation Metrics')   # Add caption
        )
    
        # Display styled DataFrame
        st.write("metrics_df:")
        st.write(styled_metrics_df)

    if selected_algorithm == "LogisticRegression":
        model = LogisticRegression()
        parameters = {
            "C": np.linspace(details["min_regparam"], details["max_regparam"], num=5),
            "max_iter": np.linspace(details["min_iter"], details["max_iter"], num=5, dtype=int),
            "l1_ratio": np.linspace(details["min_elasticnet"], details["max_elasticnet"], num=5)
        }
        
        # Modify GridSearchCV instantiation to use parallel processing
        grid_search = GridSearchCV(model, parameters, cv=5, n_jobs=num_cores)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
    
        # Evaluate Models
        Y_pred = best_model.predict(X_test)


        
        confusion_mat = confusion_matrix(Y_test, Y_pred)

        # Create heatmap with seaborn
        sns.heatmap(confusion_mat, annot=True, cmap="viridis", fmt="d", cbar=False,
                    linewidths=0.5, linecolor='gray', square=True,
                    xticklabels=True, yticklabels=True, annot_kws={"size": 10})
        
        # Customize axis labels
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        # Show plot
        plt.savefig('LogisticRegression.png') 
        st.image('LogisticRegression.png')
        # plt.show()      
        # Assign value to classification_rep_df
        classification_rep_dict = classification_report(Y_test, Y_pred, output_dict=True)
        classification_rep_df = pd.DataFrame(classification_rep_dict)
        # Add some styling to the DataFrame
        classification_rep_styled = classification_rep_df.style.background_gradient(cmap='viridis')
        # Inside each block where you print the classification report, replace the print statement with the following:
        # Print the styled classification report
        st.write("Classification Report:")
        st.write(classification_rep_styled)

     
        
    if selected_algorithm in ["RidgeRegression", "LassoRegression"]:
        if selected_algorithm == "RidgeRegression":
            model = Ridge()
        elif selected_algorithm == "LassoRegression":
            model = Lasso()
    
        parameters = {
            "alpha": [i/10 for i in range(int(details["min_regparam"]*10), int(details["max_regparam"]*10)+1)],
            "max_iter": list(range(details["min_iter"], details["max_iter"] + 1))}
        # Modify GridSearchCV instantiation to use parallel processing
        grid_search = GridSearchCV(model, parameters, cv=5)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
    
        # Evaluate Models
        Y_pred = best_model.predict(X_test)
        
        # Calculate R-squared
        r_squared = best_model.score(X_test, Y_test)
        
        # Calculate adjusted R-squared
        n = len(Y_test)
        k = X_test.shape[1]  # Number of predictors
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

        # Create DataFrame for metrics
        metrics_df = pd.DataFrame({
            'Metric': ['R-squared', 'Adjusted R-squared', 'Root Mean Squared Error (RMSE)'],
            'Value': [r_squared, adjusted_r_squared, rmse]
        })
    
        # Style DataFrame
        styled_metrics_df = (
            metrics_df.style
                .set_properties(**{'text-align': 'left'})  # Align text to the left
                .highlight_max(color='lightgreen')         # Highlight maximum value
                .set_caption('Model Evaluation Metrics')   # Add caption
        )
    
        # Display styled DataFrame
        st.write("metrics_df:")
        st.write(styled_metrics_df)
        
    if selected_algorithm == "ElasticNetRegression":
        model = ElasticNet()
        # Hyperparameters
        parameters = {
            "alpha": [i/10 for i in range(int(details["min_regparam"]*10), int(details["max_regparam"]*10)+1)],
            "l1_ratio": [i/10 for i in range(int(details["min_elasticnet"]*10), int(details["max_elasticnet"]*10)+1)],
            "max_iter": list(range(details["min_iter"], details["max_iter"] + 1))}
        # Modify GridSearchCV instantiation to use parallel processing
        grid_search = GridSearchCV(model, parameters, cv=5)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
    
        # Evaluate Models
        Y_pred = best_model.predict(X_test)
        
        # Calculate R-squared
        r_squared = best_model.score(X_test, Y_test)
        
        # Calculate adjusted R-squared
        n = len(Y_test)
        k = X_test.shape[1]  # Number of predictors
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

        # Create DataFrame for metrics
        metrics_df = pd.DataFrame({
            'Metric': ['R-squared', 'Adjusted R-squared', 'Root Mean Squared Error (RMSE)'],
            'Value': [r_squared, adjusted_r_squared, rmse]
        })
    
        # Style DataFrame
        styled_metrics_df = (
            metrics_df.style
                .set_properties(**{'text-align': 'left'})  # Align text to the left
                .highlight_max(color='lightgreen')         # Highlight maximum value
                .set_caption('Model Evaluation Metrics')   # Add caption
        )
    
        # Display styled DataFrame
        st.write("metrics_df:")
        st.write(styled_metrics_df)
        
    if selected_algorithm == "xg_boost":
        # XGBoost specific handling
        model = xgb.XGBClassifier(objective='multi:softmax', 
                                   booster='dart' if details['dart'] else 'gbtree',
                                   tree_method = details['tree_method'] if details['tree_method'] != "" else "auto",
                                   random_state=details['random_state'],)
        parameters = {
            'n_estimators': [details["max_num_of_trees"]] if details["max_num_of_trees"] > 0 else [5],
            'max_depth': details['max_depth_of_tree'],
            'learning_rate': [value * 0.001 for value in details['learningRate']],
            'reg_alpha':  [value * 0.01 for value in details['l1_regularization']],
            'reg_lambda':  [value * 0.01 for value in details['l2_regularization']],
            'gamma': [value * 0.01 for value in details['gamma']],
            'min_child_weight': [value * 0.01 for value in details['min_child_weight']],
            'subsample': [value * 0.01 for value in details['sub_sample']],
            'colsample_bytree': [value * 0.01 for value in details['col_sample_by_tree']]
        }
       
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(model, parameters, cv=5 , n_jobs=num_cores)
        
        # Fit the model with early stopping on the validation set
        grid_search.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], early_stopping_rounds=details['early_stopping_rounds'] if details['early_stopping'] else None)
        
        # Get the best model from grid search
        best_model = grid_search.best_estimator_
        
        # Make predictions on the test set
        Y_pred = best_model.predict(X_test)
        


        confusion_mat = confusion_matrix(Y_test, Y_pred)

        # Create heatmap with seaborn
        sns.heatmap(confusion_mat, annot=True, cmap="viridis", fmt="d", cbar=False,
                    linewidths=0.5, linecolor='gray', square=True,
                    xticklabels=True, yticklabels=True, annot_kws={"size": 10})
        
        
        # Customize axis labels
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        # Show plot
        plt.savefig('Xg_boost.png') 
        st.image('Xg_boost.png')
        # plt.show()      
        # Assign value to classification_rep_df
        classification_rep_dict = classification_report(Y_test, Y_pred, output_dict=True)
        classification_rep_df = pd.DataFrame(classification_rep_dict)
        # Add some styling to the DataFrame
        classification_rep_styled = classification_rep_df.style.background_gradient(cmap='viridis')
        # Inside each block where you print the classification report, replace the print statement with the following:
        # Print the styled classification report
        st.write("Classification Report:")
        st.write(classification_rep_styled)

     
    
    if selected_algorithm == "DecisionTreeClassifier":
        # Decision Tree Classifier specific handling
        criterion = 'gini' if details['use_gini'] else 'entropy'
        # Fix the following line to use 'use_entropy' instead of 'use_best'
        splitter = 'best' if details['use_best'] and not details['use_random'] else 'random'
        
        model = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
        parameters = {
            'max_depth': list(range(details['min_depth'], details['max_depth'] + 1)),
            'min_samples_leaf': details['min_samples_per_leaf']}
        grid_search = GridSearchCV(model, parameters, cv=5)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
        Y_pred = best_model.predict(X_test)


        confusion_mat = confusion_matrix(Y_test, Y_pred)

        # Create heatmap with seaborn
        sns.heatmap(confusion_mat, annot=True, cmap="viridis", fmt="d", cbar=False,
                    linewidths=0.5, linecolor='gray', square=True,
                    xticklabels=True, yticklabels=True, annot_kws={"size": 10})
        
        # Customize axis labels
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        # Show plot
        plt.savefig('dt_class_cm.png') 
        st.image('dt_class_cm.png')
        # plt.show()      
        # Assign value to classification_rep_df
        classification_rep_dict = classification_report(Y_test, Y_pred, output_dict=True)
        classification_rep_df = pd.DataFrame(classification_rep_dict)
        # Add some styling to the DataFrame
        classification_rep_styled = classification_rep_df.style.background_gradient(cmap='viridis')
        # Inside each block where you print the classification report, replace the print statement with the following:
        # Print the styled classification report
        st.write("Classification Report:")
        st.write(classification_rep_styled)

      
    if selected_algorithm == "DecisionTreeRegressor":
        # Decision Tree Regressor specific handling
        splitter = 'best' if details.get('use_best', False) and not details.get('use_random', False) else 'random'
        random_state = details.get('random_state', 10)  # Use the provided random state or default to 10
        
        model = DecisionTreeRegressor( splitter=splitter, random_state=random_state)
        parameters = {
            'max_depth': list(range(details['min_depth'], details['max_depth'] + 1)),
            'min_samples_leaf': details['min_samples_per_leaf']
        }
        
        grid_search = GridSearchCV(model, parameters, cv=5)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
        
        # Predict on test data
        Y_pred = best_model.predict(X_test)
        
        # Calculate R-squared
        r_squared = best_model.score(X_test, Y_test)
        
        # Calculate adjusted R-squared
        n = len(Y_test)
        k = X_test.shape[1]  # Number of predictors
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

        # Create DataFrame for metrics
        metrics_df = pd.DataFrame({
            'Metric': ['R-squared', 'Adjusted R-squared', 'Root Mean Squared Error (RMSE)'],
            'Value': [r_squared, adjusted_r_squared, rmse]
        })
    
        # Style DataFrame
        styled_metrics_df = (
            metrics_df.style
                .set_properties(**{'text-align': 'left'})  # Align text to the left
                .highlight_max(color='lightgreen')         # Highlight maximum value
                .set_caption('Model Evaluation Metrics')   # Add caption
        )
    
        # Display styled DataFrame
        st.write("metrics_df:")
        st.write(styled_metrics_df)
        
    if selected_algorithm == "SVM":
        # SVM specific handling
        kernels = []
        if details['linear_kernel']:
            kernels.append('linear')
        if details['rep_kernel']:
            kernels.append('rbf')
        if details['polynomial_kernel']:
            kernels.append('poly')
        if details['sigmoid_kernel']:
            kernels.append('sigmoid')
        
        model = SVC()
        parameters = {
            'C': details['c_value'],
            'kernel': kernels,
            'gamma': ['auto', 'scale'] if details['scale'] else details['custom_gamma_values'],
            'tol': [10 ** -details['tolerance']],
            'max_iter': [details['max_iterations']]
        }
        grid_search = GridSearchCV(model, parameters, cv=5)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
        
        # Predict on test data
        Y_pred = best_model.predict(X_test)
        


        confusion_mat = confusion_matrix(Y_test, Y_pred)

        # Create heatmap with seaborn
        sns.heatmap(confusion_mat, annot=True, cmap="viridis", fmt="d", cbar=False,
                    linewidths=0.5, linecolor='gray', square=True,
                    xticklabels=True, yticklabels=True, annot_kws={"size": 10})
        
        
        # Customize axis labels
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        # Show plot
        plt.savefig('SVM.png') 
        st.image('SVM.png')
        # plt.show()      
        # Assign value to classification_rep_df
        classification_rep_dict = classification_report(Y_test, Y_pred, output_dict=True)
        classification_rep_df = pd.DataFrame(classification_rep_dict)
        # Add some styling to the DataFrame
        classification_rep_styled = classification_rep_df.style.background_gradient(cmap='viridis')
        # Inside each block where you print the classification report, replace the print statement with the following:
        # Print the styled classification report
        st.write("Classification Report:")
        st.write(classification_rep_styled)

     

    if selected_algorithm == "KNN":
        model = KNeighborsClassifier()
        parameters = {
            'n_neighbors': details['k_value'],
            'weights': ['uniform', 'distance'] if details['distance_weighting'] else ['uniform'],
            'algorithm': ['auto'] if details['neighbour_finding_algorithm'] == "Automatic" else [details['neighbour_finding_algorithm']],
            'p': [details['p_value']] if details['p_value'] > 0 else [1]
        }
        grid_search = GridSearchCV(model, parameters, cv=5)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
        Y_pred = best_model.predict(X_test)

        confusion_mat = confusion_matrix(Y_test, Y_pred)

        # Create heatmap with seaborn
        sns.heatmap(confusion_mat, annot=True, cmap="viridis", fmt="d", cbar=False,
                    linewidths=0.5, linecolor='gray', square=True,
                    xticklabels=True, yticklabels=True, annot_kws={"size": 10})
        
        # Customize axis labels
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        # Show plot
        plt.savefig('KNN.png') 
        st.image('KNN.png')
        # plt.show()      
        # Assign value to classification_rep_df
        classification_rep_dict = classification_report(Y_test, Y_pred, output_dict=True)
        classification_rep_df = pd.DataFrame(classification_rep_dict)
        # Add some styling to the DataFrame
        classification_rep_styled = classification_rep_df.style.background_gradient(cmap='viridis')
        # Inside each block where you print the classification report, replace the print statement with the following:
        # Print the styled classification report
        st.write("Classification Report:")
        st.write(classification_rep_styled)
  
    if selected_algorithm == "neural_network":
        # Neural Network specific handling
  # Initialize the MLPClassifier model with early stopping parameter
        model = MLPClassifier(early_stopping=details['early_stopping'])

        # Define the parameters for grid search
        parameters = {
            'hidden_layer_sizes': details['hidden_layer_sizes'],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'alpha': [details['alpha_value']] if details['alpha_value'] > 0 else [.1],
            'max_iter': [details['max_iterations']] if details['max_iterations'] > 0 else [100],
            'tol': [10 ** -details['convergence_tolerance']] if details['convergence_tolerance'] > 0 else [0.1],
            'solver': [details['solver'].lower()],
            'learning_rate_init': [details['initial_learning_rate']] if details['initial_learning_rate'] > 0 else [0.01],
            'shuffle': [details['shuffle_data']],
            'batch_size': ['auto'] if details['automatic_batching'] else [details['batch_size']],
            'beta_1': [details['beta_1']] if details['beta_1'] != 0 else [.1],
            'beta_2': [details['beta_2']] if details['beta_2'] != 0 else [.1],
            'epsilon': [details['epsilon']] if details['epsilon'] != 0 else [.1],
            'power_t': [details['power_t']] if details['power_t'] != 0 else [.1],
            'momentum': [details['momentum']] if details['momentum'] != 0 else [.1],
            'nesterovs_momentum': [details['use_nesterov_momentum']]
        }
        grid_search = GridSearchCV(model, parameters, cv=5)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
        Y_pred = best_model.predict(X_test)

        confusion_mat = confusion_matrix(Y_test, Y_pred)

        # Create heatmap with seaborn
        sns.heatmap(confusion_mat, annot=True, cmap="viridis", fmt="d", cbar=False,
                    linewidths=0.5, linecolor='gray', square=True,
                    xticklabels=True, yticklabels=True, annot_kws={"size": 10})
        # Customize axis labels
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        # Show plot
        plt.savefig('neural_network.png') 
        st.image('neural_network.png')
        # plt.show()      
        # Assign value to classification_rep_df
        classification_rep_dict = classification_report(Y_test, Y_pred, output_dict=True)
        classification_rep_df = pd.DataFrame(classification_rep_dict)
        # Add some styling to the DataFrame
        classification_rep_styled = classification_rep_df.style.background_gradient(cmap='viridis')
        # Inside each block where you print the classification report, replace the print statement with the following:
        # Print the styled classification report
        st.write("Classification Report:")
        st.write(classification_rep_styled)
  

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return file_path

# Define the main function
def main():
    st.set_page_config(page_title="AutoML with Streamlit", layout="wide")  # Set page title and layout
    
    # Set background color of sidebar to primary color
    st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #3498db; /* Primary color */
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Set background color of main web area to light gray
    st.markdown("""
    <style>
        .block-container {
            background-color: #f9f9f9; /* Background color */
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Add a title section with accent color text
    st.title("AutoML with Json")
    st.write("This application allows you to upload an RTF file and perform AutoML tasks.")

    # Add a file uploader section
    st.sidebar.title("Upload RTF File")
    uploaded_file = st.sidebar.file_uploader("", type=["rtf"], help="Please upload your RTF file here")

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        rtf_parser(file_path)

# Entry point of the script
if __name__ == "__main__":
    main()