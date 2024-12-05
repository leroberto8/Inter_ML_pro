# ML_Mode.py  

import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  
from sklearn.preprocessing import OneHotEncoder  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline  

# Load the cleaned dataset  
data = pd.read_csv('cleaned_in_vehicle_coupon_data.csv')  

# Task 1: Split the Data  
def split_data(data):  
    # Define features (X) and target variable (y)  
    X = data.drop('Y', axis=1)  # Drop the target variable from features  
    y = data['Y']  # Target variable  

    # Split the data into training (80%) and test (20%) sets  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
    return X_train, X_test, y_train, y_test  

# Task 2: Implement and Evaluate Models  
def evaluate_models(X_train, X_test, y_train, y_test):  
    # Identify categorical and numerical features  
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()  
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()  

    # Define the Column Transformer for preprocessing  
    preprocessor = ColumnTransformer(  
        transformers=[  
            ('num', 'passthrough', numerical_features),   
            ('cat', OneHotEncoder(), categorical_features)   
        ]  
    )  

    # Initialize models with pipelines  
    models = {  
        'Logistic Regression': Pipeline(steps=[('preprocessor', preprocessor),  
                                               ('classifier', LogisticRegression(max_iter=1000))]),  
        'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor),   
                                          ('classifier', DecisionTreeClassifier())]),  
        'Random Forest': Pipeline(steps=[('preprocessor', preprocessor),   
                                          ('classifier', RandomForestClassifier())])  
    }  

    # Store metrics for plotting  
    metrics = {   
        'Model': [],  
        'Accuracy': [],  
        'Precision': [],  
        'Recall': [],  
        'F1 Score': []  
    }  

    # Train and evaluate each model  
    for model_name, model in models.items():  
        model.fit(X_train, y_train)  # Train the model  
        y_pred = model.predict(X_test)  # Make predictions  

        # Calculate performance metrics  
        accuracy = accuracy_score(y_test, y_pred)  
        precision = precision_score(y_test, y_pred, average='binary')  # Adjust for multiclass if needed  
        recall = recall_score(y_test, y_pred, average='binary')  # Adjust for multiclass if needed  
        f1 = f1_score(y_test, y_pred, average='binary')  # Adjust for multiclass if needed  

        # Append results to metrics dictionary  
        metrics['Model'].append(model_name)  
        metrics['Accuracy'].append(accuracy)  
        metrics['Precision'].append(precision)  
        metrics['Recall'].append(recall)  
        metrics['F1 Score'].append(f1)  

        # Print the evaluation results  
        print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, "  
              f"Recall: {recall:.2f}, F1 Score: {f1:.2f}")  

    return metrics  

# Plotting function  
def plot_metrics(metrics):  
    df_metrics = pd.DataFrame(metrics)  
    df_metrics.set_index('Model', inplace=True)  

    # Plot metrics  
    df_metrics.plot(kind='bar', figsize=(10, 6))  
    plt.title('Model Performance Metrics')  
    plt.xlabel('Models')  
    plt.ylabel('Scores')  
    plt.ylim(0, 1)  
    plt.xticks(rotation=45)  
    plt.legend(title='Metrics')  
    plt.tight_layout()  
    plt.show()  

# Execute the functions  
if __name__ == "__main__":  
    X_train, X_test, y_train, y_test = split_data(data)  
    metrics = evaluate_models(X_train, X_test, y_train, y_test)  
    plot_metrics(metrics)