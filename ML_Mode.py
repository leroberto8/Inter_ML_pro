# Import necessary libraries
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  

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
    # Initialize models  
    models = {  
        'Logistic Regression': LogisticRegression(max_iter=1000),  
        'Decision Tree': DecisionTreeClassifier(),  
        'Random Forest': RandomForestClassifier()  
    }  

    # Train and evaluate each model  
    for model_name, model in models.items():  
        model.fit(X_train, y_train)  # Train the model  
        y_pred = model.predict(X_test)  # Make predictions  

        # Calculate performance metrics  
        accuracy = accuracy_score(y_test, y_pred)  
        precision = precision_score(y_test, y_pred)  
        recall = recall_score(y_test, y_pred)  
        f1 = f1_score(y_test, y_pred)  

        # Print the evaluation results  
        print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, "  
              f"Recall: {recall:.2f}, F1 Score: {f1:.2f}")  

# Execute the functions  
if __name__ == "__main__":  
    X_train, X_test, y_train, y_test = split_data(data)  
    evaluate_models(X_train, X_test, y_train, y_test)