# Final_rep.py  

import pandas as pd  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import GridSearchCV  

# Load the cleaned dataset  
data = pd.read_csv('cleaned_in_vehicle_coupon_data.csv')  

# Assuming you have already split the data and trained the best model  
X = data.drop('Y', axis=1)  
y = data['Y'].map({'No': 0, 'Yes': 1})  # Convert target variable to numeric  

# Fine-Tune the Best Model  
def fine_tune_model(X_train, y_train):  
    """Fine-tune the Random Forest model using GridSearchCV."""  
    param_grid = {  
        'n_estimators': [50, 100, 200],  
        'max_depth': [None, 10, 20, 30],  
        'min_samples_split': [2, 5, 10]  
    }  

    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1')  
    grid_search.fit(X_train, y_train)  

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_  

# Evaluate the Tuned Model  
def evaluate_model(model, X_test, y_test):  
    """Evaluate the tuned model on the test data."""  
    y_pred = model.predict(X_test)  
    
    accuracy = accuracy_score(y_test, y_pred)  
    precision = precision_score(y_test, y_pred)  
    recall = recall_score(y_test, y_pred)  
    f1 = f1_score(y_test, y_pred)  

    return accuracy, precision, recall, f1  

# Generate a Final Report  
def generate_report(best_model, best_params, accuracy, precision, recall, f1):  
    """Create a summary report of the findings."""  
    report = {  
        'Dataset Summary': data.describe(),  
        'Best Model Parameters': best_params,  
        'Best Model Performance': {  
            'Accuracy': accuracy,  
            'Precision': precision,  
            'Recall': recall,  
            'F1 Score': f1  
        }  
    }  

    # Save the report to a file  
    with open('final_report.txt', 'w') as f:  
        for key, value in report.items():  
            f.write(f"{key}:\n{value}\n\n")  

if __name__ == "__main__":  
    # Assuming you have already split your data into training and testing sets  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjust as needed  

    best_model, best_params, best_score = fine_tune_model(X_train, y_train)  
    accuracy, precision, recall, f1 = evaluate_model(best_model, X_test, y_test)  
    
    print(f"Best parameters: {best_params}")  
    print(f"Best cross-validation F1 score: {best_score:.2f}")  
    print(f"Tuned Random Forest -)