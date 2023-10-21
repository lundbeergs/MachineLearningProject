

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

if __name__ == '__main__':
    # Load in the dataset
    x_test_classification = np.load('Xtest_Classification1.npy')
    x_train_classification = np.load('Xtrain_Classification1.npy')
    y_train_classification = np.load('ytrain_Classification1.npy')

    # Reshaping and normalizing the dataset
    x_train_class_normalized = x_train_classification / 255.0
    x_train_class_normalized_reshaped = x_train_class_normalized.reshape(-1, 3 * 28 * 28)
    x_test_class_normalized = x_test_classification / 255.0
    x_test_class_normalized_reshaped = x_test_class_normalized.reshape(-1, 3 * 28 * 28)

    # Split the dataset into training
    X_train, X_temp, y_train, y_temp = train_test_split(x_train_class_normalized_reshaped, y_train_classification, test_size=0.3, random_state=39)

    # Split part of the training set into validation
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=39)

    # Define hyperparameters
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30]
    }

    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=39, n_jobs=-1)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, scoring='balanced_accuracy', n_jobs=-1)

    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters from the grid search
    best_params = grid_search.best_params_
    print(f'Best Hyperparameters: {best_params}')

    # Initialize a Random Forest Classifier with the best hyperparameters
    best_rf_classifier = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=39, n_jobs=-1)

    # Train the Random Forest Classifier with the best hyperparameters
    best_rf_classifier.fit(X_train, y_train)

    # Make predictions on the validation set
    y_valid_pred = best_rf_classifier.predict(X_valid)

    # Calculate balanced accuracy on the validation set
    balanced_accuracy = balanced_accuracy_score(y_valid, y_valid_pred)

    print(f'Validation Balanced Accuracy: {balanced_accuracy}')

    # Make predictions on the test set
    y_test_pred = best_rf_classifier.predict(X_test)

    # Calculate balanced accuracy on the test set
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)

    print(f'Test Balanced Accuracy: {test_balanced_accuracy}')