'''     A symptom-to-disease classifier using machine learning techniques.
        Given a symptom description, the classifier predicts the most likely associated disease.    '''

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Loading dataset
data = pd.read_csv('Symptom2Disease.csv')

# Preprocessing
data['text'] = data['text'].str.lower()  # Converting to lowercase
data['text'] = data['text'].str.replace('[^\w\s]', '')  # Removing punctuation

# Separate features and labels
X = data['text']
y = data['label']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction
vectoriser = CountVectorizer(stop_words='english')  # Apply stop-word removal
X_train_features = vectoriser.fit_transform(X_train)
X_test_features = vectoriser.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_features, y_train)

# Example prediction
new_symptom = ["I have a headache and fever"]
new_symptom_features = vectoriser.transform(new_symptom)
predicted_disease = model.predict(new_symptom_features)
print(predicted_disease)

# Model evaluation
accuracy = model.score(X_test_features, y_test)
print("Accuracy:", accuracy)

# Hyperparameter tuning, cross-validation
parameters = {'alpha': [0.1, 1.0, 10.0]}  # Parameter grid to tune 'alpha' for handling zero probabilities
grid_search = GridSearchCV(model, parameters, cv=5)  # Perform 5-fold cross-validation
grid_search.fit(X_train_features, y_train)

# Best hyperparameter values and performance
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Final evaluation on test set with best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_features)
print(classification_report(y_test, y_pred))