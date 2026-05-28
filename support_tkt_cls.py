import pandas as pd

# Load dataset
data = pd.read_csv('customer_support_tickets.csv')

# Show first 5 rows
print(data.head())
print(data.columns)

# Select important columns
data = data[['Ticket Subject', 'Ticket Description', 'Ticket Type', 'Ticket Priority']]

# Show selected columns
print(data.head())

#Checking Missing values
print(data.isnull().sum())

# Remove rows with missing values
data = data.dropna()

# Check again
print(data.isnull().sum())

import re

# Function to clean text
def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text
# Apply cleaning function
# Combine subject and description
data['Combined Text'] = (
    data['Ticket Subject'] + " " + data['Ticket Description']
)

# Clean combined text
data['Cleaned Text'] = data['Combined Text'].apply(clean_text)

# Show result
print(data[['Combined Text', 'Cleaned Text']].head())

# Show cleaned text
print(data[['Combined Text', 'Cleaned Text']].head())
print(data.head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer object
vectorizer = TfidfVectorizer()

# Convert cleaned text into numerical vectors
X = vectorizer.fit_transform(data['Cleaned Text'])

# Output labels
y = data['Ticket Type']

#print(X)

from sklearn.model_selection import train_test_split

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
print(X_train.shape)
print(X_test.shape)

from sklearn.naive_bayes import MultinomialNB

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)

print("Model Training Completed")

# Predict categories for test data
y_pred = model.predict(X_test)

print(y_pred)

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Custom ticket
sample_ticket = ["My payment was deducted but order not confirmed"]

# Clean text
cleaned_sample = [clean_text(sample_ticket[0])]

# Convert into vector
sample_vector = vectorizer.transform(cleaned_sample)

# Predict category
prediction = model.predict(sample_vector)

print("Predicted Ticket Type:", prediction[0])

# Priority prediction model

# Target for priority
y_priority = data['Ticket Priority']

# Split data
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X,
    y_priority,
    test_size=0.2,
    random_state=42
)

# Create model
priority_model = MultinomialNB()

# Train model
priority_model.fit(X_train_p, y_train_p)

# Predict
priority_pred = priority_model.predict(X_test_p)

# Accuracy
priority_accuracy = accuracy_score(y_test_p, priority_pred)

print("Priority Prediction Accuracy:", priority_accuracy)

# Predict priority for custom ticket
priority_result = priority_model.predict(sample_vector)

print("Predicted Priority:", priority_result[0])

import matplotlib.pyplot as plt

# Count ticket categories
data['Ticket Type'].value_counts().plot(kind='bar')

# Graph title
plt.title("Ticket Type Distribution")

# X-axis label
plt.xlabel("Ticket Type")

# Y-axis label
plt.ylabel("Count")

# Show graph
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(xticks_rotation='vertical')

plt.title("Confusion Matrix")

plt.show()

import pickle

# Save ticket type model
pickle.dump(model, open('ticket_type_model.pkl', 'wb'))

# Save vectorizer
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

print("Model Saved Successfully")