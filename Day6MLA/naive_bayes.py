import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset
# For your GitHub, you can use the SMS Spam Collection dataset from UCI
data = {
    'text': [
        'Hey, are we still meeting for lunch?', 
        'WINNER! Claim your prize of $1000 now!', 
        'Call me back when you get this.', 
        'URGENT: Your account has been compromised. Click here.',
        'Can you send me the files by EOD?',
        'Free entry into the contest. Text STOP to opt out.',
        'Lunch today at the usual place?',
        'Congratulations! You have been selected for a cash prize.',
        'Meeting at 3pm in the conference room.',
        'Claim your free trial now, no credit card required!'
    ],
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam']
}

df = pd.DataFrame(data)

# 2. Vectorization: Convert text to numbers
# Adding 'stop_words' helps filter out noise like 'is', 'the', 'at'

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model 
# MultinomialNB handles discrete counts perfectly
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred = model.predict(X_test)

print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Real-world test
def predict_spam(message):
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    return prediction[0]

test_msg = "Claim your free 1000 dollars by clicking this link!"
print(f"Test Message: '{test_msg}'")
print(f"Prediction: {predict_spam(test_msg).upper()}")

# 7. Visualization: Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Spam Classifier Confusion Matrix')
plt.show()