# Day 6: Naive Bayes --- Spam Email Classification 📧

## 📌 Overview

Welcome to Day 6 of the Machine Learning Journey! Today's focus is on
**Naive Bayes**, a powerful probabilistic algorithm widely used for text
classification tasks like spam detection and sentiment analysis.

Even in the era of advanced deep learning models, Naive Bayes remains a
strong baseline because of its speed, simplicity, and efficiency.

------------------------------------------------------------------------

## 🧠 Core Concepts

### 1. What is Naive Bayes?

Naive Bayes is a probabilistic classifier based on **Bayes' Theorem**.

-   **"Bayes"**: It uses probability to make decisions.
-   **"Naive"**: It assumes that every feature (word) is independent of
    others.

For example, it treats "Winner" and "Prize" as independent indicators of
spam, even though they often appear together.

------------------------------------------------------------------------

## 📐 Bayes' Theorem (Mathematical Foundation)

Bayes' Theorem:

P(A \| B) = (P(B \| A) \* P(A)) / P(B)

Where: - P(A \| B) = Posterior Probability - P(B \| A) = Likelihood -
P(A) = Prior Probability - P(B) = Evidence

In classification:

P(Class \| Features) = (P(Features \| Class) \* P(Class)) / P(Features)

For spam detection:

P(Spam \| Words) = (P(Words \| Spam) \* P(Spam)) / P(Words)

The class with the highest probability is selected.

------------------------------------------------------------------------

## 🧮 Mathematical Intuition in Text Classification

For a message containing words w1, w2, ..., wn:

P(Spam \| Message) ∝\
P(Spam) × P(w1 \| Spam) × P(w2 \| Spam) × ... × P(wn \| Spam)

The same is calculated for Not Spam.

The larger probability determines the predicted class.

------------------------------------------------------------------------

## 🔧 Laplace Smoothing (Zero Probability Fix)

If a word never appeared in the training data, its probability becomes
0.\
Since probabilities are multiplied, the final result would also become
0.

Laplace Smoothing fixes this:

P(word \| class) =\
(Count(word, class) + 1) / (Total words in class + Vocabulary size)

This ensures no probability becomes zero.

------------------------------------------------------------------------

## 🛠️ The Pipeline: From Text to Numbers

Machine Learning models process numbers, not text. We use
**CountVectorizer** to convert text into numerical features.

Steps:

1.  Tokenization -- Splitting text into words\
2.  Stopword Removal -- Removing common words like "the", "is", "at"\
3.  Vectorization -- Creating a Bag-of-Words matrix

------------------------------------------------------------------------

## 💻 Implementation Snippet

``` python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

model.fit(X_train, y_train)
```

------------------------------------------------------------------------

## 📊 Project Steps

1.  Data Setup -- Created labeled SMS/Email dataset\
2.  Preprocessing -- Used CountVectorizer for tokenization and stopword
    removal\
3.  Modeling -- Applied MultinomialNB (designed for word counts)\
4.  Evaluation -- Checked Accuracy, Precision, Recall via classification
    report\
5.  Inference -- Built prediction function for custom user input

------------------------------------------------------------------------

## 📈 Comparison: Why Not Decision Trees?

Feature Comparison:

Speed\
- Naive Bayes: Extremely Fast\
- Decision Trees: Slower (especially in high dimensions)

Data Volume\
- Naive Bayes: Works well with small datasets\
- Decision Trees: Needs more data to avoid overfitting

Independence\
- Naive Bayes: Assumes features are independent\
- Decision Trees: Captures feature interactions

Best For\
- Naive Bayes: Text, Spam Detection, Medical Diagnosis\
- Decision Trees: Tabular Data, Credit Scoring

------------------------------------------------------------------------

## 🌍 Real-World Applications

-   Spam Email Filtering\
-   Sentiment Analysis\
-   News Classification\
-   Fake News Detection\
-   Document Categorization\
-   Medical Diagnosis

------------------------------------------------------------------------

## ✅ Advantages

-   Simple and easy to implement\
-   Extremely fast\
-   Works well with high-dimensional text data\
-   Requires less training data\
-   Performs surprisingly well despite independence assumption

------------------------------------------------------------------------

## ⚠️ Limitations

-   Assumes feature independence\
-   Not ideal for highly correlated features\
-   Struggles with complex non-linear relationships\
-   Performance depends on quality of preprocessing

------------------------------------------------------------------------

## 🛠️ Tools & Libraries

Language: Python\
Libraries: scikit-learn, pandas\
Vectorization: CountVectorizer\
Environment: VS Code / Jupyter Notebook

------------------------------------------------------------------------

## ✍️ Author

Likhitha J\
Machine Learning Learning Journey
