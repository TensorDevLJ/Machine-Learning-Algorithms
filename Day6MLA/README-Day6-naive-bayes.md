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

---

## 📈 Comparison: Naive Bayes vs. Decision Trees

In text classification, choosing the right model is critical. Here is why we often prefer Naive Bayes over Tree-based models for high-dimensional text:

| Feature | Naive Bayes | Decision Trees |
| :--- | :--- | :--- |
| **Speed** | **Extremely Fast** (Linear complexity) | **Slower** (O(depth) or worse in high dimensions) |
| **Data Volume** | Works well with small/sparse datasets | Needs more data to avoid **Overfitting** |
| **Independence** | Assumes features are independent | Captures complex **Feature Interactions** |
| **Best For** | Text, Spam, Medical Diagnosis | Tabular Data, Credit Scoring, Fraud |

---

## 🌍 Real-World Applications

Naive Bayes is a workhorse in production environments where speed and efficiency are priorities:

* 📩 **Spam Email Filtering:** Identifying junk mail based on word frequencies.
* 🎭 **Sentiment Analysis:** Classifying reviews as positive, negative, or neutral.
* 📰 **News Classification:** Sorting articles into categories like Sports, Tech, or Politics.
* 🔍 **Fake News Detection:** Identifying patterns in misleading headlines.
* 📂 **Document Categorization:** Organizing large-scale digital libraries.
* 🩺 **Medical Diagnosis:** Predicting disease probability based on independent symptoms.

---

## ✅ Advantages

* **Simple & Fast:** Easy to implement and computationally inexpensive.
* **High-Dimensionality:** Excels in text classification where the number of features (words) is very large.
* **Small Data friendly:** Requires significantly less training data than complex models like Neural Networks.
* **Robust Baseline:** Often performs surprisingly well even when the "Independence" assumption is violated.

---

## ⚠️ Limitations

> [!IMPORTANT]
> Understanding the constraints of your model is key to being a good ML Engineer.

* **Feature Independence:** It assumes every word is independent, missing the context provided by word pairings (e.g., "Not good").
* **Correlated Features:** Performance can drop if features are highly dependent on each other.
* **Zero Frequency Issue:** If a word appears in the test set but not in training, the probability becomes zero (mitigated by **Laplace Smoothing**).
* **Preprocessing Dependent:** Highly sensitive to how you clean your text (Stopwords, Stemming, etc.).

---

## 🛠️ Tools & Libraries

* **Python:** Primary language.
* **Pandas:** For data manipulation and structuring.
* **Scikit-learn:** For `CountVectorizer` and `MultinomialNB`.
* **Matplotlib/Seaborn:** For visualizing the Confusion Matrix.

------------------------------------------------------------------------

## ✍️ Author

Likhitha J\
Machine Learning Learning Journey
