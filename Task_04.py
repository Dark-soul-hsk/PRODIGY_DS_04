import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import re
import string


TRAIN_FILE = 'twitter_training.csv'
VAL_FILE = 'twitter_validation.csv'
COLUMNS = ['tweet_id', 'entity', 'sentiment', 'text']


def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
    else:
        text = ''
    return text


try:
    train_df = pd.read_csv(TRAIN_FILE, names=COLUMNS, encoding='utf-8')
    val_df = pd.read_csv(VAL_FILE, names=COLUMNS, encoding='utf-8')
except FileNotFoundError:
    print("Error: Ensure twitter_training.csv and twitter_validation.csv are in the current directory.")
    exit()

train_df['text'] = train_df['text'].apply(preprocess_text)
val_df['text'] = val_df['text'].apply(preprocess_text)

train_df.dropna(subset=['text'], inplace=True)
val_df.dropna(subset=['text'], inplace=True)

X_train = train_df['text']
y_train = train_df['sentiment']
X_val = val_df['text']
y_val = val_df['sentiment']


model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LinearSVC(C=1.0, dual=True, random_state=42)),
])


print("Starting Model Training...")
model.fit(X_train, y_train)
print("Training Complete.")


y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

print(f"Model Accuracy on Validation Set: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred))