def plot_wordclouds(df):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    fake_text = ' '.join(df[df['label'] == 0]['clean_content'])
    real_text = ' '.join(df[df['label'] == 1]['clean_content'])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(WordCloud(width=400, height=400, background_color='white').generate(fake_text), interpolation='bilinear')
    plt.title('Fake News')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(WordCloud(width=400, height=400, background_color='white').generate(real_text), interpolation='bilinear')
    plt.title('Real News')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
import pandas as pd

# Step 1: Load dataset (combine Fake and True)
def load_data(fake_path, true_path):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df['label'] = 0  # 0 for fake
    true_df['label'] = 1  # 1 for real
    df = pd.concat([fake_df, true_df], ignore_index=True)
    return df

# Step 2: Preprocess text (to be implemented)
def preprocess_text(df):
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    import string

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Combine title and text columns if both exist
    if 'title' in df.columns and 'text' in df.columns:
        df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'text' in df.columns:
        df['content'] = df['text'].fillna('')
    elif 'title' in df.columns:
        df['content'] = df['title'].fillna('')
    else:
        raise ValueError('No text or title column found!')

    def clean_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    df['clean_content'] = df['content'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_content'])
    y = df['label']
    return X, y

# Step 3: Train model (to be implemented)

def train_model(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate model (to be implemented)

def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")

if __name__ == "__main__":
    df = load_data('Fake.csv', 'True.csv')
    print(df.head())
    print(df['label'].value_counts())
    print('Preprocessing text...')
    X, y = preprocess_text(df)
    print('Shape of features:', X.shape)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    plot_wordclouds(df)
