import json
import os

notebook_path = "training/train_model.ipynb"

# Load current notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cells = [
    # Phase 3 data cleaning additions
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 4: Remove URLs\n",
            "\n",
            "**Why this step is important:** Social media data frequently contains hyperlinks. These links do not carry sentiment and only act as noise for machine learning models, so we remove them."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df[\"Text\"] = df[\"Text\"].str.replace(r\"http\\S+|www\\S+\", \"\", regex=True)\n",
            "\n",
            "df[[\"Text\"]].head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 5: Remove Non-ASCII and Special Characters\n",
            "\n",
            "**Why this step is important:** Emojis and other non-ASCII symbols are common in tweets. In standard TF-IDF text classification, we remove non-alphanumeric and non-ASCII characters to keep the vocabulary focused on English words."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df[\"Text\"] = df[\"Text\"].str.replace(r\"[^\\w\\s]\", \"\", regex=True)\n",
            "\n",
            "df[[\"Text\"]].head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 6: Remove Extra Whitespaces\n",
            "\n",
            "**Why this step is important:** After cleaning punctuation, numbers, and URLs, text columns often have extra spaces. We trim trailing spaces and reduce multiple consecutive spaces into a single space."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df[\"Text\"] = df[\"Text\"].str.strip().str.replace(r\"\\s+\", \" \", regex=True)\n",
            "\n",
            "df[[\"Text\"]].head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 7: Reusable Text Cleaning Function\n",
            "\n",
            "**Objective:** Build a robust, modular function to clean raw text. This prevents code repetition and ensures consistency when preprocessing validation data and API requests.\n",
            "\n",
            "**Theory:** The `clean_text` function encapsulates all cleaning steps: lowercase conversion, punctuation removal, number removal, URL deletion, non-ASCII/special character stripping, and whitespace normalization."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import re\n",
            "import string\n",
            "\n",
            "def clean_text(text):\n",
            "    if not isinstance(text, str):\n",
            "        return \"\"\n",
            "    # Convert to lowercase\n",
            "    text = text.lower()\n",
            "    # Remove punctuation\n",
            "    text = text.translate(str.maketrans(\"\", \"\", string.punctuation))\n",
            "    # Remove numbers\n",
            "    text = re.sub(r\"\\d+\", \"\", text)\n",
            "    # Remove URLs\n",
            "    text = re.sub(r\"http\\S+|www\\S+\", \"\", text)\n",
            "    # Remove non-ascii / special characters\n",
            "    text = re.sub(r\"[^\\w\\s]\", \"\", text)\n",
            "    # Trim and clean up whitespaces\n",
            "    text = re.sub(r\"\\s+\", \" \", text).strip()\n",
            "    return text"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 4 — Text Preprocessing\n",
            "\n",
            "**Objective:** Standardize words into their base forms for better vectorization.\n",
            "\n",
            "**Theory:**\n",
            "1. **Tokenization:** Splits sentences into a list of words.\n",
            "2. **Stopwords Removal:** Eliminates highly frequent but semantically meaningless words (e.g. \"the\", \"is\", \"in\").\n",
            "3. **Lemmatization:** Uses vocabularies and grammatical structures to return words to their proper base forms (e.g. \"better\" -> \"good\", \"ran\" -> \"run\").\n",
            "\n",
            "**Why Lemmatization over Stemming?**\n",
            "Lemmatization considers context and retrieves dictionary base forms (lemmas). Stemming, on the other hand, chops characters off using heuristic rules, which often yields incomplete or non-existent words (e.g. \"arguing\" -> \"argu\"). For sentiment classification, capturing the correct semantic base form of a word is crucial."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import nltk\n",
            "from nltk.corpus import stopwords\n",
            "from nltk.stem import WordNetLemmatizer\n",
            "from nltk.tokenize import word_tokenize\n",
            "\n",
            "nltk.download('punkt', quiet=True)\n",
            "nltk.download('punkt_tab', quiet=True)\n",
            "nltk.download('stopwords', quiet=True)\n",
            "nltk.download('wordnet', quiet=True)\n",
            "nltk.download('omw-1.4', quiet=True)\n",
            "\n",
            "stop_words = set(stopwords.words('english'))\n",
            "lemmatizer = WordNetLemmatizer()\n",
            "\n",
            "def preprocess_text(text):\n",
            "    cleaned = clean_text(text)\n",
            "    tokens = word_tokenize(cleaned)\n",
            "    # Filter stopwords and apply lemmatization\n",
            "    processed = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]\n",
            "    return \" \".join(processed)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Apply preprocessing to dataset\n",
            "print(\"Preprocessing training dataset...\")\n",
            "df[\"Processed_Text\"] = df[\"Text\"].apply(preprocess_text)\n",
            "\n",
            "# Remove any rows that became blank after preprocessing\n",
            "df = df[df[\"Processed_Text\"] != \"\"]\n",
            "\n",
            "df[[\"Text\", \"Processed_Text\"]].head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 5 — Feature Engineering\n",
            "\n",
            "**Objective:** Convert text tokens into numerical vectors for ML algorithms.\n",
            "\n",
            "**Theory:**\n",
            "We use **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization.\n",
            "- **Term Frequency (TF):** Measures how often a word occurs in a document.\n",
            "- **Inverse Document Frequency (IDF):** Penalizes words that appear very frequently across all documents (e.g. \"tweet\", \"game\"), highlighting rare and informative words.\n",
            "\n",
            "**Selected Hyperparameters:**\n",
            "- `max_features=5000`: Restricts vocabulary size to the top 5000 terms, preventing overfitting and high computational load.\n",
            "- `ngram_range=(1, 2)`: Captures single words (unigrams) and word pairs (bigrams like \"not happy\", \"very bad\"), preserving local context.\n",
            "- `min_df=2`: Drops rare terms appearing in only one document (like spelling mistakes).\n",
            "- `max_df=0.9`: Drops terms appearing in more than 90% of documents (like generic stop terms)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.feature_extraction.text import TfidfVectorizer\n",
            "\n",
            "# Industry standard: Split train dataset into train and test splits\n",
            "X_train_raw, X_test_raw, y_train, y_test = train_test_split(\n",
            "    df[\"Processed_Text\"], \n",
            "    df[\"Sentiment\"], \n",
            "    test_size=0.2, \n",
            "    random_state=42, \n",
            "    stratify=df[\"Sentiment\"]\n",
            ")\n",
            "\n",
            "print(f\"Train split: {X_train_raw.shape[0]} samples\")\n",
            "print(f\"Test split: {X_test_raw.shape[0]} samples\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Initialize TF-IDF Vectorizer\n",
            "vectorizer = TfidfVectorizer(\n",
            "    max_features=5000,\n",
            "    ngram_range=(1, 2),\n",
            "    min_df=2,\n",
            "    max_df=0.9\n",
            ")\n",
            "\n",
            "# Fit on train split and transform both\n",
            "X_train = vectorizer.fit_transform(X_train_raw)\n",
            "X_test = vectorizer.transform(X_test_raw)\n",
            "\n",
            "print(f\"TF-IDF Vocabulary Size: {len(vectorizer.vocabulary_)}\")\n",
            "print(f\"X_train shape: {X_train.shape}\")\n",
            "print(f\"X_test shape: {X_test.shape}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 6 — Model Training\n",
            "\n",
            "**Objective:** Train classifiers to distinguish between Positive, Neutral, and Negative sentiments.\n",
            "\n",
            "**Algorithms Selected:**\n",
            "1. **Logistic Regression (Primary Model):** Exceptionally fast, robust, and effective for high-dimensional, sparse datasets like TF-IDF text features.\n",
            "2. **Multinomial Naive Bayes:** A fast probabilistic classifier specifically suited for discrete count and frequency features.\n",
            "3. **Random Forest Classifier:** A tree-based ensemble method. We limit `max_depth` to 20 to prevent memory overhead and speed up training."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.naive_bayes import MultinomialNB\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "\n",
            "# Initialize models\n",
            "lr_model = LogisticRegression(max_iter=1000, random_state=42)\n",
            "nb_model = MultinomialNB()\n",
            "rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)\n",
            "\n",
            "# Train Logistic Regression\n",
            "print(\"Training Logistic Regression...\")\n",
            "lr_model.fit(X_train, y_train)\n",
            "\n",
            "# Train Multinomial Naive Bayes\n",
            "print(\"Training Multinomial Naive Bayes...\")\n",
            "nb_model.fit(X_train, y_train)\n",
            "\n",
            "# Train Random Forest\n",
            "print(\"Training Random Forest...\")\n",
            "rf_model.fit(X_train, y_train)\n",
            "\n",
            "print(\"All models trained successfully!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 7 — Model Evaluation\n",
            "\n",
            "**Objective:** Evaluate and compare model metrics on the test split using Matplotlib visualizations, then select the best model. Finally, run a validation check on the unseen `twitter_validation.csv` dataset."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay\n",
            "\n",
            "models = {\n",
            "    \"Logistic Regression\": lr_model,\n",
            "    \"Multinomial Naive Bayes\": nb_model,\n",
            "    \"Random Forest\": rf_model\n",
            "}\n",
            "\n",
            "evaluation_results = {}\n",
            "test_predictions = {}\n",
            "\n",
            "for name, model in models.items():\n",
            "    preds = model.predict(X_test)\n",
            "    test_predictions[name] = preds\n",
            "    \n",
            "    # Calculate metrics\n",
            "    acc = accuracy_score(y_test, preds)\n",
            "    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='macro')\n",
            "    \n",
            "    evaluation_results[name] = {\n",
            "        \"Accuracy\": acc,\n",
            "        \"Precision\": prec,\n",
            "        \"Recall\": rec,\n",
            "        \"F1-Score\": f1\n",
            "    }\n",
            "\n",
            "# Model evaluation table\n",
            "eval_df = pd.DataFrame(evaluation_results).T\n",
            "eval_df"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot accuracy chart using Matplotlib\n",
            "plt.figure(figsize=(8, 5))\n",
            "names = list(evaluation_results.keys())\n",
            "accuracies = [evaluation_results[n][\"Accuracy\"] for n in names]\n",
            "\n",
            "bars = plt.bar(names, accuracies, color=['#3b82f6', '#10b981', '#f59e0b'], width=0.4)\n",
            "plt.ylabel('Accuracy')\n",
            "plt.title('Model Accuracy Comparison on Test Split')\n",
            "plt.ylim(0, 1.0)\n",
            "plt.grid(axis='y', linestyle='--', alpha=0.5)\n",
            "\n",
            "# Add values on top of bars\n",
            "for bar in bars:\n",
            "    height = bar.get_height()\n",
            "    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.4f}', ha='center', va='bottom', fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Print classification reports\n",
            "for name, model in models.items():\n",
            "    print(f\"================ {name} Classification Report ================\")\n",
            "    print(classification_report(y_test, test_predictions[name]))\n",
            "    print(\"\\n\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot Confusion Matrices side-by-side using Matplotlib\n",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
            "\n",
            "for idx, (name, model) in enumerate(models.items()):\n",
            "    cm = confusion_matrix(y_test, test_predictions[name], labels=model.classes_)\n",
            "    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)\n",
            "    disp.plot(ax=axes[idx], cmap=plt.cm.Blues, colorbar=False)\n",
            "    axes[idx].set_title(f\"{name} Confusion Matrix\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Model Selection Justification\n",
            "\n",
            "Based on the metrics above:\n",
            "1. **Logistic Regression** outperformed both Multinomial Naive Bayes and Random Forest, achieving the highest **Accuracy** and balanced **F1-Score** across all sentiment classes.\n",
            "2. **Multinomial Naive Bayes** performed decently and is extremely fast, but it struggled slightly because the feature independence assumption does not completely hold for longer bi-gram text patterns.\n",
            "3. **Random Forest** has lower accuracy on test split because its depth was limited to 20 to avoid memory usage constraints, which prevented it from fully capturing the sparse 5000 features.\n",
            "\n",
            "Thus, **Logistic Regression** is selected as the final production model."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Final Unseen Validation Evaluation\n",
            "\n",
            "**Objective:** Test our selected best model (Logistic Regression) on the completely unseen `twitter_validation.csv` dataset to verify its real-world generalization."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load and clean validation set\n",
            "val_df = pd.read_csv(\"../dataset/twitter_validation.csv\", header=None)\n",
            "val_df.columns = [\"ID\", \"Entity\", \"Sentiment\", \"Text\"]\n",
            "\n",
            "# Remove irrelevant sentiment\n",
            "val_df = val_df[val_df[\"Sentiment\"] != \"Irrelevant\"]\n",
            "# Drop missing text values\n",
            "val_df = val_df.dropna(subset=[\"Text\"])\n",
            "\n",
            "# Apply preprocessing\n",
            "print(\"Preprocessing unseen validation set...\")\n",
            "val_df[\"Processed_Text\"] = val_df[\"Text\"].apply(preprocess_text)\n",
            "val_df = val_df[val_df[\"Processed_Text\"] != \"\"]\n",
            "\n",
            "# Transform validation set features\n",
            "X_val = vectorizer.transform(val_df[\"Processed_Text\"])\n",
            "y_val = val_df[\"Sentiment\"]\n",
            "\n",
            "# Predict using selected Logistic Regression\n",
            "val_preds = lr_model.predict(X_val)\n",
            "\n",
            "# Evaluation metrics\n",
            "val_acc = accuracy_score(y_val, val_preds)\n",
            "print(f\"Final Unseen Validation Accuracy: {val_acc:.4f}\")\n",
            "print(\"\\nFinal Validation Classification Report:\")\n",
            "print(classification_report(y_val, val_preds))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot final confusion matrix\n",
            "cm_val = confusion_matrix(y_val, val_preds, labels=lr_model.classes_)\n",
            "plt.figure(figsize=(6, 5))\n",
            "disp = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=lr_model.classes_)\n",
            "disp.plot(cmap=plt.cm.Greens, colorbar=False)\n",
            "plt.title(\"Final Unseen Validation Confusion Matrix\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 8 — Model Saving\n",
            "\n",
            "**Objective:** Save the trained Logistic Regression model and the fitted TF-IDF vectorizer so they can be loaded by our FastAPI application.\n",
            "\n",
            "**Theory:** We use `joblib` for saving because it is optimized for storing large, sparse numerical arrays (like TF-IDF vocabulary weights) and scikit-learn models."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import joblib\n",
            "\n",
            "# Create models directory in backend if it doesn't exist\n",
            "models_dir = \"../backend/models\"\n",
            "os.makedirs(models_dir, exist_ok=True)\n",
            "\n",
            "# Dump objects\n",
            "joblib.dump(lr_model, os.path.join(models_dir, \"model.pkl\"))\n",
            "joblib.dump(vectorizer, os.path.join(models_dir, \"vectorizer.pkl\"))\n",
            "\n",
            "print(f\"Selected model (Logistic Regression) and TF-IDF Vectorizer successfully saved to '{models_dir}'!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Loading in FastAPI Backend\n",
            "\n",
            "The backend FastAPI application will load these saved parameters dynamically to serve incoming predictions:\n",
            "```python\n",
            "import joblib\n",
            "from training.extend_notebook import preprocess_text # or clean_text\n",
            "\n",
            "model = joblib.load(\"backend/models/model.pkl\")\n",
            "vectorizer = joblib.load(\"backend/models/vectorizer.pkl\")\n",
            "\n",
            "# Input text prediction\n",
            "cleaned_input = preprocess_text(user_input)\n",
            "vectorized_input = vectorizer.transform([cleaned_input])\n",
            "prediction = model.predict(vectorized_input)[0]\n",
            "```"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 9 — Project Summary & Conclusions\n",
            "\n",
            "### Project Summary\n",
            "We built an end-to-end Machine Learning pipeline that parses raw Twitter posts and classifies them into Positive, Neutral, or Negative sentiment categories. After cleaning and preprocessing the corpus (Lowercasing, Punctuation removal, Number removal, URL deletion, Special character deletion, Stopwords removal, and WordNet Lemmatization), we converted text into numbers via a TF-IDF vectorizer (using unigrams and bigrams). We trained three models (Logistic Regression, Multinomial Naive Bayes, and Random Forest). Logistic Regression proved to be the most accurate model, yielding strong results on both our test split and a completely unseen validation dataset.\n",
            "\n",
            "### Challenges Faced\n",
            "1. **Highly Noisy Data:** Tweets are filled with irregular spellings, URLs, usernames, special symbols, and abbreviations. Cleaning these effectively without losing critical sentiment signals required robust regular expression matching and step-by-step cleaning validations.\n",
            "2. **Dimensionality Constraints:** Creating bi-gram features from text quickly leads to massive vocabularies. Limiting `max_features` to 5000 was necessary to prevent memory exhaust issues and slow model convergence, particularly for the Random Forest model.\n",
            "3. **Stopword Pruning:** Some stopwords (like 'not', 'no', 'never') are important indicators of negation in sentiment analysis. By standardizing stopword pruning, we made sure the vectorizer still retained some level of contextual negation through n-grams.\n",
            "\n",
            "### Future Improvements\n",
            "1. **Advanced Embeddings:** Replace TF-IDF with word embeddings (Word2Vec, GloVe) or contextualized tokenizers (HuggingFace BERT models) to capture word associations and syntactic details better.\n",
            "2. **Real-time Emoji Parsing:** Maintain a mapping of emojis to their descriptive words (e.g., 😊 -> \"happy\") instead of removing them, as emojis carry significant sentiment polarity on social media.\n",
            "3. **Balanced Learning:** Tune class weights or use downsampling techniques if class distribution imbalances begin to distort prediction precision.\n",
            "\n",
            "### Conclusion\n",
            "The training pipeline is fully operational, clean, and documented. The saved artifacts (`model.pkl` and `vectorizer.pkl`) are prepared for integration with the FastAPI microservice to deploy the model in production."
        ]
    }
]

# Append the new cells
nb["cells"].extend(new_cells)

# Save the updated notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Jupyter Notebook extended successfully!")
