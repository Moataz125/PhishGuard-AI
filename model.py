
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_model(csv_path):

    # STEP 1: Load Dataset
    df = pd.read_csv(
        csv_path,
        usecols=["Email Text", "Email Type"]
    )

    # STEP 2: Clean Dataset
    df["Email Text"] = (
        df["Email Text"].fillna("").astype(str).str.strip()
    )

    df["label"] = (
        df["Email Type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "safe email": 0,
            "phishing email": 1
        })
    )

    df = df.dropna(subset=["label"])
    df = df[df["Email Text"] != ""].copy()

    # Remove duplicates and conflicting labels
    df["text_key"] = (
        df["Email Text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    counts = df.groupby("text_key")["label"].transform("nunique")

    df = df[counts == 1].drop_duplicates("text_key")

    X = df["Email Text"]
    y = df["label"].astype(int)

    # STEP 3: Split Dataset
    # 80% development, 20% final testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Validation split from the development data
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    # STEP 4: Feature Extraction
    features = FeatureUnion([
        (
            "word_features",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True
            )
        ),
        (
            "character_features",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 4),
                max_features=10000,
                min_df=2,
                sublinear_tf=True
            )
        )
    ])

    # STEP 5: Machine Learning Model
    model = Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42
            )
        )
    ])

    # Train Model
    model.fit(X_fit, y_fit)

    # STEP 6: Select Classification Threshold
    val_probabilities = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.50
    best_f1 = -1

    for i in range(35, 76, 5):

        threshold = i / 100

        val_predictions = (
            val_probabilities >= threshold
        ).astype(int)

        recall = recall_score(y_val, val_predictions)
        f1 = f1_score(y_val, val_predictions)

        # Prioritize detecting phishing emails
        if recall < 0.95:
            continue

        if (
            f1 > best_f1 or
            (
                f1 == best_f1 and
                abs(threshold - 0.50) <
                abs(best_threshold - 0.50)
            )
        ):
            best_f1 = f1
            best_threshold = threshold

    model.threshold_ = best_threshold

    # STEP 7: Final Evaluation
    test_probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions)
    }

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    return model, metrics, matrix, len(df)


def analyze_email(model, email_text):

    # Get phishing score
    probability = float(
        model.predict_proba([email_text])[0][1]
    )

    # Apply validation-selected threshold
    prediction = int(
        probability >= model.threshold_
    )

    return prediction, probability
