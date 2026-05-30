from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .features import build_bow_vectorizer, build_tfidf_vectorizer


def make_nb_bow_pipeline():
    """Create the BoW + Multinomial Naive Bayes baseline model."""
    return Pipeline([
        ("vectorizer", build_bow_vectorizer(min_df=2, max_df=0.95, max_features=10000)),
        ("model", MultinomialNB()),
    ])


def make_logreg_tfidf_pipeline():
    """Create the TF-IDF + Logistic Regression model."""
    return Pipeline([
        ("vectorizer", build_tfidf_vectorizer(min_df=2, max_df=0.95, max_features=10000)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def make_linear_svm_tfidf_pipeline():
    """Create the TF-IDF + Linear SVM model."""
    return Pipeline([
        ("vectorizer", build_tfidf_vectorizer(min_df=2, max_df=0.95, max_features=10000)),
        ("model", LinearSVC(class_weight="balanced", random_state=42)),
    ])


def make_tfidf_svd_logreg_pipeline(n_components: int = 300):
    """Create the TF-IDF + TruncatedSVD + Logistic Regression model."""
    return Pipeline([
        ("vectorizer", build_tfidf_vectorizer(min_df=2, max_df=0.95, max_features=10000)),
        ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def make_logreg_classifier():
    """Create Logistic Regression for TF-IDF plus engineered numeric features."""
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def make_rf_classifier():
    """Create Random Forest for nonlinear comparison."""
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


def make_xgb_classifier():
    """Create XGBoost for regularized nonlinear comparison."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


# Backward-compatible names for older code.
def make_nb_pipeline():
    return make_nb_bow_pipeline()


def make_logreg_pipeline():
    return make_logreg_tfidf_pipeline()


def make_rf_pipeline():
    return Pipeline([
        ("vectorizer", build_tfidf_vectorizer(min_df=2, max_df=0.95, max_features=10000)),
        ("model", make_rf_classifier()),
    ])
