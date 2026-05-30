import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TECHNICAL_SKILLS = [
    "python", "sql", "java", "javascript", "r",
    "aws", "azure", "docker", "kubernetes",
    "tensorflow", "pytorch", "scikit-learn",
    "deep", "learning", "nlp","spark", "vba",
    "hadoop", "tableau","excel", "react", 
    "node", "git", "linux", "machine",
    "nodejs", "powerbi", "cnn", "lstm", 
    "api", "rest", "graphql", "ci/cd", "jenkins",
    "airflow", "kafka", "redis", "mongodb",
    "postgresql", "mysql", "oracle", "snowflake",
    "sas", "matlab", "scala", "go", "ruby",
    "c++", "c#", "flutter", "dart", "swift",
    "android", "ios", "angular", "vue", "django",
    "flask", "fastapi", "spring", "hibernate",
]


def build_bow_vectorizer(
    min_df: int = 1,
    max_df: float = 1.0,
    max_features: int = 10000,
) -> CountVectorizer:
    """
    Build a CountVectorizer for BoW feature extraction with specified parameters.
    """
    return CountVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]*\b",
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )


def build_tfidf_vectorizer(
    min_df: int = 1,
    max_df: float = 1.0,
    max_features: int = 10000,
) -> TfidfVectorizer:
    """
    Build a TfidfVectorizer for TF-IDF feature extraction with specified parameters.
    """
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]*\b",
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text for overlap ratio analysis.

    This is used for feature engineering rather than for BoW/TF-IDF modeling,
    because sklearn vectorizers handle their own tokenization.
    """
    if pd.isna(text):
        return []

    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", str(text).lower())
    return [token for token in tokens if token not in ENGLISH_STOP_WORDS]


def add_token_columns(
    df: pd.DataFrame,
    resume_col: str = "resume",
    job_col: str = "job_description",
) -> pd.DataFrame:
    """Create tokenized resume and job description columns for overlap features."""
    df = df.copy()
    df["resume_tokens"] = df[resume_col].apply(tokenize_text)
    df["job_description_tokens"] = df[job_col].apply(tokenize_text)
    return df


def _to_token_set(tokens) -> set[str]:
    """Convert a token list-like value to a set of string tokens."""
    if tokens is None:
        return set()
    if isinstance(tokens, float) and pd.isna(tokens):
        return set()
    if isinstance(tokens, str):
        return set(tokens.split())
    return set(tokens)


def calculate_overlap_ratio(
    df: pd.DataFrame,
    resume_tokens_col: str = "resume_tokens",
    job_tokens_col: str = "job_description_tokens",
) -> pd.DataFrame:
    """
    Calculate the resume-job overlap ratio for each resume-job pair.

    The overlap ratio is defined as:
        |resume_tokens ∩ job_description_tokens| / |job_description_tokens|

    Repeated tokens are counted only once because the calculation uses sets.
    """
    df = df.copy()

    resume_sets = df[resume_tokens_col].apply(_to_token_set)
    job_sets = df[job_tokens_col].apply(_to_token_set)

    # Calculate the number of overlapping unique tokens between resume and JD.
    df["resume_jd_overlap_set_len"] = [
        len(resume_set.intersection(job_set))
        for resume_set, job_set in zip(resume_sets, job_sets)
    ]

    # Calculate the number of unique JD tokens as the denominator.
    df["jd_tokens_set_len"] = job_sets.apply(len)

    # Calculate overlap ratio and avoid division by zero.
    df["resume_jd_overlap_ratio"] = [
        0.0 if jd_len == 0 else overlap_len / jd_len
        for overlap_len, jd_len in zip(
            df["resume_jd_overlap_set_len"],
            df["jd_tokens_set_len"],
        )
    ]

    return df


def calculate_pairwise_cosine_similarity(resume_matrix, job_matrix) -> np.ndarray:
    """
    Calculate cosine similarity for corresponding resume-job vector pairs.

    TfidfVectorizer normalizes rows by default, so row-wise dot product equals
    cosine similarity for TF-IDF vectors.
    """
    if sparse.issparse(resume_matrix) or sparse.issparse(job_matrix):
        return resume_matrix.multiply(job_matrix).sum(axis=1).A1

    return np.sum(resume_matrix * job_matrix, axis=1)


def add_skill_match_features(
    df: pd.DataFrame,
    resume_col: str = "resume",
    job_col: str = "job_description",
    skills: list[str] | None = None,
) -> pd.DataFrame:
    """
    Create technical skill matching features.

    Each skill feature equals 1 when the skill appears in both the resume and
    the job description for the same instance.
    """
    df = df.copy()
    skills = skills or TECHNICAL_SKILLS

    resume_text = df[resume_col].fillna("").str.lower()
    job_text = df[job_col].fillna("").str.lower()
    skill_feature_cols = []

    for skill in skills:
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", skill).strip("_")
        col_name = f"skill_match_{safe_name}"
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"

        resume_has_skill = resume_text.str.contains(pattern, regex=True, na=False)
        job_has_skill = job_text.str.contains(pattern, regex=True, na=False)
        df[col_name] = (resume_has_skill & job_has_skill).astype(int)
        skill_feature_cols.append(col_name)

    df["skill_match_score"] = df[skill_feature_cols].sum(axis=1)
    df["skill_match_ratio"] = df["skill_match_score"] / len(skill_feature_cols)

    return df


def build_engineered_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered matching features used by the main models.
    """
    df = add_token_columns(df)
    df = calculate_overlap_ratio(df)
    df = add_skill_match_features(df)
    return df
