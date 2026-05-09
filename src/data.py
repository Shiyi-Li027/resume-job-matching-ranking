import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: str) -> pd.DataFrame:
    """Load the Resume_Screening_Dataset from a local CSV file."""
    
    """
    This function loads the dataset from a specified CSV file path. 
    It reads the CSV file into a pandas DataFrame, ensuring that 
    the 'Resume', 'Job_Description', and 'Decision' columns are 
    treated as strings to prevent any unintended type inference 
    issues. The function returns the loaded DataFrame for further 
    processing in the data pipeline.
    
    Args:
        path (str): The file path to the dataset CSV file.
        
    Returns: A pd.DataFrame containing the loaded dataset.
    """
    return pd.read_csv(path, dtype={"Resume": str, "Job_Description": str, "Decision": str})

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates and handle missing values."""
    
    """
    This function performs basic data cleaning on the input DataFrame. It includes:
    - Lowercasing column names for consistency.
    - Replacing empty strings in the 'resume', 'job_description', and 'decision' columns with NaN to properly identify missing values.
    - Dropping rows where either the resume-job description pair or the decision is missing, as these rows cannot be used for modeling.
    - Optionally filling any remaining missing values in the resume and job description columns with empty strings, allowing the model to still learn from the other text if one of them is missing.
    - Dropping duplicate rows with the same resume, job description, and decision to ensure data quality.
    - Printing the shape of the dataset before and after cleaning, as well as the number of missing values in the relevant columns before and after dropping. 
    
    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.
        
    Returns: A cleaned pd.DataFrame ready for further processing.
    """
    
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    # print the number of rows and columns before cleaning
    print(f"Dataset shape before cleaning: {df.shape}")
    
    # replace empty strings with NaN to properly identify missing values, as empty strings in resume, job description, or decision would not be useful for matching and should be treated as missing.
    df[["resume", "job_description", "decision"]] = df[["resume", "job_description", "decision"]].replace("", pd.NA)
    
    # print the number of missing values in resume, job description, and decision columns before dropping
    print("Missing values before dropping:")
    print(df[["resume", "job_description", "decision"]].isnull().sum())
    
    # drop rows where both resume-job description pair or decision is missing, as we can't match resumes with job descriptions if one of them is missing. And we can't train a model without the decision label.
    df.dropna(subset=["resume", "job_description", "decision"], inplace=True, ignore_index=True)
    # print the number of missing values in resume, job description, and decision columns after dropping
    print("Missing values after dropping:")
    print(df[["resume", "job_description", "decision"]].isnull().sum())
    
    
    # fill any remaining missing values in resume and job description with empty strings, as the model can still learn from the other text if one of them is missing. This also ensures that we don't have any NaN values when we create the combined text column.
    # df["resume"] = df["resume"].fillna("")
    # df["job_description"] = df["job_description"].fillna("")

    
    # drop duplicates rows with same resume, job description, and decision and reset index
    df.drop_duplicates(subset=["resume", "job_description", "decision"], inplace=True, ignore_index=True)
    
    # print the number of rows and columns after data cleaning
    print(f"Dataset shape after cleaning: {df.shape}")
    
    return df

def make_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create new text columns for modeling by combining resume and job description."""
    
    """
    This function creates new text columns in the DataFrame for modeling. It generates 
    a 'combined_text' column that concatenates the resume and job description with a 
    separator. This allows us to use both the resume and job description information 
    together when training our models, which can help capture the relationship between 
    the two texts for better matching. The function also handles any missing values 
    in the resume and job description by filling them with empty strings before 
    concatenation to ensure that we don't have NaN values in the combined text.
    
    Args:
        df (pd.DataFrame): The input DataFrame with 'resume' and 'job_description' columns. 
        
    Returns: A pd.DataFrame with new 'resume_text' and 'combined_text' columns added.
    """
    
    df = df.copy()
    df["combined_text"] = (
        df["resume"].fillna("") + " \n[JOB_DESCRIPTION] " + df["job_description"].fillna("")
    )
    return df

def convert_decision_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 'decision' column to binary labels (1 for select, 0 for reject)."""
    
    """
    This function converts the 'decision' column in the DataFrame to binary labels, 
    where "select" is mapped to 1 and "reject" is mapped to 0. This is useful for 
    training classification models that require numerical labels. The function also 
    standardizes the text in the 'decision' column by stripping whitespace and converting 
    it to lowercase before mapping to ensure consistent labeling.
    
    Args:
        df (pd.DataFrame): The input DataFrame with a 'decision' column containing categorical labels ("select" and "reject").
        
    Returns: A pd.DataFrame with the 'decision' column converted to binary labels (1 for select, 0 for reject).
    """
    
    df = df.copy()
    df["decision"] = df["decision"].str.strip().str.lower()  # standardize text
    df["decision"] = df["decision"].map({"select": 1, "reject": 0})  # map to binary labels
    
    return df

def convert_decision_back_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 'decision' column back to categorical labels (1 for select, 0 for reject)."""
    
    """
    This function is useful for converting the binary labels back to their original
    categorical labels (1 for select, 0 for reject) for interpretation or reporting purposes.
    
    Args:
        df (pd.DataFrame): The input DataFrame with a 'decision' column containing binary labels (1 for select, 0 for reject).
        
    Returns: A pd.DataFrame with the 'decision' column converted back to categorical labels ("select" for 1 and "reject" for 0).
    """
    
    df = df.copy()
    df["decision"] = df["decision"].map({1: "select", 0: "reject"})  # map to categorical labels
    
    return df

def split_data(df: pd.DataFrame, label_col: str = "decision", test_size: float = 0.2, random_state: int = 42):
    """
    This function splits the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        label_col (str, optional): The column name for the labels. Defaults to "decision".
        test_size (float, optional): The proportion of the dataset to include in the test set. Defaults to 0.2.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.

    Returns: A tuple containing the training and testing sets (X_train, X_test, y_train, y_test).
    """
    
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col]                                                      
    )
    
def preprocess_data(path: str) -> pd.DataFrame:
    """
    This function orchestrates the entire data preprocessing pipeline, including:
    - loading the dataset from a specified path
    - performing basic cleaning (such as handling missing values and duplicates)
    - creating new combined resume-job description text columns for modeling
    - converting the decision labels to binary format.
    
    Args:
        path (str): The file path to the dataset CSV file.
    
    Returns: A pd.DataFrame ready for use in model training and evaluation.
    """
    
    df = load_dataset(path)
    df = clean_dataset(df)
    df = make_text_columns(df)
    df = convert_decision_to_binary(df)
    
    return df
