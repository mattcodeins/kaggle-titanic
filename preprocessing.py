import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

def preprocess_df(df):
    df = remove_nan(df)
    df = add_features(df)
    df = categories_to_idx(df)
    return df

def add_features(df):
    df = add_title(df)
    df = add_family_size(df)
    df = add_floor(df)
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    return df

def categories_to_idx(df):
    df['Sex'] = df['Sex'].map({'male':0, "female":1})
    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})
    if 'Title' in df.columns:
        df['Title'] = df['Title'].map({'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Other': 4})
    if 'Floor' in df.columns:
        df['Floor'] = df['Floor'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'nan': 8})
    return df

def create_xy(df):
    X, y = df.drop('Survived', axis=1), df['Survived']
    return X, y

def add_title(df):
    """ Adds name title category, limiting to 4 most common. """
    titles = df['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip(' '))
    df['Title'] = titles.apply(lambda x: x if x in ['Mr', 'Mrs', 'Miss', 'Master'] else 'Other')
    return df

def add_family_size(df):
    """ Adds family size category, which is the number of matching surnames. """
    surnames = df['Name'].apply(lambda x: x.split(",")[0])
    df['Family Size'] = surnames.apply(lambda a: Counter(surnames)[a]-1)
    return df

def add_floor(df):
    df['Floor'] = df['Cabin'].astype(str).apply(lambda x: x[0] if x != "nan" else "nan")
    return df

def remove_nan(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df

if __name__ == "__main__":
    """
    Preprocess data deterministically.
    """
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    train = preprocess_df(train)
    test = preprocess_df(test)
    train.to_csv('data/train_processed.csv', index=False)
    test.to_csv('data/test_processed.csv', index=False)
