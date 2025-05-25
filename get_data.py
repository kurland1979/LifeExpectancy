import pandas as pd

def get_file(verbose=False):
    my_file = r'C:\Users\User\OneDrive\מסמכים\LifeExpectancy\Life Expectancy Data.csv'
    df = pd.read_csv(my_file)
    
    if verbose:
        print(df.info())
        print(df.describe())
    
    return df

def clean_data(df):
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/','_').str.lower()
    df.drop(columns='year',inplace=True)
    df.drop(columns='country',inplace=True)
    df['status'] = df['status'].map({'Developing':0, 'Developed':1})
    df = df[df['life_expectancy'].notna()]
    df['adult_mortality'] = df['adult_mortality'].fillna(df['adult_mortality'].median())
    df['alcohol'] = df['alcohol'].fillna(df['alcohol'].median())
    df.drop(columns='hepatitis_b',inplace=True)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df['polio'] = df['polio'].fillna(df['polio'].median())
    df.drop(columns='total_expenditure',inplace=True)
    df.drop(columns='diphtheria',inplace=True)
    df.drop(columns='gdp',inplace=True)
    df.drop(columns='population',inplace=True)
    df['thinness__1-19_years'] = df['thinness__1-19_years'].fillna(df['thinness__1-19_years'].median())
    df['thinness_5-9_years'] = df['thinness_5-9_years'].fillna(df['thinness_5-9_years'].median())
    df.drop(columns='income_composition_of_resources',inplace=True)
    df.drop(columns='schooling',inplace=True)
    
    return df
    

