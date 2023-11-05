import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# df = pd.read_csv("C:\\Users\\ayael\\OneDrive\\Desktop\\FALL2023\\big data\\Assignment_1_big data\\bd-a1\\bd-a1\\diabetes.csv")

def data_cleaning(df):
    # Drop missing values
    df = df.dropna()
    # Drop duplicates
    df = df.drop_duplicates()
    # Replace missing values in a specific column with the median
    df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace=True)
    df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
    df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
    # remove outliers
    columns_to_check = ['BMI', 'Insulin', 'BloodPressure']
    for column in columns_to_check:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]
    return df


def data_transformation(df):
    scaler = StandardScaler()
    columns_to_scale = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
                        'Age']

    # Check for zero variance
    for column in columns_to_scale:
        if df[column].std() == 0:
            df.drop(column, axis=1, inplace=True)
            columns_to_scale.remove(column)

    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale].values)

    # Only apply log transformation to positive values
    for column in ['Insulin', 'DiabetesPedigreeFunction']:
        df[column + '_log'] = df[column].apply(lambda x: np.log1p(x) if x > 0 else x)

    age_bins = [20, 30, 40, 50, 60, 70, 80]  # Adjust bins according to your needs
    age_labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

    return df


def data_reduction(df, n_components=0.95):
    # Check and handle NaN values before PCA
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_imputed = imputer.fit_transform(df)
    # Scaling the features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_imputed)
    # Applying PCA
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(df_scaled)
    # Create a DataFrame with the principal components
    principalDf = pd.DataFrame(data=principalComponents)
    return principalDf, pca


def data_discretization(df):
    # Age bins
    age_bins = [0, 30, 40, 50, 60, 70, 120]
    age_labels = ['0-29', '30-39', '40-49', '50-59', '60-69', '70+']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

    # BMI bins according to WHO standards
    bmi_bins = [0, 18.5, 25, 30, 35, 40, np.inf]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese Class I', 'Obese Class II', 'Obese Class III']
    df['BMI_Category'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_labels, right=False)

    return df


# Make sure to replace the file_path with the correct path where your file is located
# df = pd.read_csv("C:\\Users\\ayael\\OneDrive\\Desktop\\FALL2023\\big data\\Assignment_1_big data\\bd-a1\\bd-a1\\diabetes.csv")

# ... rest of your functions

if __name__ == "__main__":
    file_path = r"C:\Users\ayael\OneDrive\Desktop\FALL2023\big data\Assignment_1_big data\bd-a1\diabetes.csv"
    df = pd.read_csv(file_path)

    df = data_cleaning(df)

    # Preserve 'Age' and 'BMI'
    age_bmi_df = df[['Age', 'BMI']].copy()

    df = data_transformation(df)

    # Remove 'Age' and 'BMI' before PCA
    df_without_age_bmi = df.drop(columns=['Age', 'BMI'])

    principalDf, pca_model = data_reduction(df_without_age_bmi)  # PCA without 'Age' and 'BMI'

    # Combine 'Age' and 'BMI' back after PCA
    principalDf = pd.concat([principalDf, age_bmi_df.reset_index(drop=True)], axis=1)

    # You should define the data_discretization function here as it's not provided in your code snippet

    df = data_discretization(principalDf)

    # Save the dataset
    df.to_csv('res_dpre.csv', index=False)
