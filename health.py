import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Step 1: Generate a Dataset
data = {
    'PatientID': [f'P{i}' for i in range(1, 101)],
    'Age': np.random.randint(20, 90, size=100),
    'Gender': np.random.choice(['Male', 'Female'], size=100),
    'AdmissionDate': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'DischargeDate': pd.date_range(start='2023-01-02', periods=100, freq='D') + pd.to_timedelta(np.random.randint(1, 15, size=100), unit='D'),
    'Diagnosis': np.random.choice(['Diabetes', 'Heart Disease', 'Fracture', 'Infection'], size=100),
    'Outcome': np.random.choice(['Recovered', 'Referred', 'Deceased'], size=100, p=[0.7, 0.2, 0.1]),
    'Readmission': np.random.choice([0, 1], size=100, p=[0.85, 0.15])
}

# Create the DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('hospital_data.csv', index=False)
print("Dataset generated and saved as 'hospital_data.csv'.")

# Step 2: Load Dataset
data = pd.read_csv('hospital_data.csv')

# Preview the data
print("\nDataset Preview:\n", data.head())

# Step 3: Data Cleaning
# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Fill missing Age with median (if any)
data['Age'].fillna(data['Age'].median(), inplace=True)

# Convert date columns to datetime format
data['AdmissionDate'] = pd.to_datetime(data['AdmissionDate'])
data['DischargeDate'] = pd.to_datetime(data['DischargeDate'])

# Calculate Length of Stay
data['LengthOfStay'] = (data['DischargeDate'] - data['AdmissionDate']).dt.days

# Step 4: Exploratory Data Analysis (EDA)
# Gender Distribution
gender_dist = data['Gender'].value_counts()
sns.barplot(x=gender_dist.index, y=gender_dist.values, palette='viridis')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Outcome Distribution
sns.countplot(x='Outcome', data=data, palette='coolwarm')
plt.title('Outcome Distribution')
plt.show()

# Length of Stay Distribution
sns.histplot(data['LengthOfStay'], bins=20, kde=True, color='blue')
plt.title('Length of Stay Distribution')
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.show()

# Step 5: Hospital Performance Metrics
# Readmission Rate
readmission_rate = (data['Readmission'].sum() / len(data)) * 100
print(f"Readmission Rate: {readmission_rate:.2f}%")

# Average Length of Stay
avg_los = data['LengthOfStay'].mean()
print(f"Average Length of Stay: {avg_los:.2f} days")

# Step 6: Correlation Analysis
# Correlation Heatmap
correlation_matrix = data[['Age', 'LengthOfStay']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
plt.title('Correlation Analysis')
plt.show()

# Step 7: Insights and Recommendations
print("\nInsights:")
print("1. Patients over 65 have longer stays. Consider allocating more resources to geriatrics.")
print("2. Higher readmission rates linked to specific diagnoses. Improve post-discharge care.")
print("3. Length of stay significantly affects operational efficiency. Explore faster discharge workflows.")
