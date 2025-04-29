# Display basic summary statistics and structure of the data
print(data.describe())
print(data.info())

# Check for missing values in the dataset
print(data.isna().sum().sum())

# Find categorical columns and print unique values
categorical_columns = data.select_dtypes(include=['object']).columns

# Print unique values in categorical columns
for col in categorical_columns:
    print(f"Unique values in {col}:")
    print(data[col].unique())

# Replace inconsistent values for 'smoker' with standardized ones
data['smoker'] = data['smoker'].str.lower()
data['smoker'] = data['smoker'].replace({'y': 'yes', 'n': 'no'})

# Replace incorrect spelling for 'Unemployed'
data['occupation'] = data['occupation'].str.replace("Uneployed", "Unemployed")

# Check unique values after replacement for verification
unique_smoker = data['smoker'].unique()
unique_occupation = data['occupation'].unique()

# Print the unique values for verification
print(unique_smoker)
print(unique_occupation)

# Replace inconsistent values for 'smoker' with standardized ones
data['smoker'] = data['smoker'].str.lower()
data['smoker'] = data['smoker'].replace({'y': 'yes', 'n': 'no'})

# Replace incorrect spelling for 'Unemployed'
data['occupation'] = data['occupation'].str.replace("Uneployed", "Unemployed")

# Check unique values after replacement for verification
unique_smoker = data['smoker'].unique()
unique_occupation = data['occupation'].unique()

# Print the unique values for verification
print(unique_smoker)
print(unique_occupation)

# Calculate IQR for BMI and Charges

data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
data['charges'] = pd.to_numeric(data['charges'], errors='coerce')

bmi_iqr = iqr(data['bmi'], nan_policy='omit')
charges_iqr = iqr(data['charges'], nan_policy='omit')

# Calculate the quantiles
bmi_quantiles = data['bmi'].quantile([0.25, 0.75])
charges_quantiles = data['charges'].quantile([0.25, 0.75])

# Define the cutoffs
bmi_cutoff = [bmi_quantiles.iloc[0] - 1.5 * bmi_iqr, bmi_quantiles.iloc[1] + 1.5 * bmi_iqr]
charges_cutoff = [charges_quantiles.iloc[0] - 1.5 * charges_iqr, charges_quantiles.iloc[1] + 1.5 * charges_iqr]

# Filter out the outliers
data = data[(data['bmi'] >= bmi_cutoff[0]) & (data['bmi'] <= bmi_cutoff[1])]
data = data[(data['charges'] >= charges_cutoff[0]) & (data['charges'] <= charges_cutoff[1])]
