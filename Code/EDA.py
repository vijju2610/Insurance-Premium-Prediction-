# Histogram for BMI to see the distribution
sns.histplot(data['bmi'], binwidth=1, kde=False, color="blue")
plt.title("Histogram of BMI")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()

# Histogram for Charges to see the distribution
sns.histplot(data['charges'], binwidth=100, kde=False, color="green")
plt.title("Histogram of Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()

# Histogram for age distribution
sns.histplot(data['age'], binwidth=2, kde=False, color="blue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Scatterplot of BMI vs Charges for smoker/non-smoker
sns.scatterplot(data=data, x='bmi', y='charges', hue='smoker')
plt.title("BMI vs. Charges by Smoker Status")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show()

# Charges histogram based on Coverage Level
sns.histplot(data=data, x='charges', hue='coverage_level', binwidth=100, element='step', palette='Set2')
plt.title("Charges Histogram Based on Coverage Level")
plt.xlabel("Charges")
plt.ylabel("Count")
plt.show()

# Average Charges Medical History Bar Chart
avg_charges_med_hist = data.groupby('medical_history')['charges'].mean().reset_index()
sns.barplot(data=avg_charges_med_hist, x='medical_history', y='charges', palette='Set2')
plt.title("Average Charges by Medical History")
plt.xticks(rotation=45)
plt.xlabel("Medical History")
plt.ylabel("Average Charges")
plt.show()


# Convert categorical data to numeric for correlation analysis
data['smoker_numeric'] = pd.Categorical(data['smoker']).codes
data['region_numeric'] = pd.Categorical(data['region']).codes
data['coverage_level_numeric'] = pd.Categorical(data['coverage_level']).codes

# Relationship between Region and charges
sns.boxplot(data=data, x='region', y='charges', palette='Set2')
plt.title("Charges by Region")
plt.xlabel("Region")
plt.ylabel("Charges")
plt.show()

# Calculate the correlation matrix
numeric_data = data[['charges', 'bmi', 'age', 'smoker_numeric', 'region_numeric', 'coverage_level_numeric']]
cor_matrix = numeric_data.corr()

# Plot the full correlation matrix without a mask
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', vmax=1.0, vmin=-1.0)
plt.title("Correlation Matrix of Numerical Variables")
plt.show()

# Relationship between smoker status and charges
sns.boxplot(data=data, x='smoker', y='charges', palette='Set2')
plt.title("Charges by Smoker Status")
plt.xlabel("Smoker")
plt.ylabel("Charges")
plt.show()
