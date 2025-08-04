# Data Preprocessing with Pandas & NumPy - Complete Tutorial
# Copy this entire code to Google Colab and run each cell

# ==========================================
# SECTION 1: SETUP AND IMPORTS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# ==========================================
# SECTION 2: CREATING SAMPLE DATASET
# ==========================================

# Create a realistic sample dataset
np.random.seed(42)

# Generate sample data
n_samples = 1000
names = [f"Person_{i}" for i in range(n_samples)]
ages = np.random.normal(35, 10, n_samples).astype(int)
ages = np.clip(ages, 18, 65)  # Clip ages between 18-65

cities = np.random.choice(['New York', 'London', 'Paris', 'Tokyo', 'Berlin', 'Sydney', None], 
                         n_samples, p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15, 0.05])

departments = np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Sales'], n_samples)

# Create salary based on age and department (with some noise)
base_salaries = {'IT': 70000, 'HR': 55000, 'Finance': 65000, 'Marketing': 60000, 'Sales': 58000}
salaries = []
for i in range(n_samples):
    dept = departments[i]
    base = base_salaries[dept]
    age_factor = (ages[i] - 25) * 1000  # $1000 per year of experience
    noise = np.random.normal(0, 5000)
    salary = base + age_factor + noise
    salaries.append(max(30000, salary))  # Minimum salary of $30k

# Introduce some missing values in salary
missing_indices = np.random.choice(n_samples, size=50, replace=False)
for idx in missing_indices:
    salaries[idx] = np.nan

# Create DataFrame
df = pd.DataFrame({
    'Name': names,
    'Age': ages,
    'City': cities,
    'Department': departments,
    'Salary': salaries
})

print("📊 Sample Dataset Created!")
print(f"Dataset shape: {df.shape}")
print("\nFirst 10 rows:")
print(df.head(10))
print("\nDataset Info:")
print(df.info())

# ==========================================
# SECTION 3: READING DATA (CSV/EXCEL SIMULATION)
# ==========================================
# Save to CSV and read it back (simulating real-world scenario)
df.to_csv('employee_data.csv', index=False)
print("\n Data saved to CSV file")

# Reading CSV with different parameters
df_csv = pd.read_csv('employee_data.csv')
print("Data read from CSV:")
print(f"Shape: {df_csv.shape}")
print(df_csv.head())
# Reading with specific parameters
df_selected = pd.read_csv('employee_data.csv', 
                         usecols=['Name', 'Age', 'Salary'],
                         nrows=100)  # Read only first 100 rows
print(f"\n Selected columns dataset shape: {df_selected.shape}")

# Reading with custom data types
df_typed = pd.read_csv('employee_data.csv', 
                      dtype={'Name': 'string', 'Age': 'int32'})
print("\n Data types after specification:")
print(df_typed.dtypes)

# ==========================================
# SECTION 4: HANDLING MISSING VALUES
# ==========================================

print("\n" + "="*50)
print("🔍 HANDLING MISSING VALUES")
print("="*50)

# Check missing values
print("Missing values per column:")
missing_info = df.isnull().sum()
print(missing_info)
print(f"\nTotal missing values: {df.isnull().sum().sum()}")
print(f"Percentage of missing data: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Method 1: Drop rows with missing values
df_dropped = df.dropna()
print(f"\n🗑️ After dropping rows with missing values:")
print(f"Original shape: {df.shape} → New shape: {df_dropped.shape}")
print(f"Rows removed: {df.shape[0] - df_dropped.shape[0]}")

# Method 2: Drop columns with missing values
df_drop_cols = df.dropna(axis=1)
print(f"\n🗑️ After dropping columns with missing values:")
print(f"Original shape: {df.shape} → New shape: {df_drop_cols.shape}")

# Method 3: Fill missing values with different strategies
df_filled = df.copy()

# Fill City with mode (most frequent value)
city_mode = df['City'].mode()[0]
df_filled['City'].fillna(city_mode, inplace=True)
print(f"\n🏙️ Filled missing cities with mode: '{city_mode}'")

# Fill Salary with mean
salary_mean = df['Salary'].mean()
df_filled['Salary'].fillna(salary_mean, inplace=True)
print(f"💰 Filled missing salaries with mean: ${salary_mean:.2f}")

# Method 4: Forward fill and backward fill
df_ffill = df.copy()
df_ffill['City'].fillna(method='ffill', inplace=True)  # Forward fill
df_ffill['City'].fillna(method='bfill', inplace=True)  # Backward fill for any remaining

print(f"\n⏭️ After forward/backward fill:")
print(f"Missing cities: {df_ffill['City'].isnull().sum()}")

# Method 5: Fill with interpolation (for numerical data)
df_interpolated = df.copy()
df_interpolated['Salary'] = df_interpolated['Salary'].interpolate()
print(f"\n📈 After interpolation:")
print(f"Missing salaries: {df_interpolated['Salary'].isnull().sum()}")

# Compare different strategies
print("\n📊 Comparison of missing value strategies:")
strategies = {
    'Original': df.isnull().sum().sum(),
    'Drop rows': df_dropped.isnull().sum().sum(),
    'Fill with mean/mode': df_filled.isnull().sum().sum(),
    'Forward/Backward fill': df_ffill.isnull().sum().sum(),
    'Interpolation': df_interpolated.isnull().sum().sum()
}

for strategy, missing_count in strategies.items():
    print(f"{strategy}: {missing_count} missing values")

# ==========================================
# SECTION 5: LABEL ENCODING & ONE-HOT ENCODING
# ==========================================

print("\n" + "="*50)
print("🔤 ENCODING CATEGORICAL VARIABLES")
print("="*50)

# Use the filled dataset for encoding
df_encode = df_filled.copy()

print("Original categorical data:")
print(df_encode[['City', 'Department']].head(10))
print(f"\nUnique cities: {df_encode['City'].unique()}")
print(f"Unique departments: {df_encode['Department'].unique()}")

# Method 1: Label Encoding
print("\n🏷️ LABEL ENCODING:")
le_city = LabelEncoder()
le_dept = LabelEncoder()

df_label = df_encode.copy()
df_label['City_encoded'] = le_city.fit_transform(df_label['City'])
df_label['Department_encoded'] = le_dept.fit_transform(df_label['Department'])

print("After Label Encoding:")
print(df_label[['City', 'City_encoded', 'Department', 'Department_encoded']].head(10))

# Show the mapping
print("\nCity encoding mapping:")
city_mapping = dict(zip(le_city.classes_, le_city.transform(le_city.classes_)))
for city, code in city_mapping.items():
    print(f"'{city}' → {code}")

print("\nDepartment encoding mapping:")
dept_mapping = dict(zip(le_dept.classes_, le_dept.transform(le_dept.classes_)))
for dept, code in dept_mapping.items():
    print(f"'{dept}' → {code}")

# Method 2: One-Hot Encoding
print("\n🎯 ONE-HOT ENCODING:")
df_onehot = pd.get_dummies(df_encode, columns=['City', 'Department'], prefix=['City', 'Dept'])
print(f"Original shape: {df_encode.shape}")
print(f"After One-Hot Encoding: {df_onehot.shape}")
print(f"New columns added: {df_onehot.shape[1] - df_encode.shape[1]}")

print("\nNew columns created:")
new_columns = [col for col in df_onehot.columns if col not in df_encode.columns]
print(new_columns)

print("\nSample of One-Hot encoded data:")
print(df_onehot[new_columns[:10]].head())

# Method 3: Manual encoding with mapping
print("\n🗺️ MANUAL ENCODING WITH MAPPING:")
city_map = {city: i for i, city in enumerate(df_encode['City'].unique())}
dept_map = {dept: i for i, dept in enumerate(df_encode['Department'].unique())}

df_manual = df_encode.copy()
df_manual['City_manual'] = df_manual['City'].map(city_map)
df_manual['Department_manual'] = df_manual['Department'].map(dept_map)

print("Manual encoding result:")
print(df_manual[['City', 'City_manual', 'Department', 'Department_manual']].head())

# ==========================================
# SECTION 6: FEATURE SCALING
# ==========================================

print("\n" + "="*50)
print("📏 FEATURE SCALING")
print("="*50)

# Prepare numerical data for scaling
numerical_features = ['Age', 'Salary']
df_scale = df_filled[numerical_features].copy()

print("Original numerical data statistics:")
print(df_scale.describe())

# Method 1: StandardScaler (Z-score normalization)
print("\n📊 STANDARD SCALER (Standardization):")
scaler_standard = StandardScaler()
df_standard = df_scale.copy()
df_standard[numerical_features] = scaler_standard.fit_transform(df_scale[numerical_features])

print("After StandardScaler:")
print(df_standard.describe())
print(f"Mean values: {df_standard.mean().values}")
print(f"Std values: {df_standard.std().values}")

# Method 2: MinMaxScaler (Min-Max normalization)
print("\n🎯 MIN-MAX SCALER (Normalization):")
scaler_minmax = MinMaxScaler()
df_minmax = df_scale.copy()
df_minmax[numerical_features] = scaler_minmax.fit_transform(df_scale[numerical_features])

print("After MinMaxScaler:")
print(df_minmax.describe())
print(f"Min values: {df_minmax.min().values}")
print(f"Max values: {df_minmax.max().values}")

# Method 3: Robust Scaler (using IQR)
from sklearn.preprocessing import RobustScaler
print("\n💪 ROBUST SCALER:")
scaler_robust = RobustScaler()
df_robust = df_scale.copy()
df_robust[numerical_features] = scaler_robust.fit_transform(df_scale[numerical_features])

print("After RobustScaler:")
print(df_robust.describe())

# Visualize the scaling effects
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Feature Scaling Comparison', fontsize=16)

# Original data
axes[0,0].hist(df_scale['Age'], bins=30, alpha=0.7, color='blue')
axes[0,0].hist(df_scale['Salary'], bins=30, alpha=0.7, color='red')
axes[0,0].set_title('Original Data')
axes[0,0].legend(['Age', 'Salary'])

# StandardScaler
axes[0,1].hist(df_standard['Age'], bins=30, alpha=0.7, color='blue')
axes[0,1].hist(df_standard['Salary'], bins=30, alpha=0.7, color='red')
axes[0,1].set_title('StandardScaler')
axes[0,1].legend(['Age', 'Salary'])

# MinMaxScaler
axes[1,0].hist(df_minmax['Age'], bins=30, alpha=0.7, color='blue')
axes[1,0].hist(df_minmax['Salary'], bins=30, alpha=0.7, color='red')
axes[1,0].set_title('MinMaxScaler')
axes[1,0].legend(['Age', 'Salary'])

# RobustScaler
axes[1,1].hist(df_robust['Age'], bins=30, alpha=0.7, color='blue')
axes[1,1].hist(df_robust['Salary'], bins=30, alpha=0.7, color='red')
axes[1,1].set_title('RobustScaler')
axes[1,1].legend(['Age', 'Salary'])

plt.tight_layout()
plt.show()

# ==========================================
# SECTION 7: DATAFRAMES VS SERIES VS NUMPY ARRAYS
# ==========================================

print("\n" + "="*50)
print("🔄 DATAFRAMES vs SERIES vs NUMPY ARRAYS")
print("="*50)

# Create examples of each data structure
sample_data = df_filled.head()

# DataFrame
df_example = sample_data[['Name', 'Age', 'Salary']]
print("📋 DATAFRAME:")
print(f"Type: {type(df_example)}")
print(f"Shape: {df_example.shape}")
print(f"Size: {df_example.size}")
print(f"Memory usage: {df_example.memory_usage().sum()} bytes")
print(df_example)

# Series
series_example = sample_data['Age']
print(f"\n📊 SERIES:")
print(f"Type: {type(series_example)}")
print(f"Shape: {series_example.shape}")
print(f"Size: {series_example.size}")
print(f"Memory usage: {series_example.memory_usage()} bytes")
print(series_example)

# NumPy Array
numpy_example = sample_data['Age'].values
print(f"\n🔢 NUMPY ARRAY:")
print(f"Type: {type(numpy_example)}")
print(f"Shape: {numpy_example.shape}")
print(f"Size: {numpy_example.size}")
print(f"Memory usage: {numpy_example.nbytes} bytes")
print(f"Data type: {numpy_example.dtype}")
print(numpy_example)

# Performance comparison
print("\n⚡ PERFORMANCE COMPARISON:")
import time

# Create larger dataset for performance testing
large_data = np.random.randn(100000)
large_df = pd.DataFrame({'values': large_data})
large_series = large_df['values']
large_array = large_data

# Test operation: calculate mean
iterations = 1000

# DataFrame
start_time = time.time()
for _ in range(iterations):
    result_df = large_df['values'].mean()
df_time = time.time() - start_time

# Series
start_time = time.time()
for _ in range(iterations):
    result_series = large_series.mean()
series_time = time.time() - start_time

# NumPy Array
start_time = time.time()
for _ in range(iterations):
    result_array = np.mean(large_array)
array_time = time.time() - start_time

print(f"DataFrame mean calculation: {df_time:.4f} seconds")
print(f"Series mean calculation: {series_time:.4f} seconds")
print(f"NumPy array mean calculation: {array_time:.4f} seconds")

# Memory usage comparison
print(f"\nMemory usage comparison (100k elements):")
print(f"DataFrame: {large_df.memory_usage().sum()} bytes")
print(f"Series: {large_series.memory_usage()} bytes")
print(f"NumPy Array: {large_array.nbytes} bytes")

# When to use what?
print("\n🎯 WHEN TO USE WHAT:")
comparison_data = {
    'Data Structure': ['DataFrame', 'Series', 'NumPy Array'],
    'Best for': [
        'Mixed data types, complex operations, data analysis',
        'Single column operations, time series',
        'Numerical computations, mathematical operations'
    ],
    'Advantages': [
        'Labels, mixed types, rich functionality',
        'Pandas integration, indexing',
        'Speed, memory efficiency, vectorization'
    ],
    'Disadvantages': [
        'Memory overhead, slower for pure numerical',
        'Single dimension only',
        'Homogeneous data only, less functionality'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# ==========================================
# SECTION 8: COMPLETE PREPROCESSING PIPELINE
# ==========================================

print("\n" + "="*50)
print("🔧 COMPLETE PREPROCESSING PIPELINE")
print("="*50)

def preprocess_data(df):
    """Complete preprocessing pipeline"""
    print("Starting preprocessing pipeline...")
    
    # Step 1: Handle missing values
    df_processed = df.copy()
    
    # Fill categorical missing values with mode
    for col in df_processed.select_dtypes(include=['object']).columns:
        if df_processed[col].isnull().any():
            mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
            df_processed[col].fillna(mode_value, inplace=True)
            print(f"✅ Filled missing {col} with mode: {mode_value}")
    
    # Fill numerical missing values with median
    for col in df_processed.select_dtypes(include=[np.number]).columns:
        if df_processed[col].isnull().any():
            median_value = df_processed[col].median()
            df_processed[col].fillna(median_value, inplace=True)
            print(f"✅ Filled missing {col} with median: {median_value:.2f}")
    
    # Step 2: Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'Name']  # Exclude Name column
    
    if len(categorical_cols) > 0:
        df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, prefix=categorical_cols)
        print(f"✅ One-hot encoded columns: {list(categorical_cols)}")
    else:
        df_encoded = df_processed
    
    # Step 3: Scale numerical features
    numerical_cols = df_encoded.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if not col.startswith(tuple(categorical_cols))]
    
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
        print(f"✅ Standardized columns: {list(numerical_cols)}")
    
    print(f"✅ Preprocessing complete! Shape: {df.shape} → {df_encoded.shape}")
    
    return df_encoded, scaler

# Apply the complete pipeline
df_final, scaler_used = preprocess_data(df)

print(f"\nFinal preprocessed dataset:")
print(f"Shape: {df_final.shape}")
print(f"Columns: {len(df_final.columns)}")
print(f"Missing values: {df_final.isnull().sum().sum()}")

print("\nFirst 5 rows of preprocessed data:")
print(df_final.head())

print("\nSample statistics:")
print(df_final.describe())

# ==========================================
# SECTION 9: PRACTICAL EXAMPLE WITH MACHINE LEARNING
# ==========================================

print("\n" + "="*50)
print("🤖 PRACTICAL EXAMPLE: MACHINE LEARNING READY DATA")
print("="*50)

# Prepare data for machine learning
# Let's predict salary based on other features
ml_data = df_final.copy()

# Separate features and target
feature_cols = [col for col in ml_data.columns if col not in ['Name', 'Salary']]
X = ml_data[feature_cols]
y = ml_data['Salary']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {list(X.columns)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train a simple model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_
})
feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

print(f"\nFeature Importance (Top 10):")
print(feature_importance.head(10))

print("\n🎉 Data preprocessing tutorial completed successfully!")
print("📚 You now know how to:")
print("   • Read data from various sources")
print("   • Handle missing values effectively")  
print("   • Encode categorical variables")
print("   • Scale numerical features")
print("   • Understand different data structures")
print("   • Build complete preprocessing pipelines")
print("   • Prepare data for machine learning")
