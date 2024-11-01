df = pd.read_csv('Data.csv', na_values=['/'])

# Split the voltage window into two columns
df[['Lower Voltage Limit', 'Upper Voltage Limit']] = df['voltage window'].str.split('-', expand=True)

# Calculate the total sintering time
df['Average Sintering Time'] = (df['Sintering Time (h) 1st step'] + df['Sintering Time (h) 2nd step'] + df['Sintering Time (h) 3rd step']) /3

# Calculate the total sintering time
df['Total Sintering Time'] = df['Sintering Time (h) 1st step'] + df['Sintering Time (h) 2nd step'] + df['Sintering Time (h) 3rd step']

# Calculate the average sintering temperature
df['Average Sintering Temperature'] = (df['Sintering Temperature (°C) 1st step'] + df['Sintering Temperature (°C) 2nd step'] + df['Sintering Temperature (°C) 3rd step']) / 3

# Calculate the total sintering temperature
df['Total Sintering Temperature'] = df['Sintering Temperature (°C) 1st step'] + df['Sintering Temperature (°C) 2nd step'] + df['Sintering Temperature (°C) 3rd step']

# Calculate the weighted average sintering temperature
df['Weighted Average Sintering Temperature'] = ((df['Sintering Temperature (°C) 1st step'] * df['Sintering Time (h) 1st step']) + 
                                              (df['Sintering Temperature (°C) 2nd step'] * df['Sintering Time (h) 2nd step']) + 
                                              (df['Sintering Temperature (°C) 3rd step'] * df['Sintering Time (h) 3rd step'])) / df['Total Sintering Time']

# Calculate the sintering temperature range
df['Sintering Temperature Range'] = df['Sintering Temperature (°C) 3rd step'] - df['Sintering Temperature (°C) 1st step']

# Calculate the time-temperature product
df['Time-Temperature Product'] = df['Total Sintering Time'] * df['Average Sintering Temperature']

# Calculate the voltage window range
df['Voltage Window Range'] = df['Upper Voltage Limit'].astype(float) - df['Lower Voltage Limit'].astype(float)

df = df.dropna(subset=['Capacity Retention (%)'])

df['Capacity Retention (mAh/g)'] = (df['Capacity Retention (%)']/100)*df['Discharge Capacity [mAh/g]']

def data_prep():

    # Define columns to be normalized and one-hot encoded
    numerical_cols = ['Li', 'Ni', 'Mn', 'O', 'Sintering Temperature (°C) 1st step', 'Sintering Temperature (°C) 2nd step',
                  'Sintering Temperature (°C) 3rd step', 'Sintering Time (h) 1st step', 'Sintering Time (h) 2nd step',
                  'Sintering Time (h) 3rd step', 'Lattice Constant', 'Current Density/C rate',
                   'Capacity Retention (Cycles)', 'Weighted Average Sintering Temperature', 'Total Sintering Temperature', 'Average Sintering Temperature', 'Total Sintering Time', 'Average Sintering Time', 'Lower Voltage Limit', 'Upper Voltage Limit', 'Sintering Temperature Range', 'Time-Temperature Product', 'Voltage Window Range']
    categorical_cols = ['Cathode Material Composition', 'Synthesis Method', 'Li Source', 'Ni Source', 'Mn Source',
                    'Mixing method', 'Furnace', 'Gas atmosphere', 'Heating Rate', 'Cooling Rate',
                    'Space group', 'Electrolyte salt', 'Electrolyte solvent', 'voltage window', 'Surface Morphology']

    # Pipeline for numerical features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ])

    # Pipeline for categorical features
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False))
    ])

    # Combine the pipelines into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])

    # Separate input features
    X =  df.drop(columns=['Capacity Retention (%)', 'Discharge Capacity [mAh/g]', 'Rate Capability (mAh/g)', 'Rate Capability (C rate)', 'Coulombic efficiency%', 'Capacity Retention (mAh/g)'])

    # Create target variable
    y = df['Capacity Retention (mAh/g)']
  
    # Store original column names
    original_feature_names = X.columns.tolist()

    # Fit the preprocessor
    X_preprocessed = preprocessor.fit_transform(X)

    # Get feature names after preprocessing
    numeric_feature_names = numerical_cols
    categorical_feature_names =  preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    preprocessed_feature_names = np.concatenate([numeric_feature_names, categorical_feature_names])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y.values, test_size=0.15, random_state=42, shuffle=True)

    return  df, X_preprocessed, y, X_train, X_test, y_train, y_test, preprocessed_feature_names, categorical_feature_names, categorical_cols

# Call the function
df, X, y, X_train, X_test, y_train, y_test, preprocessed_feature_names, categorical_features, categorical_cols = data_prep()
