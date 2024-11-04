# Call the data
df, X, y, X_train, X_test, y_train, y_test, preprocessed_feature_names, categorical_features, categorical_cols = data_prep()

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Call individual models
linear = LR
lasso = Lasso
ridge = Ridge
dt= DT
rf = RF
xgb = XGB
mlp = NN

# Create the ensemble model
ensemble = VotingRegressor([
    ('linear', linear),
    ('lasso', lasso),
    ('ridge', ridge),
    ('dt', dt),
    ('rf', rf),
    ('xgb', xgb),
    ('mlp', mlp)
])

# Train the ensemble model
ensemble.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = ensemble.predict(X_train_scaled)
y_test_pred = ensemble.predict(X_test_scaled)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# Plot actual vs predicted values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(y_train, y_train_pred)
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Values')
ax1.set_ylabel('Predicted Values')
ax1.set_title(f'Actual vs Predicted Values (Train Set)\nRMSE: {train_rmse:.4f}, R²: {train_r2:.4f}')

ax2.scatter(y_test, y_test_pred)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Values')
ax2.set_ylabel('Predicted Values')
ax2.set_title(f'Actual vs Predicted Values (Test Set)\nRMSE: {test_rmse:.4f}, R²: {test_r2:.4f}')

plt.tight_layout()
plt.savefig(f'actual_vs_predicted-.png', bbox_inches='tight')
plt.show()
