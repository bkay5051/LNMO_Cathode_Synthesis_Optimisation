# Define RFECV function
def perform_rfecv(X, y):
    base_model = Ridge(random_state=42)
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    rfecv = RFECV(estimator=base_model, step=1, cv=cv, n_jobs=-1, min_features_to_select=5)
    rfecv.fit(X, y)
    return rfecv

# Perform RFECV once to save resources and time
rfecv = perform_rfecv(X_train, y_train)
X_train_selected = rfecv.transform(X_train)

# Define select from model objective
def objective_sfm(trial, X, y, rfecv):
    n_features_to_select = trial.suggest_int('n_features_to_select', 5, X.shape[1])
    params = {
        'alpha': trial.suggest_float('alpha', 0.05, 10, log=True),
        'max_iter': 10000,
    }

    model = Ridge(**params, random_state=42)
    selector = SelectFromModel(model, max_features=n_features_to_select, threshold=-np.inf)
    
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = []
    for step, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_val_selected)
        score = mean_squared_error(y_val, y_pred)
        scores.append(score)
        
        trial.report(np.mean(scores), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)

# Define RFECV objective
def objective_rfe(trial, X, y, rfecv):
    params = {
        'alpha': trial.suggest_float('alpha', 0.05, 10, log=True),
        'max_iter': 10000,
    }

    model = Ridge(**params, random_state=42)
    
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = []
    for step, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        X_train_selected = rfecv.transform(X_train)
        X_val_selected = rfecv.transform(X_val)
        
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_val_selected)
        score = mean_squared_error(y_val, y_pred)  
        scores.append(score)
        
        trial.report(np.mean(scores), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)

# Define full feature model objective
def objective_full(trial, X, y, rfecv):
    params = {
        'alpha': trial.suggest_float('alpha', 0.05, 10, log=True),
        'max_iter': 10000,
    }

    model = Ridge(**params, random_state=42)
    
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = []
    for step, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = mean_squared_error(y_val, y_pred)
        scores.append(score)
        
        trial.report(np.mean(scores), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)

# Define Optuna initialisation
def optimize_model(objective, X_train, y_train, rfecv):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=2)
    study = optuna.create_study(direction='minimize', pruner=pruner, sampler=optuna.samplers.TPESampler(seed=42))
    objective_with_data = partial(objective, X=X_train, y=y_train, rfecv=rfecv)
    try:
        study.optimize(objective_with_data, n_trials=100, n_jobs=-1)
    except Exception as e:
        logging.error(f"An error occurred during optimization: {e}")
        return None
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.plot_optimization_history(study)
    plt.title("Optimization History")
    plt.tight_layout()
    plt.savefig(f'optimization_history_{objective.__name__}.png', bbox_inches='tight')
    plt.close()

    # Plot parameter importances
    plt.figure(figsize=(10, 6))
    optuna.visualization.plot_param_importances(study)
    plt.title("Parameter Importances")
    plt.tight_layout()
    plt.savefig(f'param_importances_{objective.__name__}.png', bbox_inches='tight')
    plt.close()

    return study

#  Define training of the final model
def train_model(best_params, X_train, y_train, selector_type):
    model = Ridge(alpha=best_params['alpha'], random_state=42)
    
    if selector_type == 'sfm':
        selector = SelectFromModel(model, max_features=best_params['n_features_to_select'], threshold=-np.inf)
        X_train_selected = selector.fit_transform(X_train, y_train)
        model.fit(X_train_selected, y_train)
        return model, selector
    elif selector_type == 'rfe':
        selector = RFECV(estimator=model, step=1, min_features_to_select=5)
        X_train_selected = selector.fit_transform(X_train, y_train)
        model.fit(X_train_selected, y_train)
        return model, selector
    elif selector_type == 'full':
        model.fit(X_train, y_train)
        return model, None
    else:
        raise ValueError("Invalid selector_type. Must be 'sfm', 'rfe', or 'full'.")

# Define evaluation of the final model
def evaluate_model(model, selector, X_train, y_train, X_test, y_test, preprocessed_feature_names, model_name):
    if selector is not None:
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
    else:
        X_train_selected = X_train
        X_test_selected = X_test
    
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    
    logging.info(f"{model_name} Performance:")
    logging.info(f"Train RMSE: {train_rmse:.4f}")
    logging.info(f"Train MSE: {train_mse:.4f}")
    logging.info(f"Train R²: {train_r2:.4f}")
    logging.info(f"Test RMSE: {test_rmse:.4f}")
    logging.info(f"Test MSE: {test_mse:.4f}")
    logging.info(f"Test R²: {test_r2:.4f}")

    importances = np.abs(model.coef_)
    if selector is not None:
        selected_indices = selector.get_support(indices=True)
        selected_preprocessed_features = [preprocessed_feature_names[i] for i in selected_indices]
    else:
        selected_preprocessed_features = preprocessed_feature_names
    
    importance_df = pd.DataFrame({
        'Feature': selected_preprocessed_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    logging.info(f"\nTop 10 most important features ({model_name}):")
    logging.info(importance_df.head(10))

    # Plot actual vs predicted values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.scatter(y_train, y_train_pred)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'Actual vs Predicted IC Values (Train Set)\nRMSE: {train_rmse:.4f}, R²: {train_r2:.4f}')

    ax2.scatter(y_test, y_test_pred)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title(f'Actual vs Predicted IC Values (Test Set)\nRMSE: {test_rmse:.4f}, R²: {test_r2:.4f}')

    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted-{model_name}.png', bbox_inches='tight')
    plt.show()

    return importance_df, X_test_selected, y_train_pred, y_test_pred, train_rmse, train_r2, test_rmse, test_r2

# Define plotting feature importances
def plot_feature_importance(importance_df, categorical_cols, categorical_features, model_name):
    importances_df = importance_df
    aggregated_importances = importances_df.copy()

    for col in categorical_cols:
        mask = aggregated_importances['Feature'].str.startswith(col)
        aggregated_importances.loc[mask, 'Feature'] = col

    final_importances_df = aggregated_importances.groupby('Feature').sum().reset_index()

    for col in categorical_cols:
        num_categories = len([feat for feat in categorical_features if feat.startswith(col)])
        final_importances_df.loc[final_importances_df['Feature'] == col, 'Importance'] /= num_categories

    final_importances_df = final_importances_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, len(final_importances_df) * 0.3))
    plt.barh(final_importances_df['Feature'], final_importances_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title("Feature Importance's - Ridge")
    plt.tight_layout()
    plt.savefig(f'feature_importances-{model_name}.png', bbox_inches='tight')
    plt.show()

# Call and scale the data
df, X, y, X_train1, X_test1, y_train, y_test, preprocessed_feature_names, categorical_features, categorical_cols = data_prep()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train1)
X_test = scaler.transform(X_test1)

# Select from model optimization
logging.info("Starting SelectFromModel optimization...")
study_sfm = optimize_model(objective_sfm, X_train, y_train, rfecv)
best_params_sfm = study_sfm.best_params
logging.info(f'Best parameters for SelectFromModel: {best_params_sfm}')

# RFE optimization
logging.info("Starting RFE optimization...")
study_rfe = optimize_model(objective_rfe, X_train, y_train, rfecv)
best_params_rfe = study_rfe.best_params
logging.info(f'Best parameters for RFE: {best_params_rfe}')

# Full model optimization
logging.info("Starting Full model optimization...")
study_full = optimize_model(objective_full, X_train, y_train, rfecv)
best_params_full = study_full.best_params
logging.info(f'Best parameters for Full model: {best_params_full}')

# Train and evaluate models
logging.info("Training and evaluating models...")
model_sfm, selector_sfm = train_model(best_params_sfm, X_train, y_train, 'sfm')
model_rfe, selector_rfe = train_model(best_params_rfe, X_train, y_train, 'rfe')
model_full, _ = train_model(best_params_full, X_train, y_train, 'full')

importance_df_sfm, X_test_selected_sfm, y_train_pred_sfm, y_test_pred_sfm, train_rmse_sfm, train_r2_sfm, test_rmse_sfm, test_r2_sfm = evaluate_model(model_sfm, selector_sfm, X_train, y_train, X_test, y_test, preprocessed_feature_names, "SelectFromModel")
importance_df_rfe, X_test_selected_rfe, y_train_pred_rfe, y_test_pred_rfe, train_rmse_rfe, train_r2_rfe, test_rmse_rfe, test_r2_rfe = evaluate_model(model_rfe, selector_rfe, X_train, y_train, X_test, y_test, preprocessed_feature_names, "RFE")
importance_df_full, _, y_train_pred_full, y_test_pred_full, train_rmse_full, train_r2_full, test_rmse_full, test_r2_full = evaluate_model(model_full, None, X_train, y_train, X_test, y_test, preprocessed_feature_names, "Full Model")

logging.info("Plotting feature importance...")
plot_feature_importance(importance_df_sfm, categorical_cols, categorical_features, "SelectFromModel")
plot_feature_importance(importance_df_rfe, categorical_cols, categorical_features, "RFE")
plot_feature_importance(importance_df_full, categorical_cols, categorical_features, "Full")

# Print summary of results
print("\nSummary of Results:")
print(f"SelectFromModel - Train RMSE: {train_rmse_sfm:.4f}, Train R²: {train_r2_sfm:.4f}, Test RMSE: {test_rmse_sfm:.4f}, Test R²: {test_r2_sfm:.4f}")
print(f"RFE - Train RMSE: {train_rmse_rfe:.4f}, Train R²: {train_r2_rfe:.4f}, Test RMSE: {test_rmse_rfe:.4f}, Test R²: {test_r2_rfe:.4f}")
print(f"Full Model - Train RMSE: {train_rmse_full:.4f}, Train R²: {train_r2_full:.4f}, Test RMSE: {test_rmse_full:.4f}, Test R²: {test_r2_full:.4f}")

# Compare selected features
sfm_features = set(importance_df_sfm['Feature'])
rfe_features = set(importance_df_rfe['Feature'])

print("\nFeature Selection Comparison:")
print(f"Number of features selected by SelectFromModel: {len(sfm_features)}")
print(f"Number of features selected by RFE: {len(rfe_features)}")
print(f"Number of common features: {len(sfm_features.intersection(rfe_features))}")
print("\nCommon features:")
print(sfm_features.intersection(rfe_features))
print("\nFeatures unique to SelectFromModel:")
print(sfm_features - rfe_features)
print("\nFeatures unique to RFE:")
print(rfe_features - sfm_features)

# Calculate feature overlap percentage
overlap_percentage = len(sfm_features.intersection(rfe_features)) / len(sfm_features.union(rfe_features)) * 100

print(f"\nFeature overlap percentage between SelectFromModel and RFE: {overlap_percentage:.2f}%")

# Compare model complexities
print("\nModel Complexities:")
print(f"SelectFromModel - Number of non-zero coefficients: {np.sum(model_sfm.coef_ != 0)}")
print(f"RFE - Number of non-zero coefficients: {np.sum(model_rfe.coef_ != 0)}")
print(f"Full Model - Number of non-zero coefficients: {np.sum(model_full.coef_ != 0)}")

# Calculate percentage improvement over the full model
def safe_percentage_improvement(new_value, base_value, epsilon=1e-5):
    if abs(base_value) < epsilon:
        if abs(new_value - base_value) < epsilon:
            return 0.0 
        else:
            return float('inf') if new_value < base_value else float('-inf')
    else:
        return (base_value - new_value) / base_value * 100

sfm_improvement = safe_percentage_improvement(test_rmse_sfm, test_rmse_full)
rfe_improvement = safe_percentage_improvement(test_rmse_rfe, test_rmse_full)

print("\nPerformance Improvement over Full Model:")
if abs(sfm_improvement) == float('inf'):
    print(f"SelectFromModel - RMSE improvement: {'Significant improvement' if sfm_improvement > 0 else 'Significant degradation'}")
else:
    print(f"SelectFromModel - RMSE improvement: {sfm_improvement:.2f}%")

if abs(rfe_improvement) == float('inf'):
    print(f"RFE - RMSE improvement: {'Significant improvement' if rfe_improvement > 0 else 'Significant degradation'}")
else:
    print(f"RFE - RMSE improvement: {rfe_improvement:.2f}%")

# Identify the best performing model
best_model = min(
    [("SelectFromModel", test_rmse_sfm), ("RFE", test_rmse_rfe), ("Full Model", test_rmse_full)],
    key=lambda x: x[1]
)
print(f"\nBest performing model: {best_model[0]} with Test RMSE: {best_model[1]:.4f}")

# Compare selected features with full model importance
full_model_top_features = set(importance_df_full['Feature'].head(len(sfm_features)).tolist())
print("\nFeature Selection Comparison with Full Model:")
print(f"Number of features in SelectFromModel: {len(sfm_features)}")
print(f"Number of features in RFE: {len(rfe_features)}")
print(f"Number of top features in Full Model: {len(full_model_top_features)}")
print(f"Overlap between SelectFromModel and Full Model top features: {len(sfm_features.intersection(full_model_top_features))}")
print(f"Overlap between RFE and Full Model top features: {len(rfe_features.intersection(full_model_top_features))}")

# Calculate feature importance correlation
sfm_importance = importance_df_sfm.set_index('Feature')['Importance']
rfe_importance = importance_df_rfe.set_index('Feature')['Importance']
full_importance = importance_df_full.set_index('Feature')['Importance']

sfm_full_corr = sfm_importance.corr(full_importance)
rfe_full_corr = rfe_importance.corr(full_importance)
sfm_rfe_corr = sfm_importance.corr(rfe_importance)

print("\nFeature Importance Correlations:")
print(f"SelectFromModel vs Full Model: {sfm_full_corr:.4f}")
print(f"RFE vs Full Model: {rfe_full_corr:.4f}")
print(f"SelectFromModel vs RFE: {sfm_rfe_corr:.4f}")

# Plot learning curves
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f'learning_curve_{title}.png', bbox_inches='tight')
    plt.close()

print("\nGenerating learning curves...")
plot_learning_curve(model_sfm, X_test_selected_sfm, y_test, "SelectFromModel")
plot_learning_curve(model_rfe, X_test_selected_rfe, y_test, "RFE")
plot_learning_curve(model_full, X_test, y_test, "Full Model")

# Final summary
print("\nFinal Summary:")
print(f"1. Best performing model: {best_model[0]} with Test RMSE: {best_model[1]:.4f}")
print(f"2. Feature overlap between SelectFromModel and RFE: {overlap_percentage:.2f}%")
print(f"3. Feature importance correlation (SelectFromModel vs RFE): {sfm_rfe_corr:.4f}")
print(f"4. Alpha values:")
print(f"   - SelectFromModel: {model_sfm}")
print(f"   - RFE: {model_rfe}")
print(f"   - Full Model: {model_full}")
print("\nAnalysis complete")

# Define plotting SHAP values
def plot_shap_values(model, X, feature_names, categorical_cols, categorical_features, model_name):
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    
    feature_to_column = {}
    for feature in feature_names:
        for col in categorical_cols:
            if feature.startswith(col):
                feature_to_column[feature] = col
                break
        else:
            feature_to_column[feature] = feature

    present_columns = set(feature_to_column.values())

    aggregated_shap_values = np.zeros((X.shape[0], len(present_columns)))
    aggregated_features = list(present_columns)

    for i, feature in enumerate(feature_names):
        col_index = aggregated_features.index(feature_to_column[feature])
        aggregated_shap_values[:, col_index] += shap_values[:, i]

    for col in categorical_cols:
        if col in present_columns:
            col_index = aggregated_features.index(col)
            num_categories = len([feat for feat in feature_names if feat.startswith(col)])
            aggregated_shap_values[:, col_index] /= num_categories

    aggregated_X = pd.DataFrame(X, columns=feature_names).groupby(feature_to_column, axis=1).mean()

    plt.figure(figsize=(10, 12))
    shap.summary_plot(aggregated_shap_values, aggregated_X, plot_type="bar", feature_names=aggregated_features, show=False)
    plt.title(f"SHAP Feature Importance - Ridge")
    plt.xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(f"shap_feature_importance-{model_name}.png", bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 12))
    shap.summary_plot(aggregated_shap_values, aggregated_X, feature_names=aggregated_features, show=False)
    plt.title(f"SHAP Summary Plot - Ridge")
    plt.xlabel("SHAP Value")
    plt.tight_layout()
    plt.savefig(f"shap_summary_plot-{model_name}.png", bbox_inches='tight')
    plt.show()
    
# Generate SHAP plots 
logging.info("Generating SHAP plots...")
plot_shap_values(model_sfm, X_test_selected_sfm, importance_df_sfm['Feature'], categorical_cols, categorical_features, "SelectFromModel")
plot_shap_values(model_rfe, X_test_selected_rfe, importance_df_rfe['Feature'], categorical_cols, categorical_features, "RFE")
plot_shap_values(model_full, X_test, preprocessed_feature_names, categorical_cols, categorical_features, "Full Model")

# Save select from model 
Ridge = model_sfm
