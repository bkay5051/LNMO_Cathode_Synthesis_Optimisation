# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define objective
def objective(trial, X, y):
    params = {
        'penalty': trial.suggest_categorical("penalty", ["l1", "l2"]),
        'C': trial.suggest_float("C", 1e-5, 10, log=True),
        'max_iter': 5000
    }
    
    if params['penalty'] == "l1":
        params['solver'] = "liblinear"
    else:
        params['solver'] = "lbfgs"

    n_features_to_select = trial.suggest_int("n_features_to_select", 5, X.shape[1])

    model = LogisticRegression(**params, random_state=42)
    selector = SelectFromModel(model, max_features=n_features_to_select, threshold=-np.inf)
    X_sel = selector.fit_transform(X, y)
    scaler = StandardScaler()
    X_selected = scaler.fit_transform(X_sel)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy', n_jobs=-1)

    return scores.mean()

# Define Optuna initialisation
def optimize_model(X_train, y_train):
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(direction='maximize', pruner=pruner, sampler=optuna.samplers.TPESampler(seed=42))

    objective_with_data = partial(objective, X=X_train, y=y_train)

    try:
        study.optimize(objective_with_data, n_trials=100, n_jobs=-1,
                       callbacks=[lambda study, trial: study.stop() if study.best_value >= 0.95 else None])
    except Exception as e:
        logging.error(f"An error occurred during optimization: {e}")
        return None

    return study

#  Define training of the final model
def train_final_model(best_params, X_train, y_train):
    LR_best_model = LogisticRegression(**{k: v for k, v in best_params.items() if k != 'n_features_to_select'},
                                           random_state=42, max_iter=5000)
    selector = SelectFromModel(LR_best_model, max_features=best_params['n_features_to_select'], threshold=-np.inf)
    X_train_sel = selector.fit_transform(X_train, y_train)
    scaler = StandardScaler()
    X_train_selected = scaler.fit_transform(X_train_sel)
    LR_final_model = LogisticRegression(**{k: v for k, v in best_params.items() if k != 'n_features_to_select'},
                                            random_state=42)
    LR_final_model.fit(X_train_selected, y_train)
    return LR_final_model, selector, LR_best_model, X_train_selected

# Define evaluation of the final model
def evaluate_model(model, selector, X_test, y_test, preprocessed_feature_names):
    X_test_sel = selector.transform(X_test)
    scaler = StandardScaler()
    X_test_selected = scaler.fit_transform(X_test_sel)
    test_accuracy = model.score(X_test_selected, y_test)
    logging.info(f"Test accuracy: {test_accuracy:.4f}")

    y_pred = model.predict(X_test_selected)
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred))

    logging.info("\nConfusion Matrix:")
    logging.info(confusion_matrix(y_test, y_pred))

    importances = np.abs(model.coef_[0])

    selected_indices = selector.get_support(indices=True)

    selected_preprocessed_features = [preprocessed_feature_names[i] for i in selected_indices]

    importance_df = pd.DataFrame({
        'Feature': selected_preprocessed_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    logging.info("\nTop 10 most important features:")
    logging.info(importance_df.head(10))

    return importance_df, X_test_selected

# Define plotting feature importances
def plot_feature_importance(importance_df, categorical_cols, categorical_features):
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
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Features')
    plt.title('Feature Importances of the Logistic Regression Model')
    plt.tight_layout()
    plt.savefig('feature_importances-LR-class.png', bbox_inches='tight')
    plt.show()

# Define plotting optimisation history
def plot_optimization_history(study):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(study.trials)), [t.value for t in study.trials], marker='o')
    plt.xlabel('Number of Trials')
    plt.ylabel('Objective Value')
    plt.title('Optimization History of the Logistic Regression Model')
    plt.savefig('ophistory-LR-class.png', bbox_inches='tight')
    plt.show()

# Define plotting Optuna parameter importances
def plot_param_importances(study):
    importances = optuna.importance.get_param_importances(study)
    plt.figure(figsize=(10, 6))
    plt.bar(importances.keys(), importances.values())
    plt.xlabel('Hyperparameters')
    plt.ylabel('Importance')
    plt.title('Hyperparameter Importances of the Logistic Regression Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('param_importances-LR-class.png', bbox_inches='tight')
    plt.show()

# Call the data
df, X, y, X_train, X_test, y_train, y_test, preprocessed_feature_names, categorical_features, categorical_cols = data_prep()

# Call optimization
logging.info("Starting model optimization...")
study = optimize_model(X_train, y_train)

best_params = study.best_params
if best_params['penalty'] == "l1":
    best_params['solver'] = "liblinear"
else:
    best_params['solver'] = "lbfgs"
logging.info(f'Best parameters: {best_params}')

# Train final model
logging.info("Training final model...")
final_model_LR, selector, best_model_LR, X_train_selected = train_final_model(best_params, X_train, y_train)

# Evaluate final model
logging.info("Evaluating model...")
importance_df, X_test_selected = evaluate_model(final_model_LR, selector, X_test, y_test, preprocessed_feature_names)

logging.info("Plotting feature importance...")
plot_feature_importance(importance_df, categorical_cols, categorical_features)

logging.info("Plotting optimization results...")
plot_optimization_history(study)
plot_param_importances(study)

# Evaluate the final model on the test data
y_pred = final_model_LR.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Test accuracy:", test_accuracy)

y_pred = final_model_LR.predict(X_train_selected)
train_accuracy = accuracy_score(y_train, y_pred)
print(classification_report(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))
print("Train accuracy:", train_accuracy)

print(final_model_LR)
