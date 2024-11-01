# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define objective
def objective(trial, X, y):
    params = {
    'max_depth': trial.suggest_int("max_depth", 4, 20),
    'min_samples_split': trial.suggest_int("min_samples_split", 2, 50),
    'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 50),
    'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    'criterion': trial.suggest_categorical("criterion", ["gini", "entropy"])
    }

    n_features_to_select = trial.suggest_int("n_features_to_select", 5, X.shape[1])

    model = DecisionTreeClassifier(**params, random_state=42)

    selector = SelectFromModel(model, max_features=n_features_to_select, threshold=-np.inf)
    X_selected = selector.fit_transform(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy', n_jobs=-1)

    return scores.mean()

# Define Optuna initialisation
def optimize_model(X_train, y_train):
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(direction='maximize', pruner=pruner, sampler=optuna.samplers.TPESampler(seed=42))

    objective_with_data = partial(objective, X=X_train, y=y_train)

    try:
        study.optimize(objective_with_data, n_trials=500, n_jobs=-1,
                       callbacks=[lambda study, trial: study.stop() if study.best_value >= 0.95 else None])
    except Exception as e:
        logging.error(f"An error occurred during optimization: {e}")
        return None

    return study

#  Define training of the final model
def train_final_model(best_params, X_train, y_train):
    DT_best_model = DecisionTreeClassifier(**{k: v for k, v in best_params.items() if k != 'n_features_to_select'},
                                           random_state=42)
    selector = SelectFromModel(DT_best_model, max_features=best_params['n_features_to_select'], threshold=-np.inf)
    X_train_selected = selector.fit_transform(X_train, y_train)
    DT_final_model = DecisionTreeClassifier(**{k: v for k, v in best_params.items() if k != 'n_features_to_select'},
                                            random_state=42)
    DT_final_model.fit(X_train_selected, y_train)
    return DT_final_model, selector, DT_best_model, X_train_selected

# Define evaluation of the final model
def evaluate_model(model, selector, X_test, y_test, preprocessed_feature_names):
    X_test_selected = selector.transform(X_test)
    test_accuracy = model.score(X_test_selected, y_test)
    logging.info(f"Test accuracy: {test_accuracy:.4f}")

    y_pred = model.predict(X_test_selected)
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred))

    logging.info("\nConfusion Matrix:")
    logging.info(confusion_matrix(y_test, y_pred))

    importances = model.feature_importances_
    
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
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importances of the Decision Tree Model')
    plt.tight_layout()
    plt.savefig('feature_importances-DT-class.png', bbox_inches='tight')
    plt.show()

# Define plotting optimisation history
def plot_optimization_history(study):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(study.trials)), [t.value for t in study.trials], marker='o')
    plt.xlabel('Number of Trials')
    plt.ylabel('Objective Value')
    plt.title('Optimization History of the Decision Tree Model')
    plt.savefig('ophistory-DT-class.png', bbox_inches='tight')
    plt.show()

# Define plotting Optuna parameter importances
def plot_param_importances(study):
    importances = optuna.importance.get_param_importances(study)
    plt.figure(figsize=(10, 6))
    plt.bar(importances.keys(), importances.values())
    plt.xlabel('Hyperparameters')
    plt.ylabel('Importance')
    plt.title('Hyperparameter Importances of the Decision Tree Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('param_importances-DT-class.png', bbox_inches='tight')
    plt.show()

# Call the data
df, X, y, X_train, X_test, y_train, y_test, preprocessed_feature_names, categorical_features, categorical_cols = data_prep()

# Call optimization
logging.info("Starting model optimization...")
study = optimize_model(X_train, y_train)

best_params = study.best_params
logging.info(f'Best parameters: {best_params}')

# Train final model
logging.info("Training final model...")
final_model_DT, selector, best_model_DT, X_train_selected = train_final_model(best_params, X_train, y_train)

# Evaluate final model
logging.info("Evaluating model...")
importance_df, X_test_selected = evaluate_model(final_model_DT, selector, X_test, y_test, preprocessed_feature_names)

logging.info("Plotting feature importance...")
plot_feature_importance(importance_df, categorical_cols, categorical_features)

logging.info("Plotting optimization results...")
plot_optimization_history(study)
plot_param_importances(study)

# Evaluate the final model on the test data
y_pred = final_model_DT.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Test accuracy:", test_accuracy)

y_pred = final_model_DT.predict(X_train_selected)
train_accuracy = accuracy_score(y_train, y_pred)
print(classification_report(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))
print("Train accuracy:", train_accuracy)

print(final_model_DT)
