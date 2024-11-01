# Define the individual models
logistic_regression = LR_final_model
decision_tree = DT_final_model
random_forest = RF_final_model
xgboost_model = XGB_final_model
neural_network = MLP_final_model

# Define the ensemble model
voting_ensemble = VotingClassifier(estimators=[('lr', logistic_regression), ('dt', decision_tree), ('rf', random_forest), ('xgb', xgboost_model), ('nn', neural_network)])

# Train the ensemble model 
voting_ensemble.fit(X_train, y_train)

# Evaluate the ensemble model on the test set
y_pred_voting = voting_ensemble.best_estimator_.predict(X_test)

print(f'Voting ensemble test accuracy: {accuracy_score(y_test, y_pred_voting):.4f}')
