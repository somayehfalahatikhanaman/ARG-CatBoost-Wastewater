    
# FULL FEATURES LOGO EXPERIMENT
cities = df['City'].unique()

for target_gene in gene_classes:

    target_col = f'Target_{target_gene}'

    # Binary target based on median
    df[target_col] = (df[target_gene] > df[target_gene].median()).astype(int)
    other_omics = [g for g in gene_classes if g != target_gene]
    current_features = env_features_full + other_omics

    X = df[current_features]
    y = df[target_col]

    for test_city in cities:

        print(f"\n=== FULL FEATURES: Target={target_gene}, Test City={test_city} ===")

        train_idx = df['City'] != test_city
        test_idx = df['City'] == test_city

        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Logistic Regression
        clf_lr = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )

        clf_lr.fit(X_train_scaled, y_train)

        y_pred_lr = clf_lr.predict(X_test_scaled)
        y_proba_lr = clf_lr.predict_proba(X_test_scaled)[:, 1]

        # Random Forest
        clf_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight='balanced',
            random_state=42
        )

        clf_rf.fit(X_train_scaled, y_train)

        y_pred_rf = clf_rf.predict(X_test_scaled)
        y_proba_rf = clf_rf.predict_proba(X_test_scaled)[:, 1]
        
        # XGBoost
        clf_xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )

        clf_xgb.fit(X_train_scaled, y_train)

        y_pred_xgb = clf_xgb.predict(X_test_scaled)
        y_proba_xgb = clf_xgb.predict_proba(X_test_scaled)[:, 1]

        # CatBoost
        clf_cb = CatBoostClassifier(
            iterations=100,
            learning_rate=0.01,
            depth=3,
            l2_leaf_reg=10,
            boosting_type='Plain',
            bootstrap_type='Bernoulli',
            subsample=0.8,
            auto_class_weights='Balanced',
            verbose=0,
            random_state=42
        )

        clf_cb.fit(X_train_scaled, y_train)

        y_pred_cb = clf_cb.predict(X_test_scaled)
        y_proba_cb = clf_cb.predict_proba(X_test_scaled)[:, 1]

        # Save results
        model_outputs = {
            'Logistic Regression': (clf_lr, y_pred_lr, y_proba_lr),
            'Random Forest': (clf_rf, y_pred_rf, y_proba_rf),
            'XGBoost': (clf_xgb, y_pred_xgb, y_proba_xgb),
            'CatBoost': (clf_cb, y_pred_cb, y_proba_cb)
        }

        for model_name, (model, y_pred, y_proba) in model_outputs.items():

            results_list.append({
                'Experiment': 'Full Features',
                'Target_Gene': target_gene,
                'Test_City': test_city,
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred, zero_division=0),
                'ROC_AUC': roc_auc_score(y_test, y_proba)
            })

            pred_df = pd.DataFrame({
                'Experiment': 'Full Features',
                'City': df.loc[test_idx, 'City'].values,
                'Target_Gene': target_gene,
                'Model': model_name,
                'Predicted': y_pred,
                'True': y_test.values
            })

            all_predictions.append(pd.DataFrame({
                'Experiment': 'Full Features',
                'City': df.loc[test_idx, 'City'].values,
                'Target_Gene': target_gene,
                'Model': model_name,
                'Predicted': y_pred,
                'True': y_test.values
            }))

# Save results
results_df_full = pd.DataFrame(results_list)
all_predictions_full = pd.concat(all_predictions, axis=0)

results_df_full.to_csv("results/logo_full_results.csv", index=False)
all_predictions_full.to_csv("results/logo_full_predictions.csv", index=False)

model_summary_full = (
    results_df_full
    .groupby('Model')
    .agg(
        ROC_AUC_median=('ROC_AUC', 'median'),
        ROC_AUC_mean=('ROC_AUC', 'mean'),
        ROC_AUC_std=('ROC_AUC', 'std'),
        F1_median=('F1', 'median'),
        F1_mean=('F1', 'mean'),
        F1_std=('F1', 'std')
    )
    .reset_index()
    .sort_values('ROC_AUC_mean', ascending=False)
)

model_summary_full.to_csv("results/logo_full_model_summary.csv", index=False)

print("LOGO validation completed.")
print(model_summary_full)
