from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier
import pandas as pd

results_list = []
all_predictions = []

# FULL FEATURES LOGO EXPERIMENT
cities = df['City'].unique()
env_features_full = env_features.copy()

for target_gene in gene_classes:

    target_col = f'Target_{target_gene}'

    df[target_col] = (
        df[target_gene] > df[target_gene].median()
    ).astype(int)

    other_omics = [g for g in gene_classes if g != target_gene]
    current_features = env_features_full + other_omics

    X = df[current_features]
    y = df[target_col]

    for test_city in cities:

        print(f"\n=== CatBoost LOGO: Target={target_gene}, Test City={test_city} ===")

        train_idx = df['City'] != test_city
        test_idx = df['City'] == test_city

        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

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

        results_list.append({
            'Experiment': 'Full Features',
            'Target_Gene': target_gene,
            'Test_City': test_city,
            'Model': 'CatBoost',
            'Accuracy': accuracy_score(y_test, y_pred_cb),
            'F1': f1_score(y_test, y_pred_cb, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test, y_proba_cb)
        })

        all_predictions.append(pd.DataFrame({
            'Experiment': 'Full Features',
            'City': df.loc[test_idx, 'City'].values,
            'Target_Gene': target_gene,
            'Model': 'CatBoost',
            'Predicted': y_pred_cb,
            'Probability': y_proba_cb,
            'True': y_test.values
        }))

# Save results
results_df_full = pd.DataFrame(results_list)
all_predictions_full = pd.concat(all_predictions, axis=0)

results_df_full.to_csv(
    "results/catboost_logo_full_results.csv",
    index=False
)

all_predictions_full.to_csv(
    "results/catboost_logo_full_predictions.csv",
    index=False
)

catboost_summary_full = (
    results_df_full
    .agg(
        ROC_AUC_median=('ROC_AUC', 'median'),
        ROC_AUC_mean=('ROC_AUC', 'mean'),
        ROC_AUC_std=('ROC_AUC', 'std'),
        F1_median=('F1', 'median'),
        F1_mean=('F1', 'mean'),
        F1_std=('F1', 'std')
    )
)

catboost_summary_full.to_csv(
    "results/catboost_logo_full_summary.csv"
)

print("CatBoost LOGO validation completed.")
print(catboost_summary_full)
