kfold_results_full = []

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

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    model = CatBoostClassifier(
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

    scores = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring={
            'roc_auc': 'roc_auc',
            'f1': 'f1'
        }
    )

    kfold_results_full.append({
        'Experiment': 'Full Features',
        'Target_Gene': target_gene,
        'Model': 'CatBoost',
        'ROC_AUC_mean': scores['test_roc_auc'].mean(),
        'ROC_AUC_std': scores['test_roc_auc'].std(),
        'F1_mean': scores['test_f1'].mean(),
        'F1_std': scores['test_f1'].std()
    })

kfold_df_full = pd.DataFrame(kfold_results_full)

catboost_summary_full = kfold_df_full.agg(
    ROC_AUC_mean=('ROC_AUC_mean', 'mean'),
    ROC_AUC_median=('ROC_AUC_mean', 'median'),
    ROC_AUC_std=('ROC_AUC_mean', 'std'),
    F1_mean=('F1_mean', 'mean'),
    F1_median=('F1_mean', 'median'),
    F1_std=('F1_mean', 'std')
)

print("\n=== CATBOOST 5-FOLD CV FULL FEATURES SUMMARY ===")
print(catboost_summary_full)

print("\n=== CATBOOST 5-FOLD CV FULL FEATURES PER ARG CLASS ===")
print(kfold_df_full.to_string(index=False))

kfold_df_full.to_csv(
    "results/catboost_kfold_full_features_results.csv",
    index=False
)
