# SHAP for CatBoost 
all_shap = []
models_dict = {}

explainer = shap.TreeExplainer(clf_cb)
shap_values = explainer.shap_values(X_test_scaled)

shap_df = pd.DataFrame(shap_values, columns=current_features)

shap_df['City'] = df.loc[test_idx, 'City'].values
shap_df['Target_Gene'] = target_gene
shap_df['Test_City'] = test_city
shap_df['Experiment'] = 'Full Features'

all_shap.append(shap_df)

models_dict[f"Full_{target_gene}_{test_city}"] = {
    'model': clf_cb,
    'scaler': scaler,
    'features': current_features,
    'test_city': test_city
}

all_shap_df_full = pd.concat(all_shap, axis=0)

all_shap_df_full.to_csv(
    "results/catboost_logo_full_shap_values.csv",
    index=False
)

shap.summary_plot(
    all_shap_df_full[env_features_full].values,
    features=all_shap_df_full[env_features_full],
    feature_names=env_features_full
)
