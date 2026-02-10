# src/model/main.py

from src.preprocess import preprocessing as pp
from src.loader import data_loader as dl
from src.model import evaluate_model as em
from src.model import train_model as tm

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

if __name__ == "__main__":
    
    from config import LOAN_FILE, MODEL_DIR, PKL_DIR
    
    df = dl.load_data(LOAN_FILE)
    df = dl.inspect_and_prepare_data(df)
    dl.print_dataframe_stats(df)

    print("[INFO] Data loaded and preprocessed successfully.")

    df_processed, dropped_missing = pp.handle_missing_values(df.copy(), missing_percentage_threshold=30)
    df_processed = pp.convert_specific_categorical_to_numeric(df_processed)
    df_processed = pp.impute_missing_values(df_processed)
    df_processed, dropped_low_variance = pp.drop_low_variance_features(df_processed, variance_threshold=0.01)
    df_processed, dropped_descriptive = pp.drop_descriptive_features(df_processed)
    df_processed, dropped_correlated = pp.drop_highly_correlated_features(df_processed, correlation_threshold=0.6)
    df_processed, dropped_low_ks = pp.drop_low_ks_features(df_processed, target_col='bad_good', ks_threshold=0.03)
    df_processed, dropped_low_iv, iv_df = pp.drop_low_iv_categorical_features(df_processed, df_processed.select_dtypes(include=['object']).columns.tolist(), target='bad_good', iv_threshold=0.02)

    df_processed, dropped_low_ks_iv = select_top_features_manual(df_processed, target='bad_good')

    print("\n[INFO] Processed DataFrame head:")
    print(df_processed.head())
    print("\n[INFO] Processed DataFrame info:")    
    df_processed.info()

    print(f"\n[INFO] Processed DataFrame shape: {df_processed.shape}")

    X_train, X_test, y_train, y_test = split_data(df_processed, target_col='bad_good', test_size=0.3)
    
    write_pkl_file(X_train, path.join(PKL_DIR, "X_train.pkl"))
    write_pkl_file(y_train, path.join(PKL_DIR, "y_train.pkl"))
    write_pkl_file(X_test, path.join(PKL_DIR, "X_test.pkl"))      
    write_pkl_file(y_test, path.join(PKL_DIR, "y_test.pkl"))

    print("\n[INFO] Data split into training and testing sets and saved as pickle files.")
    
    rf_pipeline, rf_metrics, X_train_encoded, cat_mappings = tm.train_random_forest_optimized(
        X_train, y_train,
        n_estimators=200,
        max_features='sqrt',
        max_depth=5,
        verbose=True
    )
    
    rf_paths = save_model(
        model=rf_pipeline,
        metrics=rf_metrics,
        X_train_encoded=X_train_encoded,
        category_mappings=cat_mappings,
        model_name="random_forest_NEW_encoded",
        output_dir=MODEL_DIR
    )
    
    catboost_model, catboost_metrics = None, None
    if CATBOOST_AVAILABLE:
        catboost_model, catboost_metrics, X_train_encoded, cat_mappings = tm.train_catboost_optimized(
            X_train, y_train,
            depth=5,
            iterations=200,
            verbose=True,
            learning_rate=1
        )

    cb_paths = save_model(
        model=catboost_model,
        metrics=catboost_metrics,
        X_train_encoded=X_train_encoded,
        category_mappings=cat_mappings,
        model_name="catboost_NEW_encoded",
        output_dir=MODEL_DIR
    )
    
    lightgbm_model, lightgbm_metrics = None, None
  
    lightgbm_model, lightgbm_metrics, X_train_encoded, cat_mappings = tm.train_lightgbm_optimized(
        X_train, y_train,
        max_depth=5,
        n_estimators=200,
        validation_split=0.3,
        verbose=True
    )
    
    lightgbm_paths = save_model(
        model=lightgbm_model,
        metrics=lightgbm_metrics,
        X_train_encoded=X_train_encoded,
        category_mappings=cat_mappings,
        model_name="lightgbm_NEW_encoded",
        output_dir=MODEL_DIR
    )
    
    comparison = compare_models(
        rf_metrics=rf_metrics,
        catboost_metrics=catboost_metrics,
        lightgbm_metrics=lightgbm_metrics,
        focus_metric='Train F1'
    )
    
    print("\n[INFO] -------------------- Models Saved --------------------")

    print(f"\n[INFO] Random Forest:")
    print(f"  {rf_paths['model']}")
    print(f"\n[INFO] CatBoost:")
    print(f"  {cb_paths['model']}")
    print(f"\n[INFO] LightGBM:")
    print(f"  {lightgbm_paths['model']}")
    print("\n[INFO] -------------------- Save Complete --------------------")

    print("\n[INFO] CatBoost model evaluating start.")

    X_test_encoded = tm.encode_data_like_training(X_test, cat_mappings)
    threshold = catboost_metrics['optimal_threshold']
    em.evaluate_model(catboost_model, X_test_encoded, y_test, threshold=threshold)
    print("\n[INFO] CatBoost model evaluating successfully.")
    print(cat_mappings)

    print("For analizing fuzzy inference system go to Execute Fuzzy Inference.ipynb")