import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

# Try to import ensemble models with fallbacks
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available")

# Fallback to sklearn if gradient boosting models not available
if not any([LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE, CATBOOST_AVAILABLE]):
    from sklearn.ensemble import RandomForestClassifier

    print(
        "Warning: No gradient boosting libraries available, falling back to RandomForest"
    )

# Try to import SHAP for feature importance
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available, using model built-in feature importance")

warnings.filterwarnings("ignore")


# Simple rule-based signal: 1 if short-term SMA > long-term SMA, else 0
def rule_signal(feature: pd.DataFrame, test_start_idx: int = None) -> pd.Series:
    """Generate rule-based trading signals.

    Args:
        feature: DataFrame with technical indicators
        test_start_idx: Optional index to start predictions from (for fair ML comparison)
    """
    signals = (feature["sma_10"] > feature["sma_50"]).astype(int)

    # If test_start_idx is provided, only return predictions from that point onwards
    if test_start_idx is not None:
        aligned_signals = pd.Series(0, index=feature.index, dtype=int)
        aligned_signals.iloc[test_start_idx:] = signals.iloc[test_start_idx:]
        return aligned_signals

    return signals


# Sophisticated ML-based signal using LightGBM with enhanced features
def ml_signal(feature: pd.DataFrame, return_aligned_rule: bool = False) -> pd.Series:
    """Generate ML-based trading signals.

    Args:
        feature: DataFrame with technical indicators
        return_aligned_rule: If True, returns tuple (ml_signals, rule_signals_aligned)
    """
    # Enhanced feature set
    feature_cols = [
        # Original features
        "rsi_14",
        "sma_10",
        "sma_50",
        "bb_low",
        "bb_high",
        # New technical indicators
        "macd",
        "macd_signal",
        "macd_diff",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "atr",
        "bb_width",
        "bb_position",
        # Price and volume features
        "high_low_pct",
        "close_open_pct",
        "vol_roc",
        "mfi",
        # Momentum features
        "momentum_5",
        "momentum_10",
        "momentum_20",
        # Volatility features
        "volatility_10",
        "volatility_30",
        # Moving averages
        "ema_12",
        "ema_26",
        "sma_200",
        # Position features
        "position_52w",
    ]

    # Filter to only use available features
    available_features = [col for col in feature_cols if col in feature.columns]
    X = feature[available_features]

    # Create target: 1 if next day return is positive, 0 otherwise
    y = (feature["ret_1d"].shift(-1) > 0).astype(int)

    # Remove last row since we don't have next day return
    X, y = X.iloc[:-1], y.iloc[:-1]

    # Handle NaN values by dropping rows with any missing values
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    if len(X_clean) < 100:  # Increased minimum samples for ensemble model
        empty_predictions = pd.Series(0, index=feature.index, dtype=int)
        if return_aligned_rule:
            # Return empty rule signals aligned as well
            empty_rule_signals = pd.Series(0, index=feature.index, dtype=int)
            return empty_predictions, empty_rule_signals
        return empty_predictions

    print(
        f"ML model: Processing {len(X_clean)} samples with {len(available_features)} features"
    )

    # Use expanding window approach for fair comparison with rule-based model
    # This ensures both models predict on the same time range
    min_train_size = max(
        100, int(len(X_clean) * 0.3)
    )  # At least 30% for initial training
    predictions = pd.Series(0, index=feature.index, dtype=int)

    print(f"ML model: Using expanding window starting with {min_train_size} samples")

    # Generate predictions using expanding window (retrain every 60 days)
    for i in range(min_train_size, len(X_clean), 60):
        # Training data: everything up to current point
        X_train = X_clean.iloc[:i]
        y_train = y_clean.iloc[:i]

        # Test data: next 60 days (or remaining data)
        end_idx = min(i + 60, len(X_clean))
        X_test = X_clean.iloc[i:end_idx]

        if len(X_test) == 0:
            break

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), index=X_test.index, columns=X_test.columns
        )

        # Create XGBoost classifier
        if XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=4,
                verbosity=0,
                eval_metric="logloss",
            )
            print("Using XGBoost classifier")
        elif LIGHTGBM_AVAILABLE:
            # Fallback to LightGBM if XGBoost not available
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                min_split_gain=0.01,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                class_weight="balanced",
                n_jobs=4,
                verbosity=-1,
            )
            print("XGBoost not available, using LightGBM classifier")
        else:
            # Final fallback to RandomForest
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                bootstrap=True,
                random_state=42,
                class_weight="balanced",
                n_jobs=4,
            )
            print("XGBoost and LightGBM not available, using RandomForest classifier")

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Make predictions with probability threshold
        pred_proba = model.predict_proba(X_test_scaled)
        preds = (pred_proba[:, 1] > 0.55).astype(int)

        # Store predictions
        predictions.loc[X_test.index] = preds

    total_predictions = predictions.sum()
    print(
        f"ML model: Generated {total_predictions} signals out of {len(predictions)} total samples"
    )

    # SHAP feature importance analysis (using the trained model)
    try:
        if "model" in locals():  # Check if we trained a model
            print("ML model: Computing SHAP values...")

            if SHAP_AVAILABLE:
                explainer = shap.TreeExplainer(model)
                # Use a sample of the last training data for SHAP analysis (for performance)
                sample_size = min(100, len(X_train_scaled))
                shap_values = explainer.shap_values(
                    X_train_scaled.sample(sample_size, random_state=42)
                )

                # Get mean absolute SHAP values for feature importance
                if isinstance(
                    shap_values, list
                ):  # Binary classification returns list of arrays
                    shap_importance = pd.DataFrame(
                        {
                            "feature": X_train_scaled.columns,
                            "shap_importance": abs(shap_values[1]).mean(
                                axis=0
                            ),  # Use positive class SHAP values
                        }
                    ).sort_values("shap_importance", ascending=False)
                else:
                    shap_importance = pd.DataFrame(
                        {
                            "feature": X_train_scaled.columns,
                            "shap_importance": abs(shap_values).mean(axis=0),
                        }
                    ).sort_values("shap_importance", ascending=False)

                print("Top 5 most important features (SHAP):")
                print(shap_importance.head().to_string(index=False))
            else:
                # Use built-in feature importance
                if hasattr(model, "feature_importances_"):
                    feature_importance = pd.DataFrame(
                        {
                            "feature": X_train_scaled.columns,
                            "importance": model.feature_importances_,
                        }
                    ).sort_values("importance", ascending=False)
                    print("Top 5 most important features (model built-in):")
                    print(feature_importance.head().to_string(index=False))
        else:
            print("No model trained - insufficient data")

    except Exception as e:
        print(f"Feature importance analysis failed: {e}")
        # Try fallback to built-in feature importance if available
        if "model" in locals() and hasattr(model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {
                    "feature": X_train_scaled.columns,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            print("Top 5 most important features (fallback built-in):")
            print(feature_importance.head().to_string(index=False))

    # Optionally return rule signals aligned to same prediction range
    if return_aligned_rule:
        # Calculate the test start index in the original feature DataFrame
        # We need to account for the rows removed due to NaN values
        original_test_start_idx = min_train_size + (len(feature) - len(X_clean) - 1)
        rule_signals_aligned = rule_signal(
            feature, test_start_idx=original_test_start_idx
        )
        return predictions, rule_signals_aligned

    return predictions
