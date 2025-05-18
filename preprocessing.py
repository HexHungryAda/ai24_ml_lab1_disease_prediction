import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

# pre-split

def pre_split(df):
    df_processed = df.copy()
    df_processed = df_processed.drop("id", axis=1)
    df_processed["age"] = df["age"] / 365.25
    df_processed = add_bmi(df_processed)
    df_processed = drop_impossible(df_processed)
    return df_processed

def bounds_filter(df, column, min, max):
    rows_to_keep = (df[column] >= min) & (df[column] <= max)
    return df[rows_to_keep].copy()

def drop_inconsistent_ap(df):
    valid_artery_pressure = (df["ap_lo"] <= df["ap_hi"])
    return df[valid_artery_pressure].copy()

def drop_impossible(df):
    """drops weight, height, ap_hi, ap_lo, bmi"""
    df_processed = (df.copy()
                    .pipe(bounds_filter, "weight", 30, 250)
                    .pipe(bounds_filter, "height", 130, 220)
                    .pipe(bounds_filter, "ap_hi", 70, 250)
                    .pipe(bounds_filter, "ap_lo", 40, 150)
                    .pipe(bounds_filter, "bmi", 12, 70)
                    .pipe(drop_inconsistent_ap)
                    )
    return df_processed

def add_bmi(df):
    df_processed = df.copy()
    df_processed["bmi"] = df["weight"] / (df["height"] / 100)**2
    return df_processed 

# post-split

def post_split_cat(X_train, X_val, X_test):
    X_train_processed, X_val_processed, X_test_processed = winsorize(X_train, X_val, X_test)
    X_train_processed = add_bmi_cat(X_train_processed).pipe(add_ap_cat)
    X_val_processed = add_bmi_cat(X_val_processed).pipe(add_ap_cat)
    X_test_processed = add_bmi_cat(X_test_processed).pipe(add_ap_cat)
    return X_train_processed, X_val_processed, X_test_processed 

def add_ap_cat(df):

    ap_conditions = [
        (df["ap_hi"] > 180) | (df["ap_lo"] > 120),
        (df["ap_hi"] >= 140) | (df["ap_lo"] >= 90),
        (df["ap_hi"] >= 130) | (df["ap_lo"] >= 80),
        (df["ap_hi"] >= 120) & (df["ap_lo"] < 80),
        (df["ap_hi"] < 120) & (df["ap_lo"] < 120),
    ]
    ap_labels = [
        "Hypertension crisis",
        "Hyptertension 2",
        "Hypertension 1",
        "Elevated",
        "Healthy"
    ]

    df_processed = df.copy()
    df_processed["ap_cat"] = np.select(ap_conditions, ap_labels, default="Undefined")
    return df_processed

def add_bmi_cat(df):
    df_processed = df.copy()
    bins = [0, 25, 30, 35, 40, np.inf] # [low,high)
    bmi_labels = ["Normal", "Overweight", "Obese 1", "Obese 2", "Obese 3"] 
    df_processed["bmi_cat"] = pd.cut(df["bmi"], bins=bins, labels=bmi_labels, right=False, include_lowest=True)
    return df_processed 

def get_percentile_cap(data_series, lower_quantile=0.01, upper_quantile=0.99):
    lower_bound = data_series.quantile(lower_quantile)
    upper_bound = data_series.quantile(upper_quantile)
    return lower_bound, upper_bound

def winsorize_feature(df, column_name, min, max):
    df_processed = df.copy()
    df_processed[column_name] = np.clip(df_processed[column_name], min, max)
    return df_processed

def winsorize_feature_post_split(X_train, X_val, X_test, column_name, lower_quantile=0.01, upper_quantile=0.99):
    min, max = get_percentile_cap(X_train[column_name], lower_quantile, upper_quantile)

    X_train_processed = winsorize_feature(X_train, column_name, min, max)
    X_val_processed = winsorize_feature(X_val, column_name, min, max)
    X_test_processed = winsorize_feature(X_test, column_name, min, max)
    
    return X_train_processed, X_val_processed, X_test_processed

def winsorize(X_train, X_val, X_test):
    """Bundles the winzorizing. Caps the values."""
    X_train_processed, X_val_processed, X_test_processed = winsorize_feature_post_split(X_train, X_val, X_test, "weight")
    X_train_processed, X_val_processed, X_test_processed = winsorize_feature_post_split(
        X_train_processed, X_val_processed, X_test_processed, "height")
    X_train_processed, X_val_processed, X_test_processed = winsorize_feature_post_split(
        X_train_processed, X_val_processed, X_test_processed, "bmi")
    X_train_processed, X_val_processed, X_test_processed = winsorize_feature_post_split(
        X_train_processed, X_val_processed, X_test_processed, "ap_hi")
    X_train_processed, X_val_processed, X_test_processed = winsorize_feature_post_split(
        X_train_processed, X_val_processed, X_test_processed, "ap_lo")

    return X_train_processed, X_val_processed, X_test_processed 
    # refactor with .pipe()?

def tvt_split(df, target, seed=None):
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

    return X_train, y_train, X_val, y_val, X_test, y_test

# pre_train

def pre_train_num(X_train, X_val, X_test):
    columns_to_drop = ["height", "weight"]
    X_train_processed = X_train.drop(columns=columns_to_drop)
    X_val_processed = X_val.drop(columns=columns_to_drop)
    X_test_processed = X_test.drop(columns=columns_to_drop)

    ohe_features = ["gender"]
    X_train_processed, X_val_processed, X_test_processed = one_hot_encode_features(
        X_train_processed, X_val_processed, X_test_processed, ohe_features)

    columns_to_scale = ["age", "ap_hi", "ap_lo", "bmi", "cholesterol", "gluc"]
    X_train_processed, X_val_processed, X_test_processed = scaler(
        X_train_processed, X_val_processed, X_test_processed, columns_to_scale, scaler="standard")
    X_train_processed, X_val_processed, X_test_processed = scaler(
        X_train_processed, X_val_processed, X_test_processed, columns_to_scale, scaler="minmax")
    return X_train_processed, X_val_processed, X_test_processed

def pre_train_cat(X_train, X_val, X_test):
    columns_to_drop = ['bmi', 'ap_hi', 'ap_lo', 'weight', 'height']
    X_train_processed = X_train.drop(columns=columns_to_drop)
    X_val_processed = X_val.drop(columns=columns_to_drop)
    X_test_processed = X_test.drop(columns=columns_to_drop)
    
    ohe_features = ["bmi_cat", "ap_cat", "gender"] 
    X_train_processed, X_val_processed, X_test_processed = one_hot_encode_features(
        X_train_processed, X_val_processed, X_test_processed, ohe_features)
    
    columns_to_scale = ["age", "cholesterol", "gluc"]
    X_train_processed, X_val_processed, X_test_processed = scaler(
        X_train_processed, X_val_processed, X_test_processed, columns_to_scale, scaler="standard")
    X_train_processed, X_val_processed, X_test_processed = scaler(
        X_train_processed, X_val_processed, X_test_processed, columns_to_scale, scaler="minmax")
    return X_train_processed, X_val_processed, X_test_processed

def one_hot_encode_features(X_train, X_val, X_test, cat_features):
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    X_test_processed = X_test.copy()

    X_train_cats = X_train_processed[cat_features]
    X_val_cats = X_val_processed[cat_features]
    X_test_cats = X_test_processed[cat_features]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(X_train_cats)

    X_train_encoded_arr = ohe.transform(X_train_cats)
    X_val_encoded_arr = ohe.transform(X_val_cats)
    X_test_encoded_arr = ohe.transform(X_test_cats)

    encoded_column_names = ohe.get_feature_names_out(cat_features)

    # Create dataframes from the arrays
    X_train_encoded = pd.DataFrame(X_train_encoded_arr, columns=encoded_column_names, index=X_train.index)
    X_val_encoded = pd.DataFrame(X_val_encoded_arr, columns=encoded_column_names, index=X_val.index)
    X_test_encoded = pd.DataFrame(X_test_encoded_arr, columns=encoded_column_names, index=X_test.index)

    X_train_processed = X_train_processed.drop(columns=cat_features)
    X_val_processed = X_val_processed.drop(columns=cat_features)
    X_test_processed = X_test_processed.drop(columns=cat_features)

    X_train_processed = pd.concat([X_train_processed, X_train_encoded], axis=1)
    X_val_preprocessed = pd.concat([X_val_processed, X_val_encoded], axis=1)
    X_test_preprocessed = pd.concat([X_test_processed, X_test_encoded], axis=1)

    return X_train_processed, X_val_preprocessed, X_test_preprocessed
    # refactor opportunity probably

def scaler(X_train, X_val, X_test, num_columns, scaler="standard"):
    X_train_scaled, X_val_scaled, X_test_scaled = X_train.copy(), X_val.copy(), X_test.copy()
    if scaler == "minmax":
        scaler = MinMaxScaler()
    else: scaler = StandardScaler()
    X_train_scaled[num_columns] = scaler.fit_transform(X_train[num_columns])
    X_val_scaled[num_columns] = scaler.transform(X_val[num_columns])
    X_test_scaled[num_columns] = scaler.transform(X_test[num_columns])
    return X_train_scaled, X_val_scaled, X_test_scaled