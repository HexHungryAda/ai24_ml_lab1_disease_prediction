import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# pre-split

def eda(df):
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    pie_chart(df, "cholesterol", axes[0, 0], {1: "Normal", 2: "Above normal", 3: "Well-above normal"},
                            "Cholesterol distribution")
    pie_chart(df, "cardio", axes[0, 1], {0: "Healthy", 1: "Sick"}, "Cardiovascular disease distribution")
    pie_chart(df, "smoke", axes[0, 2], {0: "Not-smoker", 1: "Smoker"}, "Smoker distribution")
    pie_chart(df, "gender", axes[0, 3], {1: "Women", 2: "Men"}, "Gender distribution")
    histogram(df, axes[1, 0], "weight", "Weight distribution", "Weight (kg)", column_by="gender")
    histogram(df, axes[1, 1], "height", "Height distribution", "Height (cm)", column_by="gender")
    histogram(df, axes[1, 2], "age", "Age distribution", "Age (years)", column_by="gender")
    cardio_by_gender(df, axes[1, 3])

    fig.tight_layout()
    fig.show()

    print(f"rows: {len(df)}")
    return None

def pie_chart(df, column, ax, column_labels, title, colors=sns.color_palette("pastel")):
    column_proportions = df[column].value_counts(normalize=True)
    labelled_proportions = column_proportions.rename(index=column_labels)

    ax.set_title(title, fontweight="bold")
    ax.pie(labelled_proportions, labels=labelled_proportions.index, startangle=90, colors=colors, autopct="%.2f%%")

    return None 

def histogram(df, ax, column, title, xlabel, ylabel="Density", column_by="gender", palette="pastel", bins=80):
    # could change such that legend labels man, woman. low priority
    data_series = df[column]
    data_series_by = df[column_by]
    sns.histplot(x=data_series, hue=data_series_by, element="step", stat="density", 
                 common_norm=False, palette=palette, ax=ax, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return None

def cardio_by_gender(df, ax):

    # since cardio is boolean (1 xor 0) mean gives proportion of 1s.
    proportion_df = df.groupby('gender')["cardio"].mean().reset_index()
    proportion_df.columns = ['gender', "p(cardio)"]

    sns.barplot(
        x="gender",
        hue="gender",
        y="p(cardio)",
        data=proportion_df,
        palette="pastel",
        ax=ax, 
        legend=False
    )

    ax.set_title("p(cardiovascular_disease) by gender")
    ax.set_ylim(0, 1)
    return None 

# post-split

def correlation_heatmap(X_train, y_train, ax):
    cols_to_correlate = ["age", "height", "weight", "ap_hi", "ap_lo", "smoke", "alco", "active", "bmi"]
    df_corr = X_train[cols_to_correlate].copy()
    df_corr["cardio"] = y_train
    correlation_matrix = df_corr.corr()

    sns.heatmap(
        correlation_matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        vmin=-1,
        vmax=1,
        center=0 
    )
    ax.set_title("Correlation matrix", fontweight="bold")
    return None 

def cardio_conditional_feature(df, ax, feature, title_prefix="p(cardio) given", palette="pastel"):
    proportion_df = df.groupby(feature)["cardio"].mean().reset_index()
    proportion_df.columns = [feature, "p(cardio)"]

    sns.barplot(
        x=feature,
        hue=feature,
        y="p(cardio)",
        data=proportion_df,
        palette=palette,
        ax=ax, 
        legend=False
    )

    ax.set_title(f"{title_prefix} {feature}")
    ax.set_ylim(0, 1)
    return None
    # should also sort

def cardio_conditional_features(X_train, y_train):
    """Just aggregation code to draw entire figure"""
    df = X_train.copy()
    df["cardio"] = y_train
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    cardio_conditional_feature(df, axes[0, 0], "bmi_cat")
    cardio_conditional_feature(df, axes[0, 1], "ap_cat")
    cardio_conditional_feature(df, axes[0, 2], "smoke")
    cardio_conditional_feature(df, axes[1, 0], "gluc")
    cardio_conditional_feature(df, axes[1, 1], "cholesterol")
    cardio_conditional_feature(df, axes[1, 2], "active")

    fig.tight_layout()
    fig.show()

# post-train

def confusion_matrix(test_set_eval):
    cm_test_data = np.array(test_set_eval['confusion_matrix']) 

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_test_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Healthy (0)', 'Predicted Sick (1)'],
                yticklabels=['Actual Healthy (0)', 'Actual Sick (1)'])
    plt.title('Confusion Matrix for Final Ensemble Model (Test Set)', fontsize=14)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.show()
    return None
