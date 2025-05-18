from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, fbeta_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import time
import pandas as pd
import numpy as np
import visualization


def tuning(X_train, y_train, X_val, y_val, dataset_name):
    model_configs = {
        "Logistic regression": {
            "model": LogisticRegression(random_state=42, max_iter=3000),
            "params": {
                'C': [0.01, 0.1, 1.0, 10.0], # regularization
                'penalty': ['l1', 'l2'], # regularization type
                'solver': ['liblinear']
            } 
        },
        "Random forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, 15], # more can lead to overfit
                'min_samples_leaf': [5, 10, 20] # more lead to simpler models 
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                'n_neighbors': [9, 11, 15]
            }    
        }
    }

    # scoring metric
    beta_value = 3
    f_beta_scorer = make_scorer(fbeta_score, beta=beta_value, pos_label=1, zero_division=0)

    best_tuned_models = {}
    validation_set_evaluations = {}

    print(f"Starting hyperparameter tuning for {dataset_name} dataset")

    for model_name, config in model_configs.items():
        print(f"tuning {model_name}")
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring=f_beta_scorer,
            cv=5,
            verbose=0,
            n_jobs=-2,
            return_train_score=True
        )

        start_time_train = time.time()
        grid_search.fit(X_train, y_train)
        fit_time = time.time() - start_time_train
        print(f"gridsearch fit for {model_name} completed {fit_time:.2f} seconds")

        best_model_instance = grid_search.best_estimator_
        best_tuned_models[model_name] = best_model_instance

        # get the train-set scoring as well
        best_model_index = grid_search.best_index_

        train_f_beta = grid_search.cv_results_["mean_train_score"][best_model_index]

        # predict on separate val data using best model
        print(f"Evaluating best {model_name} on val data")
        y_pred_val = best_model_instance.predict(X_val)

        # calculate and save evaluation score
        val_f_beta = fbeta_score(y_val, y_pred_val, beta=beta_value, pos_label=1, zero_division=0)
        val_report_dict = classification_report(y_val, y_pred_val, target_names=["Healthy", "Sick"], output_dict=True, zero_division=0)
        val_confusion_matrix = confusion_matrix(y_val, y_pred_val).tolist()

        validation_set_evaluations[model_name] = {
            'best_params': grid_search.best_params_,
            'val_fbeta': val_f_beta,
            'val_recall_sick': val_report_dict["Sick"]["recall"],
            'val_precision_sick': val_report_dict["Sick"]["precision"],
            'val_confusion_matrix': val_confusion_matrix,
            'train_fbeta': train_f_beta
        }

        print("All models tuned and evaluated on validation set")

    return validation_set_evaluations
    
def ensemble(X_train, y_train, X_val, y_val, X_test, y_test, validation_set_eval):
    # ini chosen base model with best hyperparameters
    log_reg_best_params = validation_set_eval['Logistic regression']['best_params']
    rng_forest_best_params = validation_set_eval['Random forest']['best_params']
    knn_best_params = validation_set_eval['KNN']['best_params']

    # ini votingclassifier
    classifier1 = LogisticRegression(random_state=42, max_iter=3000, **log_reg_best_params)
    classifier2 = RandomForestClassifier(random_state=42, **rng_forest_best_params)
    classifier3 = KNeighborsClassifier(**knn_best_params)

    estimators = [
        ('lr', classifier1),
        ('rf', classifier2),
        ('knn', classifier3)
    ]

    voting_clf = VotingClassifier(
    estimators=estimators,
    voting='hard'
    )

    # train VotingClassifier using train + val
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True) # ignore index since merging diff split sets

    start_time = time.time()
    print("\nTraining VotingClassifier...")
    voting_clf.fit(X_train_val, y_train_val)
    fit_time = time.time() - start_time
    print(f"VotingClassifier training complete. Took {fit_time} seconds")

    # eval on test set
    test_set_eval = {}

    y_pred = voting_clf.predict(X_test)

    cm_test = confusion_matrix(y_test, y_pred)
    test_set_eval["confusion_matrix"] = cm_test

    clf_report = classification_report(y_test, y_pred, target_names=['Healthy', 'Sick'], zero_division=0)
    test_set_eval["classification_report"] = clf_report

    # F-beta score (F-3)
    beta_value = 3
    f_beta_score = fbeta_score(y_test, y_pred, beta=beta_value, pos_label=1, zero_division=0)
    test_set_eval["F_beta"] = f_beta_score

    return voting_clf, test_set_eval 

def tuning_score_summary(val_set_eval, dataset_name):
    summary_data = []
    for model_name, metrics_dict in val_set_eval.items():
        f3_score = metrics_dict["val_fbeta"]
        f3_score_train = metrics_dict["train_fbeta"]

        recall_sick = metrics_dict["val_recall_sick"]
        precision_sick = metrics_dict["val_precision_sick"]
        best_params = metrics_dict["best_params"]

        summary_data.append({
            "Model": model_name,
            "F3-Score": f3_score,
            "F3-Score-Train": f3_score_train,
            "Recall score (sick)": recall_sick,
            "Precision score (sick)": precision_sick,
            "Best parameters": str(best_params)
        })

    summary_df = pd.DataFrame(summary_data)
    mean_f3_score = summary_df["F3-Score"].mean()
    mean_f3_score_train = summary_df["F3-Score-Train"].mean()

    print(f"--{dataset_name} dataset score summary--")
    print(f"Mean f3-score: {mean_f3_score:.4f}")
    print(f"Mean f3-score-train: {mean_f3_score_train:.4f}")
    print(summary_df)
        
    return None

def final_result(test_set_eval):
    print("--Final ensemble model scores--\n")

    f3_score_test = test_set_eval["F_beta"]
    print(f"F3-score: {f3_score_test:.4f}")

    print(f"\nClassifiction Report")

    clf_report = test_set_eval["classification_report"]
    print(clf_report)

    print()
    visualization.confusion_matrix(test_set_eval)
    return None