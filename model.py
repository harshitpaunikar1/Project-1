"""
X Education lead prioritization model.
Identifies the most potential leads using logistic regression and ensemble methods.
Assigns lead scores and priority segments for the sales team.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        roc_auc_score, classification_report, precision_recall_curve,
        average_precision_score, f1_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class LeadFeatureEngineer:
    """Derives engagement and behavioral features for lead scoring."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "total_time_on_website" in df.columns and "total_visits" in df.columns:
            df["avg_time_per_visit"] = (
                df["total_time_on_website"] / df["total_visits"].replace(0, 1)
            )
        if "page_views_per_visit" in df.columns:
            df["high_engagement"] = (df["page_views_per_visit"] >= 3).astype(int)
        if "total_visits" in df.columns:
            df["visit_frequency_band"] = pd.cut(
                df["total_visits"],
                bins=[0, 2, 5, 10, float("inf")],
                labels=["low", "medium", "high", "very_high"],
            ).astype(str)
        if "last_activity" in df.columns:
            activity_map = {
                "email_opened": 1, "email_bounced": -1,
                "page_visited": 2, "form_submitted": 4,
                "sms_sent": 0, "olark_chat": 3,
                "converted_to_lead": 5, "had_phone_conversation": 4,
            }
            df["activity_score"] = df["last_activity"].str.lower().map(activity_map).fillna(0)
        if "lead_origin" in df.columns:
            origin_map = {"api": 2, "landing_page_submission": 3,
                          "lead_add_form": 1, "lead_import": 0, "quick_add_form": 1}
            df["origin_score"] = df["lead_origin"].str.lower().map(origin_map).fillna(1)
        if "email_opted_in" in df.columns:
            df["email_opted_in"] = df["email_opted_in"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(int)
        return df


class LeadPrioritizationModel:
    """
    Multi-model lead conversion predictor with AUC, precision-recall, and lead scoring.
    """

    def __init__(self, numeric_features: List[str], categorical_features: List[str],
                 target_col: str = "converted"):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.engineer = LeadFeatureEngineer()
        self.models: Dict[str, Pipeline] = {}
        self.results: List[Dict] = []
        self.best_model_name: Optional[str] = None
        self._optimal_threshold: float = 0.5
        self._feature_names: List[str] = []

    def _preprocessor(self):
        transformers = []
        if self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                                        sparse_output=False),
                                  self.categorical_features))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _estimators(self) -> Dict:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0,
                                                       class_weight="balanced", random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                     random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                             max_depth=4, random_state=42),
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=5,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, tree_method="hist", verbosity=0,
            )
        return models

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        df = self.engineer.transform(df)
        num_cols = [c for c in self.numeric_features if c in df.columns]
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        df_clean = df[num_cols + cat_cols + [self.target_col]].dropna(subset=[self.target_col])
        for col in num_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna("unknown")

        X = df_clean[num_cols + cat_cols]
        y = df_clean[self.target_col].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        prep = self._preprocessor()
        self.results = []
        for name, est in self._estimators().items():
            pipe = Pipeline([("preprocessor", prep), ("model", est)])
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_prob))
            ap = float(average_precision_score(y_test, y_prob))
            y_pred = (y_prob >= 0.5).astype(int)
            f1 = float(f1_score(y_test, y_pred))
            report = classification_report(y_test, y_pred, output_dict=True)
            recall_1 = report.get("1", {}).get("recall", 0.0)
            self.models[name] = pipe
            self.results.append({
                "model": name,
                "auc": round(auc, 4),
                "avg_precision": round(ap, 4),
                "f1": round(f1, 4),
                "recall_converted": round(recall_1, 4),
            })

        results_df = pd.DataFrame(self.results).sort_values("auc", ascending=False).reset_index(drop=True)
        self.best_model_name = results_df.iloc[0]["model"]

        best_pipe = self.models[self.best_model_name]
        best_probs = best_pipe.predict_proba(X_test)[:, 1]
        self._optimal_threshold = self._find_optimal_threshold(y_test, best_probs)

        try:
            cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(cat_cols))
        except Exception:
            cat_names = []
        self._feature_names = num_cols + cat_names

        return results_df

    def _find_optimal_threshold(self, y_true: pd.Series, y_prob: np.ndarray) -> float:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
        if len(thresholds) > 0:
            return float(thresholds[int(np.argmax(f1_scores[:-1]))])
        return 0.5

    def score_leads(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.best_model_name not in self.models:
            raise RuntimeError("Call fit() first.")
        df = self.engineer.transform(df)
        num_cols = [c for c in self.numeric_features if c in df.columns]
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        probs = self.models[self.best_model_name].predict_proba(
            df[num_cols + cat_cols]
        )[:, 1]
        result = df[["lead_id"] if "lead_id" in df.columns else []].copy().reset_index(drop=True)
        result["lead_score"] = np.round(probs * 100, 1)
        result["conversion_probability"] = np.round(probs, 4)
        result["priority"] = pd.cut(
            probs,
            bins=[0, 0.25, 0.50, 0.75, 1.0],
            labels=["low", "medium", "high", "very_high"],
        ).astype(str)
        result["recommended_action"] = result["conversion_probability"].apply(
            lambda p: "immediate_outreach" if p >= 0.75
            else "nurture_high" if p >= 0.50
            else "nurture_low" if p >= 0.25
            else "passive_monitoring"
        )
        return result

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self.best_model_name not in self.models:
            return None
        pipe = self.models[self.best_model_name]
        est = pipe.named_steps["model"]
        if not hasattr(est, "feature_importances_"):
            if hasattr(est, "coef_"):
                imp = np.abs(est.coef_[0])
            else:
                return None
        else:
            imp = est.feature_importances_
        names = self._feature_names[:len(imp)]
        return pd.DataFrame({"feature": names, "importance": imp}).sort_values(
            "importance", ascending=False
        ).head(15).reset_index(drop=True)

    def priority_summary(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        return (
            scored_df.groupby("priority")
            .agg(count=("lead_score", "count"),
                 avg_score=("lead_score", "mean"),
                 median_score=("lead_score", "median"))
            .round(2)
            .reset_index()
            .sort_values("avg_score", ascending=False)
        )

    def conversion_likelihood_report(self, scored_df: pd.DataFrame,
                                      top_n: int = 20) -> pd.DataFrame:
        return (
            scored_df.sort_values("conversion_probability", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    np.random.seed(42)
    n = 3000

    sources = ["google", "direct_traffic", "olark_chat", "organic_search",
               "welingak_website", "reference"]
    activities = ["email_opened", "page_visited", "form_submitted", "sms_sent",
                   "olark_chat", "converted_to_lead", "had_phone_conversation"]
    origins = ["api", "landing_page_submission", "lead_add_form", "lead_import"]
    occupations = ["unemployed", "working_professional", "student", "business_owner", "other"]

    df = pd.DataFrame({
        "lead_id": [f"lead_{i:05d}" for i in range(n)],
        "total_visits": np.random.randint(0, 25, n).astype(float),
        "total_time_on_website": np.random.uniform(0, 2500, n),
        "page_views_per_visit": np.random.uniform(0, 10, n),
        "lead_source": np.random.choice(sources, n),
        "last_activity": np.random.choice(activities, n),
        "lead_origin": np.random.choice(origins, n),
        "current_occupation": np.random.choice(occupations, n),
        "email_opted_in": np.random.choice(["Yes", "No"], n),
    })
    engagement = df["total_visits"] * 0.15 + df["page_views_per_visit"] * 0.3 + df["total_time_on_website"] * 0.001
    df["converted"] = (engagement + np.random.normal(0, 0.5, n) > engagement.median()).astype(int)

    model = LeadPrioritizationModel(
        numeric_features=["total_visits", "total_time_on_website", "page_views_per_visit"],
        categorical_features=["lead_source", "last_activity", "lead_origin",
                                "current_occupation", "email_opted_in"],
    )

    results = model.fit(df)
    print("Model comparison:")
    print(results.to_string(index=False))
    print(f"\nBest model: {model.best_model_name}")
    print(f"Optimal threshold: {model._optimal_threshold:.3f}")

    scored = model.score_leads(df.head(20))
    print("\nTop 10 leads by conversion probability:")
    top = model.conversion_likelihood_report(scored, top_n=10)
    print(top[["lead_id", "lead_score", "priority", "recommended_action"]].to_string(index=False))

    print("\nPriority summary:")
    print(model.priority_summary(model.score_leads(df)).to_string(index=False))

    fi = model.feature_importance()
    if fi is not None:
        print("\nTop 5 features:")
        print(fi.head(5).to_string(index=False))
