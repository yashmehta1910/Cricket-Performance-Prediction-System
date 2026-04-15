import pandas as pd
import numpy as np
import joblib, os
from sklearn.preprocessing        import StandardScaler
from sklearn.decomposition        import PCA
from sklearn.pipeline             import Pipeline
from sklearn.model_selection      import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble             import RandomForestClassifier, VotingClassifier
from sklearn.linear_model         import LogisticRegression
from sklearn.neighbors            import KNeighborsClassifier
from sklearn.svm                  import SVC
from sklearn.tree                 import DecisionTreeClassifier
from sklearn.metrics              import (accuracy_score, f1_score,
                                          jaccard_score, log_loss,
                                          classification_report)

# ── Load features ──────────────────────────────────────────────────
df = pd.read_csv("data/features.csv").dropna()

FEATURES = ['avg_runs','strike_rate','boundary_rate',
            'dot_ball_rate','matches_played','total_fours','total_sixes']

X = df[FEATURES].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {len(X_train)}  Test: {len(X_test)}")
print(f"Classes: {np.unique(y, return_counts=True)}")

# ── Define all models with preprocessing pipeline ─────────────────
models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "KNN": Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    KNeighborsClassifier(n_neighbors=5))
    ]),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    SVC(probability=True, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
}

# ── Train & evaluate all models ────────────────────────────────────
print("\n" + "="*65)
print(f"{'Model':<22} {'Acc':>6} {'F1':>6} {'Jaccard':>8} {'LogLoss':>8} {'CV':>6}")
print("="*65)

results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)

    acc     = accuracy_score(y_test, preds)
    f1      = f1_score(y_test, preds, average='weighted')
    jacc    = jaccard_score(y_test, preds, average='weighted')
    logloss = log_loss(y_test, proba)
    cv      = cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

    results[name] = {
        'model':   pipe,
        'acc':     acc,
        'f1':      f1,
        'jaccard': jacc,
        'logloss': logloss,
        'cv':      cv
    }
    print(f"{name:<22} {acc:>6.3f} {f1:>6.3f} {jacc:>8.3f} {logloss:>8.3f} {cv:>6.3f}")

print("="*65)

# ── Pick best model by CV score ────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['cv'])
best_pipe  = results[best_name]['model']
print(f"\n🏆 Best model: {best_name}  (CV Accuracy: {results[best_name]['cv']:.3f})")

# ── Voting ensemble — combine all models ───────────────────────────
print("\n⚙️  Training Voting Ensemble (all 5 models)...")
voting = VotingClassifier(
    estimators=[(n, Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    m.named_steps['clf'])
    ])) for n, m in models.items()],
    voting='soft'
)
voting.fit(X_train, y_train)
v_preds = voting.predict(X_test)
v_acc   = accuracy_score(y_test, v_preds)
v_cv    = cross_val_score(voting, X, y, cv=5, scoring='accuracy').mean()
print(f"   Voting Ensemble — Acc: {v_acc:.3f}   CV: {v_cv:.3f}")

# Use ensemble if it beats best individual model
final_model = voting if v_cv >= results[best_name]['cv'] else best_pipe
final_name  = "Voting Ensemble" if v_cv >= results[best_name]['cv'] else best_name
print(f"\n✅ Final model saved: {final_name}")

# ── Save everything ────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(final_model, "model/model.pkl")
joblib.dump(FEATURES,    "model/features.pkl")
joblib.dump(results,     "model/results.pkl")   # for app to display

print("\n📊 Classification Report (best model):")
unique_classes = sorted(set(y_test))
class_names = ['Poor', 'Average', 'Good', 'Excellent']
target_names = [class_names[i] for i in unique_classes]
print(classification_report(y_test, best_pipe.predict(X_test),
      target_names=target_names))
print("\n✅ All files saved to model/")