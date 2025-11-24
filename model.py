"""
Açıklama:
- Hedef: `fat.` sütunundan "ölüm var mı?" (0 = yok, 1 = var) sınıflandırması
- Modeller: Logistic Regression, KNN (Classifier), Gaussian Naive Bayes
- Adımlar: EDA, Temizleme, Feature engineering, Pipeline, CV, GridSearch, Metikler, ROC, Confusion Matrix, Kaydetme
- Çıktılar: plots/ ve models/ klasörlerine kaydedilir
"""

# ============================================================
# 0) Kütüphaneler
# ============================================================
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ============================================================
# 1) Ayarlar / Klasörler
# ============================================================
DATA_PATH = "/content/aircraft_crash_data.csv"
PLOTS_DIR = "plots"
MODELS_DIR = "models"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42

sns.set(style="whitegrid", context="talk")

# ============================================================
# 2) Veri Yükleme ve Basit EDA
# ============================================================
print(">> Veri yükleniyor:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print(">> Veri boyutu:", df.shape)

print("\n>> İlk 5 satır:")
print(df.head().T)

print("\n>> Kolonlar ve eksik değer sayıları:")
print(df.isna().sum())

# Kolon isimleri veri setindeki haliyle olabilir: acc. date, type, reg., operator, fat., location, dmg

# ============================================================
# 3) Temizleme & Hedef Oluşturma
# ============================================================
# 3.1 fat. -> numeric ve hedef
df['fat.'] = pd.to_numeric(df['fat.'], errors='coerce')  # hatalı stringleri NaN yap
print(f"\nÖlüm sütununda (fat.) toplam null: {df['fat.'].isna().sum()}")

# Hedef: ölüm var mı?
df = df.dropna(subset=['fat.']).copy()
df['death_flag'] = (df['fat.'] > 0).astype(int)

print("\nDeath flag dağılımı:")
print(df['death_flag'].value_counts())

# 3.2 Tarihten yıl çıkartma (kısimsel), yüksekkardinalite kolonları temizleme
# Tarih kolonunun adı "acc. date" veya "acc_date" olabilir; esnek davran:
date_cols = [c for c in df.columns if 'acc' in c.lower() or 'date' in c.lower()]
if len(date_cols) > 0:
    date_col = date_cols[0]
    # Deneme: sonda yıl/yy olarak varsa al, yoksa NaN
    df['year'] = df[date_col].astype(str).str.extract(r'(\d{4}|\d{2})$')[0]
    # numeric çevir
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(-1).astype(int)
    print(f"\nTarih sütunu bulundu: {date_col}, örnek yıllar:\n", df['year'].unique()[:10])
else:
    df['year'] = -1

# 3.3 Reg. (kayıt) ve location gibi kolonlar genelde yüksek kardinalite: çıkarıyoruz
drop_candidates = []
for col in df.columns:
    if col.lower().strip() in ['reg.', 'reg', 'location', 'loc', 'acc. date', 'acc_date', 'acc date']:
        drop_candidates.append(col)
# ama year'ı çıkarmıyoruz
for c in drop_candidates:
    if c in df.columns and c != 'year':
        df = df.drop(columns=[c])

# 3.4 Eksik kategorik verileri 'UNKNOWN' ile doldur
cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in ['fat.',]]
for c in cat_cols:
    df[c] = df[c].fillna('UNKNOWN')

# Print edelim
print("\nTemizlenmiş dataframe shape:", df.shape)
print("Kullanılacak kolon örneği:", df.columns.tolist()[:15])

# ============================================================
# 4) Özellik Seçimi / Engineering
# ============================================================
possible_features = []
for c in ['type', 'operator', 'dmg', 'cause', 'year']:
    if c in df.columns:
        possible_features.append(c)

# Eğer dataset'te başka kategorik kolonlar kaldıysa, onları da dahil edebiliriz:
extra_cat = [c for c in df.columns if (df[c].dtype == 'object' and c not in possible_features and c not in ['fat.'])]
# Limitli sayıda ekstra al (çok yüksek kardinalite tehlikesi)
# ama kullanıcı isteğine göre bütün kategorikleri kullanabiliriz; burda güvenli seçim:
extra_cat = extra_cat[:3]  # en fazla 3 fazla
features = possible_features + extra_cat
features = [f for f in features if f in df.columns]

# Herhalde fat. ve death_flag'ı bırakıyoruz
print("\nSeçilen özellikler:", features)

X = df[features].copy()
y = df['death_flag'].copy()

# ============================================================
# 5) Eğitim / Test Ayrımı (Stratified)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

print("\nTrain/Test boyutları:", X_train.shape, X_test.shape)
print("Train hedef dağılımı:\n", y_train.value_counts(normalize=True))
print("Test hedef dağılımı:\n", y_test.value_counts(normalize=True))

# ============================================================
# 6) Ön İşleme Pipeline
# ============================================================
# Tüm seçilen özellikler kategorik veya sayısal (year)
categorical_features = [c for c in X_train.columns if X_train[c].dtype == 'object']
numeric_features = [c for c in X_train.columns if c not in categorical_features]

print("\nCategorical:", categorical_features)
print("Numeric:", numeric_features)

# OneHotEncoder for categorical (handle_unknown ignore)
# StandardScaler for numeric (year)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ],
    remainder='drop'
)

# ============================================================
# 7) MODEL HAZIRLIK - Pipeline'lar
# ============================================================
# Logistic Regression pipeline
log_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE))
])

# KNN pipeline
knn_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", KNeighborsClassifier())
])

# Naive Bayes pipeline
nb_clf = GaussianNB()

# ============================================================
# 8) FIT / PREDICT / METRİKLER için yardımcı fonksiyonlar
# ============================================================
def evaluate_classifier(pipe_or_model, X_tr, y_tr, X_ts, y_ts, is_nb=False, model_name="model"):
    """
    pipe_or_model: Pipeline (sklearn) veya doğrudan estimator
    is_nb: GaussianNB için True (dense ihtiyacı)
    """
    if is_nb:
        # preprocessor ile dense transform
        X_tr_prep = preprocessor.fit_transform(X_tr).toarray()
        X_ts_prep = preprocessor.transform(X_ts).toarray()
        # estimator
        model = pipe_or_model
        model.fit(X_tr_prep, y_tr)
        preds = model.predict(X_ts_prep)
        probs = model.predict_proba(X_ts_prep)[:, 1] if hasattr(model, 'predict_proba') else None
    else:
        pipe_or_model.fit(X_tr, y_tr)
        preds = pipe_or_model.predict(X_ts)
        probs = pipe_or_model.predict_proba(X_ts)[:, 1] if hasattr(pipe_or_model, 'predict_proba') else None

    acc = accuracy_score(y_ts, preds)
    prec = precision_score(y_ts, preds, zero_division=0)
    rec = recall_score(y_ts, preds, zero_division=0)
    f1 = f1_score(y_ts, preds, zero_division=0)
    cm = confusion_matrix(y_ts, preds)

    print(f"\n--- {model_name} METRİKLER ---")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_ts, preds, zero_division=0))

    return {
        'model': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'preds': preds,
        'probs': probs
    }

# ============================================================
# 9) MODELLERİ TEK TEK EĞİT VE DEĞERLENDİR
# ============================================================
results = []

# 9.1 Logistic Regression
print("\n>> Logistic Regression eğitiliyor...")
log_res = evaluate_classifier(log_pipe, X_train, y_train, X_test, y_test, is_nb=False, model_name="Logistic Regression")
results.append(log_res)

# 9.2 KNN - Basit (k=5) ile
print("\n>> KNN (k=5) eğitiliyor...")
knn_pipe.set_params(clf__n_neighbors=5)
knn_res = evaluate_classifier(knn_pipe, X_train, y_train, X_test, y_test, is_nb=False, model_name="KNN (k=5)")
results.append(knn_res)

# 9.3 Gaussian Naive Bayes
print("\n>> Gaussian Naive Bayes eğitiliyor (dense transform)...")
nb_res = evaluate_classifier(nb_clf, X_train, y_train, X_test, y_test, is_nb=True, model_name="GaussianNB")
results.append(nb_res)

# ============================================================
# 10) ROC Eğrileri ve AUC Karşılaştırması
# ============================================================
plt.figure(figsize=(10,8))
for res in results:
    probs = res['probs']
    if probs is None:
        # prob yoksa tahmini skor yerine decision_function benzeri yok -> atla
        continue
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{res['model']} (AUC={auc:.3f})")

plt.plot([0,1],[0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Death Flag Classification")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_comparison.png"))
plt.show()

# ============================================================
# 11) Confusion Matrix Görselleştirmeleri
# ============================================================
for res in results:
    cm = res['confusion_matrix']
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {res['model']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    fn = os.path.join(PLOTS_DIR, f"confusion_{res['model'].replace(' ', '_')}.png")
    plt.savefig(fn)
    plt.show()

# ============================================================
# 12) Cross-Validation Score Örnekleri (StratifiedKFold)
# ============================================================
print("\n>> Cross-validation (5-fold Stratified) örnekleri")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Logistic CV (pipeline)
log_cv_scores = cross_val_score(log_pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print("Logistic CV accuracy scores:", log_cv_scores, "mean:", log_cv_scores.mean())

# KNN CV
knn_pipe.set_params(clf__n_neighbors=5)
knn_cv_scores = cross_val_score(knn_pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print("KNN CV accuracy scores:", knn_cv_scores, "mean:", knn_cv_scores.mean())

# Naive Bayes CV -> dense transform required in loop
nb_scores = []
for train_idx, test_idx in cv.split(X, y):
    X_tr = X.iloc[train_idx]
    X_te = X.iloc[test_idx]
    y_tr = y.iloc[train_idx]
    y_te = y.iloc[test_idx]
    X_tr_p = preprocessor.fit_transform(X_tr).toarray()
    X_te_p = preprocessor.transform(X_te).toarray()
    nb_clf.fit(X_tr_p, y_tr)
    nb_scores.append(accuracy_score(y_te, nb_clf.predict(X_te_p)))
print("GaussianNB CV accuracy scores:", nb_scores, "mean:", np.mean(nb_scores))

# ============================================================
# 13) Hyperparameter Tuning (GridSearch) - Örnek: KNN ve Logistic
# ============================================================
print("\n>> GridSearch örnekleri")

# 13.1 KNN param grid
knn_param_grid = {
    'clf__n_neighbors': [3,5,7,9],
    'clf__weights': ['uniform','distance'],
    'clf__p': [1,2]  # p=1 manhattan, p=2 euclidean
}
knn_grid = GridSearchCV(knn_pipe, knn_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
knn_grid.fit(X_train, y_train)
print("En iyi KNN param:", knn_grid.best_params_, "Best CV score:", knn_grid.best_score_)

# 13.2 Logistic param grid (C regularization)
log_param_grid = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs']
}
log_grid = GridSearchCV(log_pipe, log_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
log_grid.fit(X_train, y_train)
print("En iyi Logistic param:", log_grid.best_params_, "Best CV score:", log_grid.best_score_)

# ============================================================
# 14) En İyi Modelleri Eğit ve Kaydet
# ============================================================
print("\n>> En iyi modeller eğitilip kaydediliyor...")

best_knn = knn_grid.best_estimator_
best_log = log_grid.best_estimator_
# NB zaten hazır (no grid)

# Fit final olarak tüm train seti üzerinde
best_knn.fit(X_train, y_train)
best_log.fit(X_train, y_train)
# NB için dense transform
X_train_nb = preprocessor.fit_transform(X_train).toarray()
nb_clf.fit(X_train_nb, y_train)

# Kaydet
joblib.dump(best_knn, os.path.join(MODELS_DIR, "best_knn.pkl"))
joblib.dump(best_log, os.path.join(MODELS_DIR, "best_logistic.pkl"))
joblib.dump(nb_clf, os.path.join(MODELS_DIR, "gaussian_nb.pkl"))

print("Modeller kaydedildi:", os.listdir(MODELS_DIR))

# ============================================================
# 15) Feature Importance / Coefficients (Logistic için)
# ============================================================
ohe = preprocessor.named_transformers_['cat']
if hasattr(ohe, 'get_feature_names_out'):
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
else:
    # sklearn eski sürümlerde farklı fonksiyon
    cat_feature_names = [f"{c}_{i}" for c in categorical_features for i in range(1)]

feature_names = cat_feature_names + numeric_features

if hasattr(best_log.named_steps['clf'], 'coef_'):
    coefs = best_log.named_steps['clf'].coef_[0]
    # Eğer sparse outputsa dönüştürelim
    try:
        # Get transformed shape
        trans = preprocessor.fit_transform(X_train)
        n_features = trans.shape[1]
        if len(coefs) != n_features:
            # bu durumda feature names uydur
            feature_names = [f"f{i}" for i in range(len(coefs))]
    except:
        feature_names = [f"f{i}" for i in range(len(coefs))]

    coef_df = pd.DataFrame({'feature': feature_names[:len(coefs)], 'coef': coefs})
    coef_df['abs'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs', ascending=False).head(30)
    plt.figure(figsize=(8,10))
    sns.barplot(x='coef', y='feature', data=coef_df)
    plt.title("Logistic Regression - Top 30 Coefficients (absolute)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "logistic_top_coeffs.png"))
    plt.show()
else:
    print("Logistic modelinde coef bilgisi yok.")

# ============================================================
# 16) Son Özet Tablo
# ============================================================
final_summary = []
for res in results:
    final_summary.append({
        'model': res['model'],
        'accuracy': res['accuracy'],
        'precision': res['precision'],
        'recall': res['recall'],
        'f1': res['f1']
    })

summary_df = pd.DataFrame(final_summary).set_index('model')
print("\nFinal summary:\n", summary_df)

summary_df.to_csv("model_summary.csv")
print("Summary CSV kaydedildi: model_summary.csv")

# ============================================================
# BİTİŞ MESAJI
# ============================================================