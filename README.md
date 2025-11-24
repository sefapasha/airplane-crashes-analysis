# ✈️ Airplane Crash Analysis

![Python](https://img.shields.io/badge/Python-3.13-blue)
![pandas](https://img.shields.io/badge/pandas-2.1.3-orange)
![matplotlib](https://img.shields.io/badge/matplotlib-3.8.2-red)
![License](https://img.shields.io/badge/License-CC0-green)

Bu proje, geçmiş uçak kazası verilerini kullanarak bir kazanın ölümcül olup olmadığını (yani herhangi bir can kaybı olup olmadığını) tahmin etmek için makine öğrenimi modelleri geliştirmeyi ve karşılaştırmayı amaçlamaktadır. Proje kapsamında veri temizleme, özellik mühendisliği, çeşitli sınıflandırma algoritmalarının eğitimi, performans değerlendirmesi, çapraz doğrulama ve hiperparametre optimizasyonu adımları izlenmiştir.

---

## 📊 Proje Özeti

- **Amaç:** Uçak kazalarının analizi, ölüm sayıları ve kazaların sebepleri üzerinden trendlerin incelenmesi.
- **Veri Kaynağı:** Kaggle – [Airplane Crash and Fatalities Dataset](https://www.kaggle.com/datasets/themuneeb99/airplane-crash-and-fatalities-1948present)
- **Veri Boyutu:** 2.61 MB, CSV dosyası
- **Kapsam:** Ticari, askeri ve özel uçak kazaları
- **Özellikler:** Kaza tarihi, uçak türü, kayıt numarası, operatör, ölüm sayısı, kaza yeri, hasar ve sebep (varsa)

---

## 📊 Kullanılan Modeller

- **Logistic Regression (Lojistik Regresyon):** Doğrusal bir sınıflandırıcı.
- **K-Nearest Neighbors (KNN):** K-En Yakın Komşu, örnek tabanlı bir sınıflandırıcı.
- **Gaussian Naive Bayes (Gaussian Naif Bayes):** Bayes teoremine dayalı olasılıksal bir sınıflandırıcı.

---

## 🔍 Metodoloji

- Veri Yükleme ve Ön İşleme: Veri seti yüklenir ve ilk incelemeler yapılır.
- Eksik Değer Yönetimi: Özellikle fat. sütunundaki eksik değerler temizlenir veya doldurulur.
- Özellik Mühendisliği:
      - fat. sütunundan death_flag hedef değişkeni (ikili sınıflandırma) oluşturulur.
      - acc. date sütunundan kaza yılı (year) çıkarılır.
      - type, operator, dmg gibi kategorik özellikler kullanılır.
- Veri Bölme: Veri seti eğitim (%80) ve test (%20) kümelerine stratejik olarak (hedef dağılımını koruyarak) ayrılır.
- Ön İşleme Pipeline'ları: Kategorik özellikler için OneHotEncoder, sayısal özellikler için StandardScaler içeren ColumnTransformer ve Pipeline yapıları kurulur.
- Model Eğitimi ve Değerlendirme: Belirlenen her model (Logistic Regression, KNN, Gaussian Naive Bayes) eğitilir ve Accuracy, Precision, Recall, F1-Score, Confusion Matrix gibi metriklerle test seti    üzerinde değerlendirilir.
- Çapraz Doğrulama (Cross-Validation): Modellerin genellenebilirliğini değerlendirmek için 5 katlı Stratified K-Fold çapraz doğrulama uygulanır.
- Hiperparametre Optimizasyonu (GridSearch): KNN ve Logistic Regression modelleri için GridSearchCV kullanarak en iyi hiperparametreler bulunur.
- Görselleştirme: Model performanslarını (ROC eğrileri, Confusion Matrix'ler) karşılaştıran grafikler oluşturulur ve plots/ dizinine kaydedilir.
- Model Kaydetme: En iyi performans gösteren modeller (KNN, Logistic Regression) ve Naive Bayes modeli models/ dizinine kaydedilir.
- Sonuç Özeti: Tüm modellerin metriklerini içeren özet bir tablo oluşturulur ve model_summary.csv olarak kaydedilir.

---

## 📁 Proje Yapısı

```
# Proje Ana Dizini ├── aircraft_crash_data.csv
# Ham veri seti ├── model_summary.csv
# Model performans özet tablosu ├── plots/
# Oluşturulan grafikler (ROC, Confusion Matrix) │ ├── roc_comparison.png │ ├── confusion_Logistic_Regression.png │ ├── confusion_KNN_(k=5).png │ └── confusion_GaussianNB.png ├── models/
# Eğitilmiş modeller │ ├── best_knn.pkl │ ├── best_logistic.pkl │ └── gaussian_nb.pkl └── <notebook_adı>.ipynb
# Jupyter Notebook (bu analiz kodunu içerir)
```
---

## 🛠️ ## Kurulum ve Çalıştırma

Projeyi yerel olarak çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  **Depoyu Klonlayın:**

2.  **Gerekli Kütüphaneleri Yükleyin:**

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    ```

3.  **Jupyter Notebook'u Başlatın:**

    ```bash
    jupyter notebook
    ```

4.  `aircraft_crash_classification.ipynb` (veya benzer adla kaydedilen) dosyayı açın ve hücreleri sırasıyla çalıştırın.

## Bağımlılıklar

-   Python 3.x
-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`
-   `joblib`

---

# 🎯 Sonuçlar
Modellerin test seti performansları (F1 Skoru'na göre sıralanmıştır):

print(summary_df.to_markdown(index=False))
|   accuracy |   precision |   recall |       f1 |
|-----------:|------------:|---------:|---------:|
|   0.736368 |    0.623299 | 0.871781 | 0.726891 |
|   0.730509 |    0.656748 | 0.692049 | 0.673937 |
|   0.542361 |    0.464213 | 0.889698 | 0.610098 |

tabloya göre **Logistic Regression**, genel olarak en iyi F1 Skoru ve ROC AUC değerlerini göstererek en başarılı model olarak öne çıkmaktadır. KNN de benzer bir performans sergilerken, Gaussian Naive Bayes modelinin performansı diğerlerine göre daha düşüktür.


# 📸 Görseller
---
<img width="567" height="435" alt="download" src="https://github.com/user-attachments/assets/464a9cbc-853b-437d-af1e-302aaff14efc" />


# 🔍 Veri Seti Hakkında
```
aircraft_crash_data.csv dosyası, uçak kazalarına ilişkin çeşitli bilgileri içermektedir. Veri setindeki temel sütunlar şunlardır:

acc. date: Kaza tarihi
type: Uçak tipi
reg.: Kayıt numarası
operator: Operatör (hava yolu şirketi veya kurum)
fat.: Can kaybı sayısı
location: Kaza konumu
dmg: Hasar derecesi
```

⚠️ Bazı kazalarda zaman, rota veya sebep bilgisi eksik olabilir.

# 📈 Analiz Bulguları (Örnek)
1970’ler ve 1980’ler kazaların en yoğun olduğu yıllar

Ticari uçaklar arasında Boeing ve McDonnell Douglas modelleri daha çok kazaya karışmış

Operatör bazında ölüm sayısı en yüksek olanlar: American Airlines, Pan Am

Kazaların başlıca sebepleri: İnsan hatası, mekanik arıza, kötü hava koşulları

# 💻 Teknolojiler
```
Teknoloji	Versiyon	Kullanım Amacı
Python	3.13	Ana programlama dili
pandas	2.1.3	Veri manipülasyonu
numpy	1.24.3	Sayısal hesaplamalar
matplotlib	3.8.2	Grafikler ve görselleştirme
seaborn	0.13.0	İstatistiksel görselleştirme
scikit-learn	1.3.2	Basit ML veya trend tahminleri
JupyterLab	4.2.0	Notebook ortamı
```
# 🚀 Gelecek İyileştirmeler
Trend tahminleri için ML modelleri eklemek

Bölge ve hava durumu verileri ile veri zenginleştirme

Web tabanlı görselleştirme dashboard’u (Streamlit / Dash)

Otomatik güncelleme ve retraining pipeline’ı

# 👤 Geliştirici
Ahmet Sefa Ünal
