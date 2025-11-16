import pandas as pd
import numpy as np
# Scikit-learn (Klasik Modeller için)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Keras / TensorFlow (Derin Öğrenme Modeli için)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
print("Tüm kütüphaneler başarıyla import edildi.\n")

df = pd.read_csv('Data_Genre.csv')

#print("DataFrame Sütun Adları:")
#print(df.columns)

print("Veri yüklendi. İlk 5 satır:")
print(df.head())

# Bu kod Adım 1'deki kodların devamıdır

# 1. Öznitelikleri (X) seç: 'genre' sütunu hariç tüm sütunlar
X = df.drop('Class', axis=1)

# 2. Etiketleri (y) seç: Sadece 'genre' sütunu
y_labels = df['Class']

print("Veri, X (Öznitelikler) ve y (Etiketler) olarak ayrıldı.\n")

# 3. LabelEncoder'ı oluştur ve etiketleri sayısala çevir
encoder = LabelEncoder()
y = encoder.fit_transform(y_labels)

# Bilgi: Hangi sayının hangi türe karşılık geldiğini görmek için
#print("Sınıf etiketleri ve sayısal karşılıkları:")
#print(list(encoder.classes_))
# Bu size ['blues', 'classical', 'country', ...] gibi bir liste verecektir

print("\nAdım 2 tamamlandı. Etiketler metinden sayısal formata çevrildi.\n")

# Bu kod Adım 2'nin devamıdır

# 4. Veriyi %80 eğitim, %20 test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,    # %20'si test olsun
                                                    random_state=42,  # Her çalıştırmada aynı sonucu almak için
                                                    stratify=y)       # Her müzik türünden eşit oranda böl

print("Veri, Eğitim (%80) ve Test (%20) olarak bölündü.")
print("Eğitim seti boyutu:", X_train.shape)
print("Test seti boyutu:", X_test.shape)

# 5. StandardScaler (Ölçekleyici) oluştur
scaler = StandardScaler()

# DİKKAT: Ölçekleyici SADECE EĞİTİM VERİSİ (X_train) üzerinde eğitilir!
# Bu, 'öğrenme' işlemidir.
X_train_scaled = scaler.fit_transform(X_train)

# Aynı ölçekleyici, TEST VERİSİ (X_test) üzerinde 'eğitilmeden' uygulanır!
# Bu, 'uygulama' işlemidir.
X_test_scaled = scaler.transform(X_test)

print("\nAdım 3 tamamlandı. Veri ölçeklendirildi.")
print("Ölçeklendirilmiş eğitim verisinden bir örnek:")
print(X_train_scaled[0]) 

# Bu kod Adım 3'ün devamıdır

print("\n--- Adım 4: Model 1 (KNN) Eğitimi Başlıyor ---")

# 1. KNN modelini oluştur (n_neighbors=5, yani 5 komşuya bak)
knn = KNeighborsClassifier(n_neighbors=5)

# 2. Modeli ölçeklenmiş eğitim verisiyle eğit
#    (y_train etiketleri hala 0, 1, 2... formatında, bu KNN için uygundur)
knn.fit(X_train_scaled, y_train)

print("Adım 4 tamamlandı. KNN Modeli başarıyla eğitildi.")

# Bu kod Adım 4'ün devamıdır

print("\n--- Adım 5: Model 2 (SVM) Eğitimi Başlıyor ---")

# 1. SVM modelini oluştur
#    (kernel='rbf' genellikle bu tür sınıflandırmalar için en iyi sonucu verir)
svm = SVC(kernel='rbf', C=1.0) 

# 2. Modeli ölçeklenmiş eğitim verisiyle eğit
svm.fit(X_train_scaled, y_train)

print("Adım 5 tamamlandı. SVM Modeli başarıyla eğitildi.")

# Bu kod Adım 5'in devamıdır

print("\n--- Adım 6: Derin Öğrenme Modeli için Etiket Hazırlığı ---")

# 1. Toplam kaç sınıfımız olduğunu bul (10 müzik türü)
#    (y, Adım 2'de oluşturduğumuz sayısal etiketlerdi)
num_classes = len(np.unique(y))
print(f"Toplam {num_classes} adet müzik türü (sınıf) bulundu.")

# 2. y_train ve y_test'i One-Hot formatına dönüştür
#    (Keras'ın to_categorical fonksiyonu bu işi yapar)
y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)

print("Etiketler 'One-Hot Encoding' formatına dönüştürüldü.")
print("Orijinal y_train ilk etiket:", y_train[0]) # Örn: 7
print("One-Hot y_train ilk etiket:", y_train_onehot[0]) # Örn: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print("Yeni 'one-hot' eğitim verisi boyutu:", y_train_onehot.shape)

# Bu kod Adım 6'nın devamıdır

print("\n--- Adım 7: Model 3 (Derin Öğrenme MLP) Tasarımı ve Eğitimi ---")

# 1. Modelin giriş (input) ve çıkış (output) boyutlarını belirle
#    X_train_scaled.shape[1] bize öznitelik sayısını verir (muhtemelen 60)
input_features = X_train_scaled.shape[1] 
#    num_classes zaten Adım 6'da 10 olarak bulunmuştu

# 2. Modelin mimarisini oluştur (Katmanları tasarla)
model_mlp = Sequential()

# Giriş Katmanı ve İlk Gizli Katman (128 nöronlu)
model_mlp.add(Dense(128, input_dim=input_features, activation='relu'))
model_mlp.add(Dropout(0.3)) # Aşırı öğrenmeyi (overfitting) engellemek için

# İkinci Gizli Katman (64 nöronlu)
model_mlp.add(Dense(64, activation='relu'))
model_mlp.add(Dropout(0.3))

# Çıkış Katmanı (Sınıf sayısı (10) kadar nöron ve 'softmax')
model_mlp.add(Dense(num_classes, activation='softmax'))

# 3. Modele nasıl öğreneceğini söyle (Compile)
model_mlp.compile(loss='categorical_crossentropy', # one-hot için bu kayıp fonksiyonu şart
                  optimizer='adam',                # en popüler optimize edici
                  metrics=['accuracy'])            # başarıyı 'accuracy' ile ölç

# Cemil'e raporlaması için modelin özetini yazdır
print("Cemil'e raporlanacak model mimarisi:")
model_mlp.summary()

# 4. Modeli eğit (Bu işlem biraz zaman alabilir!)
print("\nModel eğitimi başlıyor...")
# 'history' değişkeni, eğitimin kaydını tutar
history = model_mlp.fit(X_train_scaled, y_train_onehot,
                        epochs=50,          # Veri setini 50 kez turlasın
                        batch_size=32,      # Her seferinde 32 şarkıya baksın
                        validation_split=0.1, # Eğitimin %10'unu anlık test için ayır
                        verbose=1)          # Eğitim sürecini ekrana yazdır

print("\nAdım 7 tamamlandı. Derin Öğrenme Modeli başarıyla eğitildi.")

# Bu kod Adım 7'nin devamıdır

print("\n--- Adım 8: Tahminleri Oluşturma ve Dışa Aktarma ---")

# 1. Klasik modellerden (KNN, SVM) tahminleri al
#    Bu modeller 0, 1, 2... gibi tekil sayılar döndürür
preds_knn = knn.predict(X_test_scaled)
preds_svm = svm.predict(X_test_scaled)
print("KNN ve SVM tahminleri alındı.")

# 2. Derin Öğrenme (MLP) modelinden tahminleri al
#    Bu model [0.1, 0.0, 0.9...] gibi olasılıklar döndürür
preds_mlp_raw = model_mlp.predict(X_test_scaled)
#    En yüksek olasılığa sahip index'i (sınıfı) seçmemiz gerek
preds_mlp = np.argmax(preds_mlp_raw, axis=1)
print("Derin Öğrenme (MLP) tahminleri alındı.")

# 3. Tahminleri ve Gerçek Etiketleri CSV'ye dökmek için hazırlama
#    Yunus'un işini kolaylaştırmak için sayısal etiketleri (0, 1, 2...)
#    tekrar metin etiketlerine ('blues', 'classical'...) dönüştürelim.
#    'encoder' nesnesini Adım 2'de oluşturmuştuk.

# 'y_test' (gerçek etiketler) hala sayısal (0,1,2...), metne çevir:
y_test_metin = encoder.inverse_transform(y_test)

# Tahminleri (preds_*) metne çevir:
preds_knn_metin = encoder.inverse_transform(preds_knn)
preds_svm_metin = encoder.inverse_transform(preds_svm)
preds_mlp_metin = encoder.inverse_transform(preds_mlp)

# 4. Tüm sonuçları tek bir DataFrame'e koy
results_df = pd.DataFrame({
    'Gercek_Etiketler': y_test_metin,
    'KNN_Tahminleri': preds_knn_metin,
    'SVM_Tahminleri': preds_svm_metin,
    'MLP_Tahminleri': preds_mlp_metin
})

# 5. DataFrame'i CSV dosyasına kaydet
results_df.to_csv('tahmin_sonuclari.csv', index=False)

print("\n--- TEKNİK AŞAMA TAMAMLANDI ---")
print("Yunus'un Analizi için 'tahmin_sonuclari.csv' dosyası başarıyla oluşturuldu.")
print("Dosyadan ilk 5 satır:")
print(results_df.head())