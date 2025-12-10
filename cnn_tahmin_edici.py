import numpy as np
import librosa
import tensorflow.keras as keras
import math
import os
import pandas as pd # CSV kaydı için gerekli

# --- AYARLAR ---
MODEL_PATH = "cnn_model.h5"
MAPPING_PATH = "mapping.npy"
TEST_KLASORU = "örnekler"
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3 
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION

def process_input_file(file_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """
    Şarkının tamamını işler, normalize eder ve parçalara böler.
    """
    try:
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=None)
        signal = librosa.util.normalize(signal) # Normalizasyon
        
        track_duration = len(signal) / SAMPLE_RATE
        num_segments = int(track_duration / SEGMENT_DURATION)
        num_mfcc_vectors_per_segment = math.ceil(SAMPLES_PER_SEGMENT / hop_length)
        
        X = []
        
        for d in range(num_segments):
            start = SAMPLES_PER_SEGMENT * d
            finish = start + SAMPLES_PER_SEGMENT
            
            if finish > len(signal):
                break

            segment = signal[start:finish]
            mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T
            
            if len(mfcc) == num_mfcc_vectors_per_segment:
                X.append(mfcc.tolist())
                
        return np.array(X)
    except Exception as e:
        print(f"Hata: {e}")
        return np.array([])

def predict_top3(file_path, model, mapping):
    """
    Tahmin yapar ve en çok oy alan ilk 3 türü döndürür.
    """
    X = process_input_file(file_path)
    
    if len(X) == 0:
        return []

    X = X[..., np.newaxis]
    
    # Tahminler
    predictions = model.predict(X, verbose=0) 
    predicted_indices = np.argmax(predictions, axis=1)
    
    # Oyları Say (minlength önemli, tüm türleri kapsasın diye)
    counts = np.bincount(predicted_indices, minlength=len(mapping))
    
    # Oyları Çoktan Aza Sırala (ArgSort küçükten büyüğe verir, biz ters çeviriyoruz [::-1])
    sorted_indices = np.argsort(counts)[::-1]
    
    # İlk 3'ü al
    top3_indices = sorted_indices[:3]
    total_votes = len(predicted_indices)
    
    results = []
    for idx in top3_indices:
        genre = mapping[idx]
        votes = counts[idx]
        percentage = (votes / total_votes) * 100
        results.append((genre, percentage))
        
    return results

# --- ANA PROGRAM ---
if __name__ == "__main__":
    
    if os.path.exists(TEST_KLASORU):
        print("Sistem başlatılıyor...")
        model = keras.models.load_model(MODEL_PATH)
        mapping = np.load(MAPPING_PATH)
        print("Model hazır! Analiz başlıyor...\n")

        dosyalar = [f for f in os.listdir(TEST_KLASORU) if f.endswith(('.wav', '.mp3', '.au', '.m4a'))]
        
        # Excel/CSV için verileri tutacağımız liste
        excel_verisi = []
        
        if not dosyalar:
            print("Klasörde ses dosyası yok.")
        else:
            # Ekrana yazdırma formatı
            print(f"{'DOSYA ADI':<30} | {'1. TAHMİN':<15} | {'2. TAHMİN':<15} | {'3. TAHMİN':<15}")
            print("-" * 85)

            for dosya in dosyalar:
                tam_yol = os.path.join(TEST_KLASORU, dosya)
                try:
                    top3 = predict_top3(tam_yol, model, mapping)
                    
                    if top3:
                        # Sonuçları Formatla
                        t1 = f"{top3[0][0].upper()} (%{top3[0][1]:.0f})"
                        t2 = f"{top3[1][0].upper()} (%{top3[1][1]:.0f})"
                        t3 = f"{top3[2][0].upper()} (%{top3[2][1]:.0f})"
                        
                        print(f"{dosya[:30]:<30} | {t1:<15} | {t2:<15} | {t3:<15}")
                        
                        # Listeye Kaydet (Excel için)
                        excel_verisi.append({
                            "Dosya Adı": dosya,
                            "Tahmin 1": top3[0][0].upper(),
                            "Oran 1 (%)": round(top3[0][1], 2),
                            "Tahmin 2": top3[1][0].upper(),
                            "Oran 2 (%)": round(top3[1][1], 2),
                            "Tahmin 3": top3[2][0].upper(),
                            "Oran 3 (%)": round(top3[2][1], 2)
                        })
                        
                except Exception as e:
                    print(f"{dosya:<30} | HATA: {e}")
            
            print("-" * 85)
            
            # --- CSV/EXCEL OLUŞTURMA ---
            if excel_verisi:
                df = pd.DataFrame(excel_verisi)
                cikti_adi = "ornekler_tahmin_sonuclari.csv"
                df.to_csv(cikti_adi, index=False, encoding='utf-8-sig') # utf-8-sig Türkçe karakter sorunu olmasın diye
                print(f"\n✅ Rapor başarıyla kaydedildi: {cikti_adi}")
                print("Bu dosyayı Excel ile açabilirsiniz.")
            
    else:
        print("Klasör bulunamadı.")