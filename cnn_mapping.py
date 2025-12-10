import os
import librosa
import numpy as np
import math

# --- AYARLAR ---
DATASET_PATH = "genres"  # Klasör adı
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # Saniye cinsinden
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

# Amacımız şarkıları 3 saniyelik parçalara bölmek
num_segments = 10 

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """
    Ses dosyalarından MFCC (Spektrogram benzeri özellik) çıkarır ve kaydeder.
    """
    # Verileri saklayacağımız sözlük
    data = {
        "mapping": [],  # Tür isimleri (Rock, Jazz...)
        "labels": [],   # Etiketler (0, 1, 2...)
        "mfcc": []      # Çıkarılan öznitelikler (Resimler)
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    print("Veri işleme başlıyor... Bu işlem 5-10 dakika sürebilir.")

    # Tüm tür klasörlerini gez (rock, blues...)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Kök klasörde değilsek işlem yap
        if dirpath is not dataset_path:

            # Tür ismini kaydet (klasör adından)
            semantic_label = dirpath.split("\\")[-1] # Windows için "\\"
            # Eğer Mac/Linux kullanıyorsanız "/" yapın
            data["mapping"].append(semantic_label)
            print(f"\nİşleniyor: {semantic_label}")

            # Klasördeki tüm ses dosyalarını gez
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                try:
                    # Sesi yükle
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    # Şarkıyı 10 parçaya böl ve işle
                    for d in range(num_segments):
                        
                        # Parçanın başlangıç ve bitiş noktalarını hesapla
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        # Sadece o 3 saniyelik kısmı al ve MFCC çıkar
                        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                        mfcc = mfcc.T

                        # Boyut kontrolü yap (bazen 1 eksik/fazla olabiliyor)
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            # i-1 çünkü os.walk ilk turda ana klasörü sayar

                except Exception as e:
                    print(f"Hata dosya: {file_path}")

    # Veriyi Numpy formatında kaydet
    print("\nVeriler X.npy ve y.npy olarak kaydediliyor...")
    
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    
    # Dosyaları diske yaz
    np.save("X.npy", X)
    np.save("y.npy", y)
    
    # Tür isimlerini de ayrıca kaydedelim
    np.save("mapping.npy", np.array(data["mapping"]))
    
    print("BAŞARILI! Veri hazırlığı tamamlandı.")
    print(f"Toplam Eğitim Verisi Sayısı: {len(X)}") # 10.000 civarı olmalı

# --- KODU ÇALIŞTIR ---
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=num_segments)