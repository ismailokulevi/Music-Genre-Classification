import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# --- AYARLAR ---
DATA_PATH_X = "X.npy"
DATA_PATH_Y = "y.npy"
DATA_PATH_MAPPING = "mapping.npy"

def load_data(x_path, y_path):
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y

def prepare_datasets(test_size, validation_size):
    # Veriyi yükle
    X, y = load_data(DATA_PATH_X, DATA_PATH_Y)
    
    # Eğitim ve Test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # Eğitimden bir parça alıp Doğrulama (Validation) seti yap
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    # CNN için 3. boyutu (Kanal sayısı = 1) ekle
    # (Resimler siyah-beyaz olduğu için 1 kanal, renkli olsaydı 3 kanal olurdu)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    """
    CNN Modelini Oluşturur
    """
    model = keras.Sequential()

    # 1. Konvolüsyon Bloğu (Resimdeki özellikleri yakala)
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2. Konvolüsyon Bloğu
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3. Konvolüsyon Bloğu
    model.add(keras.layers.Conv2D(128, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Düzleştirme (Flatten) ve Çıkış
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3)) # Overfitting'i önle
    
    # Çıkış Katmanı (10 Tür için 10 Nöron)
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def plot_history(history):
    """
    Eğitim grafiğini çizer
    """
    fig, axs = plt.subplots(2)

    # Doğruluk (Accuracy) Grafiği
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Eval")

    # Hata (Loss) Grafiği
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Eval")
    
    plt.tight_layout()
    plt.savefig("egitim_grafigi.png") # Grafiği kaydet
    plt.show()

if __name__ == "__main__":
    # 1. Veri Setlerini Hazırla
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # 2. Modeli Kur
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # 3. Modeli Derle
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    # 4. Modeli Eğit
    print("\n--- Model Eğitimi Başlıyor ---")
    history = model.fit(X_train, y_train, 
                        validation_data=(X_validation, y_validation), 
                        batch_size=32, 
                        epochs=30) # İstersen 50 yapabilirsin

    # 5. Modeli Test Et
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest doğruluğu: %{test_acc*100:.2f}')

    # 6. Modeli Kaydet
    model.save("cnn_model.h5")
    print("Model kaydedildi: cnn_model.h5")
    
    # 7. Grafiği Çiz
    plot_history(history)