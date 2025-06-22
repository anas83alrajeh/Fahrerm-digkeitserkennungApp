import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# === 1. Versteckte Dateien im Datenordner löschen ===
daten_verzeichnis = 'data'
for root, dirs, files in os.walk(daten_verzeichnis):
    for f in files:
        if f.startswith('.'):
            os.remove(os.path.join(root, f))

# === 2. Datenaugmentation und Datenvorbereitung ===
datengenerator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    fill_mode="nearest"
)

train_generator = datengenerator.flow_from_directory(
    daten_verzeichnis,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

val_generator = datengenerator.flow_from_directory(
    daten_verzeichnis,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

anzahl_klassen = len(train_generator.class_indices)
assert anzahl_klassen == 2, "Es sollten genau 2 Klassen sein: Drowsy und NonDrowsy"

# === 3. Modellaufbau Funktion mit variabler Lernrate ===
def build_model(learning_rate):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Gefroren zu Beginn

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(anzahl_klassen, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# === 4. Smart Callback zur Erkennung von Overfitting und Stoppen ===
class SmartEarlyStopping(Callback):
    def __init__(self, acc_diff_threshold=0.1, loss_patience=3):
        super().__init__()
        self.acc_diff_threshold = acc_diff_threshold
        self.loss_patience = loss_patience
        self.best_val_loss = float('inf')
        self.loss_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')

        diff = acc - val_acc if acc and val_acc else 0
        print(f"Epoche {epoch+1} | acc: {acc:.4f}, val_acc: {val_acc:.4f}, diff: {diff:.4f}, val_loss: {val_loss:.4f}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.loss_counter = 0
        else:
            self.loss_counter += 1

        # Stoppen, wenn Overfitting erkennbar und Verlust sich nicht verbessert
        if diff > self.acc_diff_threshold and self.loss_counter >= self.loss_patience:
            print("🛑 Training gestoppt: Overfitting erkannt und Validierungsverlust verbessert sich nicht mehr.")
            self.model.stop_training = True

# === 5. Fitness-Funktion für GWO: validierungsgenauigkeit nach kurzem Training ===
def fitness(learning_rate):
    # Baue Modell mit übergebenem learning_rate
    model = build_model(learning_rate)
    # Trainiere nur 2 Epochen, um schnell zu evaluieren
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=2,
        verbose=0
    )
    val_acc = history.history['val_accuracy'][-1]
    print(f"Test-Lernrate: {learning_rate:.8f} -> Val_Acc: {val_acc:.4f}")
    return val_acc

# === 6. Implementierung der Grey Wolf Optimizer (GWO) ===
class GreyWolfOptimizer:
    def __init__(self, obj_func, lb, ub, dim, population=5, iterations=10):
        self.obj_func = obj_func
        self.lb = lb  # Untere Schranke (z.B. 1e-6)
        self.ub = ub  # Obere Schranke (z.B. 1e-2)
        self.dim = dim
        self.population = population
        self.iterations = iterations

        # Initialisiere Positionen der "Wölfe" zufällig innerhalb der Grenzen
        self.positions = np.random.uniform(lb, ub, (population, dim))

        # Plätze für Alpha, Beta, Delta Wölfe
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = -np.inf
        self.beta_pos = np.zeros(dim)
        self.beta_score = -np.inf
        self.delta_pos = np.zeros(dim)
        self.delta_score = -np.inf

    def optimize(self):
        for iter in range(self.iterations):
            for i in range(self.population):
                score = self.obj_func(self.positions[i][0])

                # Update Alpha, Beta, Delta
                if score > self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()

                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()

                    self.alpha_score = score
                    self.alpha_pos = self.positions[i].copy()

                elif score > self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()

                    self.beta_score = score
                    self.beta_pos = self.positions[i].copy()

                elif score > self.delta_score:
                    self.delta_score = score
                    self.delta_pos = self.positions[i].copy()

            a = 2 - iter * (2 / self.iterations)  # Faktor, der mit der Iteration abnimmt

            for i in range(self.population):
                for d in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[d] - self.positions[i][d])
                    X1 = self.alpha_pos[d] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[d] - self.positions[i][d])
                    X2 = self.beta_pos[d] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[d] - self.positions[i][d])
                    X3 = self.delta_pos[d] - A3 * D_delta

                    new_pos = (X1 + X2 + X3) / 3

                    # Werte innerhalb der Schranken halten
                    self.positions[i][d] = np.clip(new_pos, self.lb, self.ub)

            print(f"Iteration {iter+1}/{self.iterations} - Beste Val_Acc: {self.alpha_score:.4f} mit Lernrate: {self.alpha_pos[0]:.8f}")

        return self.alpha_pos, self.alpha_score

# === 7. Starte Optimierung ===
print("Starte Grey Wolf Optimizer zur Lernratenoptimierung...")
gwo = GreyWolfOptimizer(obj_func=fitness, lb=1e-6, ub=1e-2, dim=1, population=5, iterations=10)
best_lr, best_score = gwo.optimize()

print(f"\nBeste gefundene Lernrate: {best_lr[0]:.8f} mit Validierungsgenauigkeit: {best_score:.4f}")

# === 8. Finales Training mit bester Lernrate und SmartEarlyStopping ===
final_model = build_model(best_lr[0])
smart_early_stop = SmartEarlyStopping(acc_diff_threshold=0.1, loss_patience=3)

final_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[smart_early_stop],
    verbose=1
)

# === 9. Modell speichern ===
final_model.save('trained_model.keras')
print("✅ Finales Modell gespeichert als 'trained_model.keras'")
