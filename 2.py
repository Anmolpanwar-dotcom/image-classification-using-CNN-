import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from PIL import Image

source_dataset_path = r"C:\Users\hp\Downloads\archive (1)\PetImages"   # original dataset
clean_dataset_path  = r"C:\Users\hp\Downloads\cleaned_petimages"        # clean dataset yahan banega

class_names = ["Cat", "Dog"]

# =====================================================
# STEP 0 — CLEAN DATASET FOLDER BANAO
# =====================================================

os.makedirs(clean_dataset_path, exist_ok=True)

for class_name in class_names:
    os.makedirs(os.path.join(clean_dataset_path, class_name), exist_ok=True)

# =====================================================
# STEP 1 — ORIGINAL DATASET CHECK KARO
# =====================================================

for folder in os.listdir(source_dataset_path):
    folder_path = os.path.join(source_dataset_path, folder)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        print(f"{folder}/ -> {len(files)} images")

# =====================================================
# STEP 2 — CORRUPT / INVALID IMAGES HATAKE CLEAN DATASET BANAO
# =====================================================

print("\nCleaning dataset... please wait\n")

for class_name in class_names:
    source_class_dir = os.path.join(source_dataset_path, class_name)
    clean_class_dir  = os.path.join(clean_dataset_path, class_name)

    saved_count = 0
    skipped_count = 0

    for file_name in os.listdir(source_class_dir):
        source_file_path = os.path.join(source_class_dir, file_name)

        try:
            with Image.open(source_file_path) as img:
                img = img.convert("RGB")

                clean_file_name = f"{class_name.lower()}_{saved_count}.jpg"
                clean_file_path = os.path.join(clean_class_dir, clean_file_name)

                img.save(clean_file_path, format="JPEG")

                saved_count += 1

        except Exception:
            skipped_count += 1
            print(f"Skipped corrupt image: {source_file_path}")

    print(f"{class_name}: saved={saved_count}, skipped={skipped_count}")

print("\nClean dataset ready.\n")

for folder in os.listdir(clean_dataset_path):
    folder_path = os.path.join(clean_dataset_path, folder)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        print(f"{folder}/ -> {len(files)} clean images")

# =====================================================
# STEP 3 — DATA LOAD KARO
# =====================================================

train_data = keras.utils.image_dataset_from_directory(
    clean_dataset_path,
    image_size=(150, 150),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42,
    color_mode="rgb"
)

val_data = keras.utils.image_dataset_from_directory(
    clean_dataset_path,
    image_size=(150, 150),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=42,
    color_mode="rgb"
)

# =====================================================
# STEP 4 — CLASS NAMES DEKHO
# =====================================================

print("\nClasses:", train_data.class_names)

# =====================================================
# STEP 5 — SPEED BADHAO
# =====================================================

train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_data   = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# =====================================================
# STEP 6 — MODEL BANAO
# =====================================================

model = keras.Sequential([

    # --- Input layer ---
    keras.Input(shape=(150, 150, 3)),

    # --- Normalize karo: pixels 0-255 -> 0-1 ---
    keras.layers.Rescaling(1./255),

    # --- CONV LAYER 1 ---
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    # --- CONV LAYER 2 ---
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    # --- CONV LAYER 3 ---
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    # --- CLASSIFIER ---
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')

])

# =====================================================
# STEP 7 — MODEL KO READY KARO
# =====================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =====================================================
# STEP 8 — TRAIN KARO
# =====================================================

history = model.fit(
    train_data,
    epochs=5,
    validation_data=val_data
)

# =====================================================
# STEP 9 — RESULTS DEKHO
# =====================================================

acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss')
plt.legend()

plt.show()


# =====================================================
# STEP 10 — SINGLE IMAGE PREDICTION KARO
# =====================================================

test_image_path = r"C:\Users\hp\Downloads\cleaned_petimages\Dog\dog_12480.jpg"  # yahan apni test image ka path do

img = keras.utils.load_img(
    test_image_path,
    target_size=(150, 150)
)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)   # batch dimension add karo

prediction = model.predict(img_array)[0][0]

print(f"\nRaw prediction value: {prediction:.4f}")

if prediction > 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title("Test Image")
plt.axis("off")
plt.show()

# Model ko save karein
model.save("pet_model.h5")
print("Model Saved Successfully!")