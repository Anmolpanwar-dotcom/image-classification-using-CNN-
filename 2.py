import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from PIL import Image

# =====================================================
# STEP 0 — PATHS SET KARO
# =====================================================
source_dataset_path = r"C:\Users\hp\Downloads\archive (1)\PetImages"
clean_dataset_path  = r"C:\Users\hp\Downloads\cleaned_petimages"
class_names = ["Cat", "Dog"]

# Folders banao
os.makedirs(clean_dataset_path, exist_ok=True)
for class_name in class_names:
    os.makedirs(os.path.join(clean_dataset_path, class_name), exist_ok=True)

# =====================================================
# STEP 1 — DATA CLEANING (Zaroori hai channels fix karne ke liye)
# =====================================================
print("\nCleaning dataset... Please wait...")

for class_name in class_names:
    source_class_dir = os.path.join(source_dataset_path, class_name)
    clean_class_dir  = os.path.join(clean_dataset_path, class_name)
    saved_count = 0

    for file_name in os.listdir(source_class_dir):
        source_file_path = os.path.join(source_class_dir, file_name)
        try:
            with Image.open(source_file_path) as img:
                img = img.convert("RGB") # 3-channels fix
                clean_file_name = f"{class_name.lower()}_{saved_count}.jpg"
                img.save(os.path.join(clean_class_dir, clean_file_name), "JPEG")
                saved_count += 1
        except:
            continue
    print(f"Finished {class_name}: Saved {saved_count} images.")

# =====================================================
# STEP 2 — DATA LOADING (With 80-20 Split)
# =====================================================
train_ds = keras.utils.image_dataset_from_directory(
    clean_dataset_path,
    image_size=(150, 150),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = keras.utils.image_dataset_from_directory(
    clean_dataset_path,
    image_size=(150, 150),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# Performance Tuning
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# =====================================================
# STEP 3 — ADVANCED CNN MODEL (With Augmentation)
# =====================================================
model = keras.Sequential([
    keras.Input(shape=(150, 150, 3)),
    
    # 1. Data Augmentation (Model ko ratta maarne se rokne ke liye)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    
    # 2. Rescaling (Normalization)
    layers.Rescaling(1./255),
    
    # 3. Convolutional Layers
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # 4. Dense Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Regularization
    layers.Dense(1, activation='sigmoid') # Binary Output
])

# =====================================================
# STEP 4 — COMPILATION & TRAINING
# =====================================================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel training starting...")
history = model.fit(
    train_ds,
    epochs=15, # Epochs badha diye hain better accuracy ke liye
    validation_data=val_ds
)

# Model Save karo
model.save("pet_model.h5")
print("\nModel saved as pet_model.h5")

# =====================================================
# STEP 5 — TESTING (With Proper Normalization)
# =====================================================
def predict_image(img_path):
    img = keras.utils.load_img(img_path, target_size=(150, 150))
    img_array = keras.utils.img_to_array(img)
    
    # IMPORTANT: Normalization (Jaisa training mein tha)
    img_array = img_array / 255.0 
    img_array = tf.expand_dims(img_array, 0)
    
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    if prediction > 0.5:
        print(f"Result: DOG ({prediction*100:.2f}%)")
    else:
        print(f"Result: CAT ({(1-prediction)*100:.2f}%)")


# Model ko save karein
model.save("pet_model.h5")
print("Model Saved Successfully!")
