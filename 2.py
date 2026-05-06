import os

import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

SOURCE_DATASET_PATH = r"C:\Users\hp\Downloads\archive (1)\PetImages"
CLEAN_DATASET_PATH = r"C:\Users\hp\Downloads\cleaned_petimages"
CLASS_NAMES = ("Cat", "Dog")
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32


def clean_dataset():
    os.makedirs(CLEAN_DATASET_PATH, exist_ok=True)

    print("\nCleaning dataset... Please wait...")
    for class_name in CLASS_NAMES:
        source_class_dir = os.path.join(SOURCE_DATASET_PATH, class_name)
        clean_class_dir = os.path.join(CLEAN_DATASET_PATH, class_name)
        os.makedirs(clean_class_dir, exist_ok=True)

        saved_count = 0
        for file_name in os.listdir(source_class_dir):
            source_file_path = os.path.join(source_class_dir, file_name)
            try:
                with Image.open(source_file_path) as img:
                    img = img.convert("RGB")
                    clean_file_name = f"{class_name.lower()}_{saved_count}.jpg"
                    img.save(os.path.join(clean_class_dir, clean_file_name), "JPEG")
                    saved_count += 1
            except Exception:
                continue

        print(f"Finished {class_name}: Saved {saved_count} images.")


def load_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        CLEAN_DATASET_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=42,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        CLEAN_DATASET_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=42,
    )

    print("Class order:", train_ds.class_names)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, val_ds


def build_model():
    return keras.Sequential(
        [
            keras.Input(shape=(150, 150, 3)),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )


def predict_image(model, img_path):
    img = keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = keras.utils.img_to_array(img)

    # Model ke andar Rescaling layer already hai, isliye yahan /255 nahi karna.
    img_array = tf.expand_dims(img_array, 0)
    dog_probability = float(model.predict(img_array, verbose=0)[0][0])

    if dog_probability >= 0.5:
        print(f"Result: DOG ({dog_probability * 100:.2f}%)")
    else:
        print(f"Result: CAT ({(1.0 - dog_probability) * 100:.2f}%)")


def main():
    clean_dataset()
    train_ds, val_ds = load_datasets()

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("\nModel training starting...")
    model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
    )

    model.save("pet_model.h5")
    print("\nModel saved as pet_model.h5")


if __name__ == "__main__":
    main()

