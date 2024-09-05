import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

image_size = (180, 180)
batch_size = 16

def train_and_save_model(dataset_dir):
    image_size = (180, 180)
    batch_size = 16

    train_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=1,
        image_size=image_size,
        batch_size=batch_size
    )

    val_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=1,
        image_size=image_size,
        batch_size=batch_size
    )

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
    )

    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    def make_model(input_shape, num_classes):
        inputs = keras.Input(shape=input_shape)
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x

        for size in [256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
            x = layers.add([x, residual])
            previous_block_activation = x

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        return keras.Model(inputs, outputs)

    model = make_model(input_shape=image_size + (3,), num_classes=2)
    epochs = 10

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    model.save("ObjectsModel", save_format="h5")

    # Выбираем случайное изображение из папки object1
    object1_dir = os.path.join(dataset_dir, 'object1')
    random_image = random.choice(os.listdir(object1_dir))
    object_predict = os.path.join(object1_dir, random_image)

    img = keras.utils.load_img(
        object_predict, target_size=image_size
    )
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Создание оси batch

    predictions = model.predict(img_array)
    score = float(predictions[0])
    print(f"This image is {100 * (1 - score):.2f}% object1 and {100 * score:.2f}% object2.")
