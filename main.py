from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
TRAIN_PATH = 'Fracture/train'
VALIDATE_PATH = 'Fracture/val'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20


def create_generators(train_path, validate_path, image_size, batch_size):
    """
    Create training and validation data generators.
    """
    train_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_generator = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_generator.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary"
    )
    val_gen = val_generator.flow_from_directory(
        validate_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary"
    )
    return train_gen, val_gen


def build_model(input_shape, freeze_layers=10):
    """
    Build the model using a pretrained base model (Xception).
    """
    base_model = keras.applications.Xception(
        weights="imagenet",
        input_shape=input_shape,
        include_top=False
    )
    base_model.trainable = False
    for layer in base_model.layers[-freeze_layers:]:
        layer.trainable = True

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model


def compile_and_train_model(model, train_gen, val_gen, epochs):
    """
    Compile and train the model.
    """
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.AUC()
        ]
    )
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen)
    return history


def evaluate_model(model, val_gen):
    """
    Evaluate the model and print metrics.
    """
    loss, accuracy, precision, recall, auc = model.evaluate(val_gen)
    print(f"Loss: {loss:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"AUC-ROC: {auc:.3f}")


if __name__ == "__main__":
    train_gen, val_gen = create_generators(TRAIN_PATH, VALIDATE_PATH, IMAGE_SIZE, BATCH_SIZE)
    model = build_model(input_shape=(224, 224, 3))
    history = compile_and_train_model(model, train_gen, val_gen, EPOCHS)
    evaluate_model(model, val_gen)
