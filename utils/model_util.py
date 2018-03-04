from keras.metrics import top_k_categorical_accuracy
from keras.models import Model, Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam

n_classes = 80
input_channels = 3

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def load_vgg16(input_size):
    base_model = VGG16(include_top=False, input_shape=(input_size, input_size, input_channels), weights="imagenet")
    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_size,input_size,input_channels)))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(n_classes, activation="softmax"))
    for layer in base_model.layers:
        layer.trainable = False  # turn trainable flag to true when doing fine tuning
    return model

def load_vgg19(input_size):
    base_model = VGG19(include_top=False, weights="imagenet", input_shape=(input_size, input_size, input_channels))
    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_size,input_size,input_channels)))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(n_classes, activation="softmax"))
    for layer in base_model.layers:
        layer.trainable = False  # turn trainable flag to true when doing fine tuning
    model.summary()
    return model

def load_inceptionV3_model(input_size):
    base_model = InceptionV3(include_top=False, weights="imagenet",
                             input_shape=(input_size, input_size, input_channels))
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    for layer in base_model.layers:
        layer.trainable = False   # turn trainable flag to true when doing fine tuning
    # for layer in base_model.layers:
    #     layer.trainable = True
    print("training with inceptionV3")
    return model

def load_inception_resnetv2(input_size):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_size, input_size, input_channels)))
    base_model = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(input_size, input_size, input_channels))
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation="softmax"))

    # change the trainable to True when doing fine tuning
    # for layer in base_model.layers:
    #     layer.trainable = True

    for layer in base_model.layers:
        layer.trainable = False  # turn trainable flag to true when doing fine tuning
    model.summary()
    print("training with inception_resnetv2")
    return model

def load_resnet50(input_size):
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(input_size,input_size,input_channels))
    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_size, input_size, input_channels)))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(80, activation="softmax"))

    # when doing fine tuning, change the trainable to True
    # for layer in base_model.layers:
    #     layer.trainable = True

    for layer in base_model.layers:
        layer.trainable = False  # turn trainable flag to true when doing fine tuning
    model.summary()
    return model

def load_resnet50_with_dropout(input_size):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_size, input_size, input_channels)))
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(input_size, input_size, input_channels))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(2048, activation="relu"))
    model.add(Dropout(0.5))  # add a drop out layer
    model.add(Dense(80,activation="softmax"))
    for layer in base_model.layers:
        layer.trainable = False  # turn trainable flag to true when doing fine tuning
    model.summary()
    print("training with resnet50 with dropout")
    return model

def train_model_imagedatagen(model, checkpoint_file, train_gen, len_train, len_valid, epochs,
                             batch_size, valid_gen, optimizer=Adam(lr=1e-4), initial_epoch=0):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, verbose=1, min_delta=1e-4),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2, cooldown=2, verbose=1),
        ModelCheckpoint(filepath=checkpoint_file, save_best_only=True, save_weights_only=True)
    ]
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy", top_3_accuracy])
    model.fit_generator(generator = train_gen, steps_per_epoch =((len_train - 1) // batch_size) + 1, epochs=epochs,
                        callbacks=callbacks, validation_data=valid_gen,
                        validation_steps=((len_valid - 1) // batch_size) + 1, verbose=1, initial_epoch=initial_epoch)

    return model