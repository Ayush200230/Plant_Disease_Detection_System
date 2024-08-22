'''
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_initial_model():
    # Data augmentation and rescaling
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        '../data/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        '../data/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks for early stopping and saving the best model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('../models/best_plant_disease_model.keras', save_best_only=True)

    # Model training
    model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Save the final model
    model.save('../models/plant_disease_model_final.keras')

if __name__ == "__main__":
    train_initial_model()




import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

class CustomImageDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size, target_size, datagen, num_classes):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.datagen = datagen
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        for file_path in batch_x:
            img = self.custom_load_img(file_path, self.target_size)
            if img is not None:
                img = self.datagen.random_transform(img)  # Apply transformations
                img = self.datagen.standardize(img)  # Rescale or normalize
                images.append(img)

        images = np.array(images)

        # Ensure that the lengths match
        if len(images) != len(batch_y):
            min_len = min(len(images), len(batch_y))
            images = images[:min_len]
            batch_y = batch_y[:min_len]

        # Convert labels to one-hot encoding
        batch_y = to_categorical(batch_y, num_classes=self.num_classes)

        return images, np.array(batch_y)

    def custom_load_img(self, path, target_size):
        try:
            img = image.load_img(path, target_size=target_size)
            return img_to_array(img)
        except (IOError, SyntaxError):
            print(f"Skipping corrupt image: {path}")
            return None  # Skip the corrupt image


def train_initial_model():
    # Define paths and labels
    train_dir = '../data/'
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Collecting file paths and labels
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=False
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Retrieve file paths and labels from the generators
    train_file_paths = train_generator.filepaths
    train_labels = train_generator.labels
    val_file_paths = validation_generator.filepaths
    val_labels = validation_generator.labels

    # Number of classes
    num_classes = train_generator.num_classes

    # Use custom generator
    train_custom_generator = CustomImageDataGenerator(train_file_paths, train_labels, 32, (128, 128), train_datagen, num_classes)
    val_custom_generator = CustomImageDataGenerator(val_file_paths, val_labels, 32, (128, 128), train_datagen, num_classes)

    # Model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Check if there's a checkpoint to load
    checkpoint_path = '../models/checkpoint.weights.h5'
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print("Loaded model from checkpoint")

    # Callbacks for early stopping and saving the best model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True)

    # Model training
    model.fit(
        train_custom_generator,
        epochs=50,
        validation_data=val_custom_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Save the final model
    model.save('../models/plant_disease_model_final.keras')

if __name__ == "__main__":
    train_initial_model()


# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# # Check for GPU availability
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print("GPU is available!")
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)  # Enable dynamic memory allocation
# else:
#     print("No GPU found. Training will use the CPU.")

# class CustomImageDataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, file_paths, labels, batch_size, image_size, datagen, num_classes, **kwargs):
#         super().__init__(**kwargs)  # Properly initialize the parent class
#         self.file_paths = file_paths
#         self.labels = labels
#         self.batch_size = batch_size
#         self.image_size = image_size
#         self.datagen = datagen
#         self.num_classes = num_classes
#         self.indexes = np.arange(len(self.file_paths))
#         self.on_epoch_end()

#     def __len__(self):
#         return int(np.ceil(len(self.file_paths) / self.batch_size))

#     def __getitem__(self, index):
#         batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#         batch_file_paths = [self.file_paths[i] for i in batch_indexes]
#         batch_labels = [self.labels[i] for i in batch_indexes]
        
#         images = []
#         labels = []
        
#         for file_path, label in zip(batch_file_paths, batch_labels):
#             try:
#                 img = self.load_and_preprocess_image(file_path)
#                 images.append(img)
#                 labels.append(label)
#             except Exception as e:
#                 print(f"Error processing image {file_path}: {e}")
        
#         if len(images) == 0:
#             return np.zeros((self.batch_size, *self.image_size, 3)), np.zeros((self.batch_size, self.num_classes))

#         images = np.array(images)
#         labels = np.array(labels)
#         labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)

#         return images, labels
    
#     def load_and_preprocess_image(self, file_path):
#         img = load_img(file_path, target_size=self.image_size)
#         img = img_to_array(img)
#         img = self.datagen.random_transform(img)
#         img = self.datagen.standardize(img)
#         return img

#     def on_epoch_end(self):
#         np.random.shuffle(self.indexes)

# def convert_to_tflite(model_path, tflite_model_path):
#     model = tf.keras.models.load_model(model_path)
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     tflite_model = converter.convert()
#     with open(tflite_model_path, 'wb') as f:
#         f.write(tflite_model)

# class EpochFileCallback(tf.keras.callbacks.Callback):
#     def __init__(self, epoch_file_path):
#         self.epoch_file_path = epoch_file_path

#     def on_epoch_end(self, epoch, logs=None):
#         with open(self.epoch_file_path, 'w') as file:
#             file.write(str(epoch + 1))

# def train_initial_model():
#     train_dir = 'C:/Users/ASUS/Downloads/potato-disease/training/PlantVillage'
#     # train_dir = 'C:/files/plant-disease-detection/data'
#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=30,
#         width_shift_range=0.3,
#         height_shift_range=0.3,
#         shear_range=0.3,
#         zoom_range=0.3,
#         horizontal_flip=True,
#         fill_mode='nearest',
#         validation_split=0.2
#     )

#     train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(128, 128),
#         batch_size=32,
#         class_mode='categorical',
#         subset='training',
#         shuffle=False
#     )

#     validation_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(128, 128),
#         batch_size=32,
#         class_mode='categorical',
#         subset='validation',
#         shuffle=False
#     )

#     train_file_paths = train_generator.filepaths
#     train_labels = train_generator.labels
#     val_file_paths = validation_generator.filepaths
#     val_labels = validation_generator.labels

#     num_classes = train_generator.num_classes

#     train_custom_generator = CustomImageDataGenerator(train_file_paths, train_labels, 32, (128, 128), train_datagen, num_classes)
#     val_custom_generator = CustomImageDataGenerator(val_file_paths, val_labels, 32, (128, 128), train_datagen, num_classes)

#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
#     model = Sequential([
#         base_model,
#         Flatten(),
#         Dense(512, activation='relu'),
#         Dropout(0.5),
#         Dense(num_classes, activation='softmax')
#     ])

#     checkpoint_path = 'C:/files/plant-disease-detection/models/checkpoint.weights.h5'
#     epoch_file_path = 'C:/files/plant-disease-detection/models/last_epoch.txt'
#     initial_epoch = 0

#     model.build(input_shape=(None, 128, 128, 3))

#     if os.path.exists(checkpoint_path):
#         try:
#             model.load_weights(checkpoint_path)
#             print("Loaded model weights from checkpoint")
#         except Exception as e:
#             print(f"Error loading weights: {e}")
#         if os.path.exists(epoch_file_path):
#             with open(epoch_file_path, 'r') as file:
#                 initial_epoch = int(file.read().strip())
#             print(f"Resuming training from epoch {initial_epoch}")
#         else:
#             print("Epoch file not found, starting from scratch.")
#     else:
#         print("Checkpoint not found, starting training from scratch.")
    
#     # Compile the model after loading weights
#     model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#     model_checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, save_freq='epoch')
#     epoch_file_callback = EpochFileCallback(epoch_file_path)
#     lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6)
#     tensorboard_callback = TensorBoard(log_dir='C:/files/plant-disease-detection/logs')

#     history = model.fit(
#         train_custom_generator,
#         epochs=50,
#         initial_epoch=initial_epoch,
#         validation_data=val_custom_generator,
#         callbacks=[early_stopping, model_checkpoint, epoch_file_callback, lr_scheduler, tensorboard_callback]
#     )

#     model.save('C:/files/plant-disease-detection/models/plant_disease_model_final.keras')

#     convert_to_tflite('C:/files/plant-disease-detection/models/plant_disease_model_final.keras', 'C:/files/plant-disease-detection/models/plant_disease_model_final.tflite')

# if __name__ == "__main__":
#     train_initial_model()
'''

# import os
# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# # Disable oneDNN optimizations for consistent results
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# # Check for GPU availability and configure memory growth
# def check_gpu():
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         print("GPU is available!")
#         for gpu in gpus:
#             try:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             except Exception as e:
#                 print(f"Error setting memory growth for GPU {gpu}: {e}")
#         cuda.init()
#         for i in range(cuda.Device.count()):
#             gpu = cuda.Device(i)
#             print(f"GPU {i} Name: {gpu.name()}")
#     else:
#         print("No GPU found. Training will use the CPU.")

# check_gpu()

# class CustomImageDataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, file_paths, labels, batch_size, image_size, datagen, num_classes, **kwargs):
#         super().__init__(**kwargs)
#         self.file_paths = file_paths
#         self.labels = labels
#         self.batch_size = batch_size
#         self.image_size = image_size
#         self.datagen = datagen
#         self.num_classes = num_classes
#         self.indexes = np.arange(len(self.file_paths))
#         self.on_epoch_end()

#     def __len__(self):
#         return int(np.ceil(len(self.file_paths) / self.batch_size))

#     def __getitem__(self, index):
#         batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#         batch_file_paths = [self.file_paths[i] for i in batch_indexes]
#         batch_labels = [self.labels[i] for i in batch_indexes]
        
#         images = []
#         labels = []
        
#         for file_path, label in zip(batch_file_paths, batch_labels):
#             try:
#                 img = self.load_and_preprocess_image(file_path)
#                 images.append(img)
#                 labels.append(label)
#             except Exception as e:
#                 print(f"Error processing image {file_path}: {e}")
        
#         if len(images) == 0:
#             return np.zeros((self.batch_size, *self.image_size, 3)), np.zeros((self.batch_size, self.num_classes))

#         images = np.array(images)
#         labels = np.array(labels)
#         labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)

#         return images, labels
    
#     def load_and_preprocess_image(self, file_path):
#         img = load_img(file_path, target_size=self.image_size)
#         img = img_to_array(img)
#         img = self.datagen.random_transform(img)
#         img = self.datagen.standardize(img)
#         return img

#     def on_epoch_end(self):
#         np.random.shuffle(self.indexes)

# def convert_to_tflite(model_path, tflite_model_path, convert=False):
#     if convert:
#         model = tf.keras.models.load_model(model_path)
#         converter = tf.lite.TFLiteConverter.from_keras_model(model)
#         tflite_model = converter.convert()
#         with open(tflite_model_path, 'wb') as f:
#             f.write(tflite_model)
#         print(f"Model converted to TensorFlow Lite format and saved at {tflite_model_path}")

# class EpochFileCallback(tf.keras.callbacks.Callback):
#     def __init__(self, epoch_file_path):
#         self.epoch_file_path = epoch_file_path

#     def on_epoch_end(self, epoch, logs=None):
#         with open(self.epoch_file_path, 'w') as file:
#             file.write(str(epoch + 1))

# def train_initial_model():
#     train_dir = 'C:/files/plant-disease-detection/data'
#     checkpoint_path = 'C:/files/plant-disease-detection/models/checkpoint.weights.h5'
#     epoch_file_path = 'C:/files/plant-disease-detection/models/last_epoch.txt'
#     model_save_path = 'C:/files/plant-disease-detection/models/plant_disease_model_final.keras'
#     tflite_model_path = 'C:/files/plant-disease-detection/models/plant_disease_model_final.tflite'

#     if not os.path.exists(train_dir):
#         raise FileNotFoundError(f"Training directory not found: {train_dir}")

#     os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
#     os.makedirs(os.path.dirname(epoch_file_path), exist_ok=True)

#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=40,
#         width_shift_range=0.3,
#         height_shift_range=0.3,
#         shear_range=0.3,
#         zoom_range=0.3,
#         horizontal_flip=True,
#         vertical_flip=True,
#         fill_mode='nearest',
#         validation_split=0.2
#     )

#     train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(128, 128),
#         batch_size=32,
#         class_mode='categorical',
#         subset='training',
#         shuffle=True
#     )

#     validation_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(128, 128),
#         batch_size=32,
#         class_mode='categorical',
#         subset='validation',
#         shuffle=False
#     )

#     train_file_paths = train_generator.filepaths
#     train_labels = train_generator.labels
#     val_file_paths = validation_generator.filepaths
#     val_labels = validation_generator.labels

#     num_classes = train_generator.num_classes

#     train_custom_generator = CustomImageDataGenerator(train_file_paths, train_labels, 16, (128, 128), train_datagen, num_classes)
#     val_custom_generator = CustomImageDataGenerator(val_file_paths, val_labels, 16, (128, 128), train_datagen, num_classes)

#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
#     base_model.trainable = True  # Unfreeze all layers for fine-tuning

#     model = Sequential([
#         base_model,
#         GlobalAveragePooling2D(),
#         Dense(512, activation='relu'),
#         Dropout(0.5),
#         Dense(num_classes, activation='softmax')
#     ])

#     model.build(input_shape=(None, 128, 128, 3))

#     if os.path.exists(checkpoint_path):
#         try:
#             model.load_weights(checkpoint_path)
#             print("Loaded model weights from checkpoint")
#         except Exception as e:
#             print(f"Error loading weights: {e}")
    
#     initial_epoch = 0
#     if os.path.exists(epoch_file_path):
#         with open(epoch_file_path, 'r') as file:
#             try:
#                 initial_epoch = int(file.read().strip())
#                 print(f"Resuming training from epoch {initial_epoch}")
#             except ValueError:
#                 print("Invalid value in epoch file, starting from scratch.")
#     else:
#         print("Epoch file not found, starting from scratch.")

#     model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     model_checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, save_freq='epoch')
#     epoch_file_callback = EpochFileCallback(epoch_file_path)
#     lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6)
#     tensorboard_callback = TensorBoard(log_dir='C:/files/plant-disease-detection/logs')

#     history = model.fit(
#         train_custom_generator,
#         epochs=100,  # Increase number of epochs for better learning
#         initial_epoch=initial_epoch,
#         validation_data=val_custom_generator,
#         callbacks=[early_stopping, model_checkpoint, epoch_file_callback, lr_scheduler, tensorboard_callback]
#     )

#     model.save(model_save_path)
#     convert_to_tflite(model_save_path, tflite_model_path, convert=True)

# if __name__ == "__main__":
#     train_initial_model()


import os
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Disable oneDNN optimizations for consistent results
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set the visible device to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check for GPU availability and configure memory growth
def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available!")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"Error setting memory growth for GPU {gpu}: {e}")
        cuda.init()
        for i in range(cuda.Device.count()):
            gpu = cuda.Device(i)
            print(f"GPU {i} Name: {gpu.name()}")
    else:
        print("No GPU found. Training will use the CPU.")

check_gpu()

class CustomImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size, image_size, datagen, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.datagen = datagen
        self.num_classes = num_classes
        self.indexes = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_file_paths = [self.file_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        
        images = []
        labels = []
        
        for file_path, label in zip(batch_file_paths, batch_labels):
            try:
                img = self.load_and_preprocess_image(file_path)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")
        
        if len(images) == 0:
            return np.zeros((self.batch_size, *self.image_size, 3)), np.zeros((self.batch_size, self.num_classes))

        images = np.array(images)
        labels = np.array(labels)
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)

        return images, labels
    
    def load_and_preprocess_image(self, file_path):
        img = load_img(file_path, target_size=self.image_size)
        img = img_to_array(img)
        img = self.datagen.random_transform(img)
        img = self.datagen.standardize(img)
        return img

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

def convert_to_tflite(model_path, tflite_model_path, convert=False):
    if convert:
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Model converted to TensorFlow Lite format and saved at {tflite_model_path}")

class EpochFileCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_file_path):
        self.epoch_file_path = epoch_file_path

    def on_epoch_end(self, epoch, logs=None):
        with open(self.epoch_file_path, 'w') as file:
            file.write(str(epoch + 1))

def train_initial_model():
    train_dir = 'C:/files/plant-disease-detection/data'
    checkpoint_path = 'C:/files/plant-disease-detection/models_temp/checkpoint.weights.h5'
    epoch_file_path = 'C:/files/plant-disease-detection/models_temp/last_epoch.txt'
    model_save_path = 'C:/files/plant-disease-detection/models_temp/plant_disease_model_final.keras'
    tflite_model_path = 'C:/files/plant-disease-detection/models_temp/plant_disease_model_final.tflite'

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(epoch_file_path), exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    train_file_paths = train_generator.filepaths
    train_labels = train_generator.labels
    val_file_paths = validation_generator.filepaths
    val_labels = validation_generator.labels

    num_classes = train_generator.num_classes

    train_custom_generator = CustomImageDataGenerator(train_file_paths, train_labels, 16, (128, 128), train_datagen, num_classes)
    val_custom_generator = CustomImageDataGenerator(val_file_paths, val_labels, 16, (128, 128), train_datagen, num_classes)

    # Explicitly place the model on GPU 0
    with tf.device('/GPU:0'):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        base_model.trainable = True  # Unfreeze all layers for fine-tuning

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.build(input_shape=(None, 128, 128, 3))

    if os.path.exists(checkpoint_path):
        try:
            model.load_weights(checkpoint_path)
            print("Loaded model weights from checkpoint")
        except Exception as e:
            print(f"Error loading weights: {e}")
    
    initial_epoch = 0
    if os.path.exists(epoch_file_path):
        with open(epoch_file_path, 'r') as file:
            try:
                initial_epoch = int(file.read().strip())
                print(f"Resuming training from epoch {initial_epoch}")
            except ValueError:
                print("Invalid value in epoch file, starting from scratch.")
    else:
        print("Epoch file not found, starting from scratch.")

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, save_freq='epoch')
    epoch_file_callback = EpochFileCallback(epoch_file_path)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6)
    tensorboard_callback = TensorBoard(log_dir='C:/files/plant-disease-detection/logs/logs_temp')

    history = model.fit(
        train_custom_generator,
        epochs=100,  # Increase number of epochs for better learning
        initial_epoch=initial_epoch,
        validation_data=val_custom_generator,
        callbacks=[early_stopping, model_checkpoint, epoch_file_callback, lr_scheduler, tensorboard_callback]
    )

    model.save(model_save_path)
    convert_to_tflite(model_save_path, tflite_model_path, convert=True)

if __name__ == "__main__":
    train_initial_model()