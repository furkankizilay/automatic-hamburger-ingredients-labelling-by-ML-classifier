import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Define constants
IMG_SIZE = 224  # VGG16 input size
BATCH_SIZE = 32
N_CLASSES = 10  # Number of classes/ingredients in your dataset
N_EPOCHS = 10
DATA_DIR = "allData"  # Base directory containing ingredient directories

# Define the VGGNet model
model = Sequential([
    # Simplified version of VGGNet
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(N_CLASSES, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare the data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Train the model
history = model.fit(train_gen, epochs=N_EPOCHS, validation_data=val_gen)

model.summary()

# Plotting Training Loss
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Losses')
plt.legend()

# Plotting Training Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracies')
plt.legend()

plt.tight_layout()
plt.show() 

# Save the model
model.save('ingredient_classifier.h5')

ingredient_names = ['Avocado', 'Cucumber', 'Patty', 'Lettuce', 'Mushrooms', 'Onion Rings', 'Pepper', 'Pickle', 'Cheddar', 'Tomato']

def predict_and_visualize(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape((1, IMG_SIZE, IMG_SIZE, 3))
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_class_name = ingredient_names[predicted_class[0]]

    # Visualization
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    cv2.rectangle(img, (0, 0), (width, height), (0, 255, 0), 2)
    cv2.putText(img, predicted_class_name, (width//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
predict_and_visualize("allData/avocado/0.png")

def predict_and_visualize_multiple(img_path, model_path):
    # Load the pre-trained model
    model = load_model(model_path)
    
    # Load the image
    img = cv2.imread(img_path)
    
    # Determine the size of the grid
    grid_size = (2, 5) if len(img) < len(img[0]) else (5, 2)
    tile_size = (img.shape[0] // grid_size[0], img.shape[1] // grid_size[1])

    # Iterate over the grid
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Extract the tile
            tile = img[i * tile_size[0] : (i+1) * tile_size[0], j * tile_size[1] : (j+1) * tile_size[1]]

            # Preprocess the tile
            tile_resized = cv2.resize(tile, (IMG_SIZE, IMG_SIZE))
            tile_resized = tile_resized / 255.0
            tile_resized = tile_resized.reshape((1, IMG_SIZE, IMG_SIZE, 3))
            
            # Make prediction
            prediction = model.predict(tile_resized)
            predicted_class = np.argmax(prediction, axis=1)
            predicted_class_name = ingredient_names[predicted_class[0]]
            
            # Visualization
            cv2.rectangle(img, (j * tile_size[1], i * tile_size[0]), ((j+1) * tile_size[1], (i+1) * tile_size[0]), (0, 255, 0), 2)
            cv2.putText(img, str(predicted_class_name), (j * tile_size[1] + tile_size[1]//2, i * tile_size[0] + tile_size[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the function with an image
predict_and_visualize_multiple("test.png", "ingredient_classifier.h5")