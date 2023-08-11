[Automatic Hamburger Ingredients Labelling by ML Classifier Paper](paper.pdf)

### Implement the project

**Classifier Model:** We will use the VGGNet CNN model for this task as it has proven effective in various image classification problems and is suitable for our case of food ingredient identification.

**Image Processing:** Since no proficiency is expected, we will rely on simple logics. We will divide the camera view into 10 equal parts and assume that ingredients are placed on the centers of these divisions. We will extract image patches centered on these division points.

**Data:** For our initial model, we will need to create a dataset from images captured with our camera or we can use existing datasets found on the internet. We would need labeled images of each ingredient, preferably centered in the image for simplicity.

**Implementation:** We will use Python and its powerful libraries for machine learning, such as TensorFlow, Keras, and OpenCV.

**Labeling:** Once our model is trained and able to make predictions, we will create a script that takes a new image, divides it into sections, extracts the central points, makes a prediction for each point and finally draws a box with the corresponding label.

### Project outline with steps to follow

**Data Collection:** Collect images of each ingredient. Each image should ideally contain a single ingredient centered in the image.

**Data Preprocessing:** Preprocess the images. This includes resizing images, normalizing pixel values, dividing images into equal parts, and extracting the central points as per the project requirements.

**Model Training:** Implement the VGGNet model using TensorFlow and Keras. Train the model using the preprocessed images and corresponding labels.

**Model Evaluation:** Split your dataset into training, validation, and test datasets. Evaluate your model on the test dataset to understand its performance.

**Model Prediction & Visualization:** Write a function to predict labels on new images. This function should also draw boxes and write the predicted labels on the boxes.

### Compile the model

**Model Definition and Compilation:** The model architecture is defined using the Sequential API, consisting of convolutional, max-pooling, flatten, and dense layers. The model is compiled with the Adam optimizer and categorical cross-entropy loss.

### Prepare the data

**Data Preparation:** The ImageDataGenerator is used to load and preprocess the image data from the specified directory. Separate generators are created for training and validation data using the flow_from_directory method.

### Train the model

**Model Training:** The model is trained using the fit method with the training data generator. The number of epochs and the validation data generator are specified.

### Plotting Training Loss and Training Accuracy

**Plotting Training Loss and Accuracy:** The training and validation loss and accuracy are plotted using Matplotlib. Two subplots are created for loss and accuracy.

### Guessing what a material is by giving a picture of it

**Prediction and Visualization:** The predict_and_visualize function takes an image path, loads and preprocesses the image, performs prediction using the trained model, and visualizes the prediction by drawing a rectangle around the image and displaying the predicted ingredient name.

### Guessing each ingredient separately by giving all the hamburger ingredients

**Multiple-Image Grid Prediction and Visualization:** The predict_and_visualize_multiple function loads the trained model, loads the input image, determines the grid size based on the image dimensions, iterates over the grid, extracts each tile from the image, preprocesses the tile, performs prediction using the model, and visualizes the predictions by drawing rectangles and displaying the predicted ingredient names within each tile.