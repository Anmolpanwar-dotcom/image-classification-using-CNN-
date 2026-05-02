PetClassifier AI: End-to-End Image Classification
Project Overview
This project is a Deep Learning application designed to classify images between Cats and Dogs. The primary objective was to build a scalable machine learning pipeline covering everything from data preprocessing to cloud deployment. I utilized Convolutional Neural Networks (CNN) which are highly effective at extracting automatic patterns like edges, textures, and shapes from raw images.

Live Demo
App Link: https://fj7a59azhyfy8xt4mhhqxc.streamlit.app/


Technical Stack and Architecture
Core Technologies
Deep Learning: TensorFlow, Keras API
Computer Vision: OpenCV, Pillow
Frontend: Streamlit for Interactive UI
Model Hosting: Hugging Face Hub

CNN Architecture Logic
The model is designed in three main segments:

Feature Extraction: Multiple Conv2D and MaxPooling2D layers that drill into the image depth to find critical features.

Regularization: A Dropout layer set at 0.5 is implemented to prevent the model from memorizing training data, effectively controlling overfitting.

Classification Head: Flatten and Dense layers finalize the extracted features to produce a binary prediction via the Sigmoid activation function.

Repository Structure
app.py: Main Streamlit application
requirements.txt: List of dependencies including TensorFlow and Streamlit
README.md: Project Documentation
.gitignore: Files to be ignored by Git

Execution Flow
Model Loading: Upon startup, the app downloads the pre-trained .h5 model from Hugging Face and stores it in the cache for efficiency.

Preprocessing: The uploaded user image is resized to 150x150 pixels and normalized to a 0-1 range.

Inference: The model applies mathematical weights to the image to generate a probability score.

Result Display: If the score is greater than 0.5, the model identifies the pet as a Dog; otherwise, it is classified as a Cat.

Performance and Key Learnings
Data Cleaning: Learned the process of filtering corrupt and non-RGB images from the dataset.
Optimization: Improved the training pipeline speed using tf.data.AUTOTUNE and caching.
Deployment: Handled the challenges of shifting a model from a local environment to the cloud, specifically managing version mismatches and dependency errors.

About the Author
Anmol Panwar
Education: BCA (2022-2025) - Chandigarh, India
Role: Aspiring Machine Learning Engineer
Skills: Python, SQL, Deep Learning, Computer Vision, YOLO
