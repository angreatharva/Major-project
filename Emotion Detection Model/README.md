Emotion Detection using TensorFlow Lite

This is a simple emotion detection model built using TensorFlow Lite. The model is trained on the FER2013 dataset, which contains images of faces with various emotions.

Step 1: Install the dependencies
run the below command in terminal
pip install -r requirements.txt

Step 2: Run the main.py file to make the model
run the below command in terminal
python main.py

Step 3: Convert the model to TensorFlow Lite format
run the below command in terminal
python convert_tflite.py

Step 3:
Test the tflite model
run the below command in terminal
python test_tflite_model.py

- to close the window press ctrl+c in the terminal

Training Pipeline:
a. Run the basic model: python main.py
b. Run the transfer learning model (typically better performance): python transfer_learning_main.py
c. Run the class-weighted model (better for imbalanced datasets): python class_weight_main.py
d. Evaluate the ensemble of models (after training at least 2 models): python ensemble_model.py

python convert_tflite.py
python test_tflite_model.py
