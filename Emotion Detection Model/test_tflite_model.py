import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="./models/emotion_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the input size for the model (assuming the model expects 48x48 RGB images)
input_size = (48, 48)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to the input size of the model
    resized_frame = cv2.resize(frame, input_size)
    
    # Convert the frame to RGB (if it's BGR, OpenCV captures in BGR by default)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Normalize the image (if the model was trained on normalized images)
    input_data = np.expand_dims(rgb_frame, axis=0).astype(np.float32) / 255.0

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the class with the highest probability
    predicted_class = np.argmax(output_data)

    # You can map the predicted class to the corresponding emotion
    emotions = ['happy', 'sad', 'angry', 'surprise', 'disgust', 'neutral', 'fear']
    predicted_emotion = emotions[predicted_class]

    # Display the result on the frame
    cv2.putText(frame, f"Emotion: {predicted_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
