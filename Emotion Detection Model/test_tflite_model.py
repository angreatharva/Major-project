import cv2
import numpy as np
import tensorflow as tf

def test_tflite_model():
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="./models/emotion_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Define the input size for the model
    input_size = (48, 48)

    # Define emotions
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Press 'q' to exit.")

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Resize the frame to the input size of the model
        resized_frame = cv2.resize(frame, input_size)
        
        # Convert the frame to RGB (OpenCV captures in BGR by default)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Normalize the image
        input_data = np.expand_dims(rgb_frame, axis=0).astype(np.float32) / 255.0

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get the class with the highest probability
        predicted_class = np.argmax(output_data[0])
        confidence = output_data[0][predicted_class] * 100  # Convert to percentage
        
        # Get the predicted emotion
        predicted_emotion = emotions[predicted_class]

        # Display the result on the frame
        cv2.putText(display_frame, f"Emotion: {predicted_emotion}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Confidence: {confidence:.2f}%", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw a box around the face if you have face detection (optional)
        # For simplicity, not including face detection here
        
        # Show the frame
        cv2.imshow('Emotion Detection', display_frame)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_tflite_model()