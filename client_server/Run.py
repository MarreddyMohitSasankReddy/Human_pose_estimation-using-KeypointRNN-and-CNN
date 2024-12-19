from flask import Flask, render_template, Response, request, send_from_directory, url_for
import tensorflow as tf
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from evaluation import *

app = Flask(__name__, template_folder='template')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load the model
interpreter = tf.lite.Interpreter(model_path="C:\\Users\\bhari\\Desktop\\dl_project_a6\\client_server\\model.tflite")
interpreter.allocate_tensors()

# Input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]

def process_image(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Resize and pad the image to the model's input size
    input_image = cv2.resize(image, (input_size, input_size))
    input_image = np.expand_dims(input_image, axis=0)

    # Run model inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Visualize the predictions with the original image
    output_overlay = draw_prediction_on_image(image, keypoints_with_scores)

    # Save the result with modified image name and format
    result_image_name = os.path.join(app.config['UPLOAD_FOLDER'], 
                                     secure_filename(image_path.split('.')[0] + '_result.' + image_path.split('.')[1]))
    cv2.imwrite(result_image_name, output_overlay)

    return result_image_name

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
        
            # Process the uploaded image and generate a prediction image
            predict_image_path = process_image(upload_path)
        
            return render_template('result.html', original_image=url_for('serve_uploaded_file', filename=filename), 
                                                   predict_image=url_for('serve_uploaded_file', filename=os.path.basename(predict_image_path)))
        else:
            return 'Invalid file type'

    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def process_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (input_size, input_size))
        input_image = np.expand_dims(frame_resized, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        output_frame = draw_prediction_on_image(frame, keypoints_with_scores)

        ret, jpeg = cv2.imencode('.jpg', output_frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the app on the local network IP address with port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
