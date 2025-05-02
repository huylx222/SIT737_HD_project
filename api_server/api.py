import cv2
import numpy as np
import os
import base64
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Import your custom service
from service import FASInferenceService

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize SocketIO with threading mode instead of eventlet
socketio = SocketIO(app, 
                  async_mode='threading',  # Use threading instead of eventlet
                  cors_allowed_origins="*",
                  ping_timeout=60,
                  ping_interval=25)

# Set up upload folder
UPLOAD_FOLDER = 'api_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize MediaPipe Face Detection
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Initialize FASInferenceService
service = FASInferenceService("model.ini")  # Provide your config path here

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({'status': 'success', 'message': 'API is working'})

@socketio.on('image_frame')
def handle_image_frame(data):
    # try:
        # Convert the base64 encoded image to binary
    image_data = base64.b64decode(data['image'])
    
    # Save temporarily
    filename = 'temp_image.jpg'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(image_data)
    
    # Process the image (Detect faces using MediaPipe)
    result = process_image(filepath)
    # Extract bounding boxes from the result
    faces = result.get('faces', [])
    spoof_results = []
    if faces:
        for face in faces:
            bbox_dict = face['bbox']
            bbox = [
                int(bbox_dict['xmin']),
                int(bbox_dict['ymin']),
                int(bbox_dict['xmax']),
                int(bbox_dict['ymax'])
            ]
            print(bbox)
            spoof_result = service.process_image(image_data, bbox)
            spoof_results.append(spoof_result)

        # Include the spoof results in the response
        result['spoof_results'] = spoof_results

    # Clean up
    if os.path.exists(filepath):
        os.remove(filepath)

    # Send back the result (including spoof results)
    emit('image_result', result)
    # except Exception as e:
    #     emit('error', {'message': str(e)})


def process_image(filepath):
    """Process an image file and detect faces"""
    try:
        image = cv2.imread(filepath)
        if image is None:
            return {'error': 'Could not read image'}, 400
            
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect faces
        results = face_detection.process(image_rgb)
        
        # Prepare response
        faces = []
        image_height, image_width, _ = image.shape
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                xmin = int(bbox.xmin * image_width)
                ymin = int(bbox.ymin * image_height)
                width = int(bbox.width * image_width)
                height = int(bbox.height * image_height)
                
                # MediaPipe gives confidence as a detection score
                confidence = detection.score[0] if detection.score else 0
                
                faces.append({
                    'bbox': {
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmin + width,
                        'ymax': ymin + height
                    },
                    'confidence': float(confidence)
                })
        
        return {
            'faces': faces,
            'image_width': image_width,
            'image_height': image_height
        }
    except Exception as e:
        return {'error': str(e)}, 500


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)
