const express = require('express');
const multer = require('multer');
const path = require('path');
const axios = require('axios');
const fs = require('fs');
const socketIoClient = require('socket.io-client');
const mongoose = require('mongoose');
const app = express();
const port = 3000;

// MongoDB connection configuration
const MONGO_USERNAME = process.env.MONGO_USERNAME || 'spoof-user';
const MONGO_PASSWORD = process.env.MONGO_PASSWORD || 'spoof-password';
const MONGO_HOST = process.env.MONGO_HOST || 'mongodb-service';
const MONGO_PORT = process.env.MONGO_PORT || '27017';
const MONGO_DB = process.env.MONGO_DB || 'spoof_detection';

const mongoURI = `mongodb://${MONGO_USERNAME}:${MONGO_PASSWORD}@${MONGO_HOST}:${MONGO_PORT}/${MONGO_DB}?authSource=admin`;

console.log(`Attempting to connect to MongoDB at ${MONGO_HOST}:${MONGO_PORT}/${MONGO_DB}`);

// Connect to MongoDB with robust error handling
mongoose.connect(mongoURI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  serverSelectionTimeoutMS: 5000,
  connectTimeoutMS: 10000
})
  .then(() => {
    console.log('Successfully connected to MongoDB');
  })
  .catch(err => {
    console.error('MongoDB connection error:', err);
  });

// Create a MongoDB schema and model for detections
const detectionSchema = new mongoose.Schema({
  timestamp: { type: Date, default: Date.now },
  faces: [Object],
  spoof_results: [Object],
  image_width: Number,
  image_height: Number,
  filename: String
});

const Detection = mongoose.model('Detection', detectionSchema);

// Connect to the Python API WebSocket
const apiUrl = process.env.API_URL || 'http://localhost:5001';
console.log(`Connecting to API at: ${apiUrl}`);
const socket = socketIoClient(apiUrl);

socket.on('connect', () => {
  console.log('Connected to API socket server');
});

socket.on('connect_error', (error) => {
  console.error('Socket connection error:', error);
});

// Ensure uploads directory exists
const uploadDir = './uploads';
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// Multer setup for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname)),
});

const fileFilter = (req, file, cb) => {
  const allowedTypes = /jpeg|jpg|JPG|png|gif/;
  const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
  const mimetype = allowedTypes.test(file.mimetype);
  if (extname && mimetype) cb(null, true);
  else cb(new Error('Only image files are allowed'));
};

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
  fileFilter,
});

// Serve static files if needed
app.use(express.static('public'));
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

// Upload endpoint
app.post('/upload', upload.single('image'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

  const imagePath = req.file.path;
  const imageBuffer = fs.readFileSync(imagePath);
  const base64Image = imageBuffer.toString('base64');

  // Timeout fallback (optional but recommended)
  const timeout = setTimeout(() => {
    fs.unlinkSync(imagePath);
    res.status(504).json({ error: 'API timeout' });
  }, 10000); // 10 seconds

  // Listen once for the response
  socket.once('image_result', (data) => {
    clearTimeout(timeout);
    
    // Save detection results to MongoDB
    if (data.faces && data.faces.length > 0) {
      try {
        const detection = new Detection({
          faces: data.faces,
          spoof_results: data.spoof_results || [],
          image_width: data.image_width,
          image_height: data.image_height,
          filename: req.file.originalname
        });
        
        detection.save()
          .then(() => console.log('Detection saved to MongoDB'))
          .catch(err => console.error('Error saving to MongoDB:', err));
      } catch (error) {
        console.error('Error creating detection document:', error);
      }
    }
    
    fs.unlinkSync(imagePath);
    console.log('Received image_result:', data);
    res.json(data);
  });

  socket.once('error', (error) => {
    clearTimeout(timeout);
    fs.unlinkSync(imagePath);
    res.status(500).json({ error: error.message || 'API error' });
  });

  // Send the image to the Python API
  socket.emit('image_frame', {
    image: base64Image,
    filename: req.file.originalname,
  });
});

// API connectivity test
app.get('/test-api', async (req, res) => {
  try {
    const response = await axios.get(`${apiUrl}/test`);
    res.json({ 
      message: 'Connection successful', 
      apiResponse: response.data 
    });
  } catch (error) {
    res.status(500).json({
      error: 'Failed to connect to API',
      details: error.message,
    });
  }
});

// Get detection history from MongoDB
app.get('/detection-history', async (req, res) => {
  try {
    const detections = await Detection.find({})
      .sort({ timestamp: -1 })
      .limit(100)
      .lean();
    
    res.json(detections);
  } catch (error) {
    console.error('Error fetching detection history:', error);
    res.status(500).json({
      error: 'Failed to fetch detection history',
      details: error.message,
    });
  }
});

// MongoDB status endpoint with detailed information
app.get('/mongodb-status', (req, res) => {
  const state = mongoose.connection.readyState;
  const stateMap = {
    0: 'disconnected',
    1: 'connected',
    2: 'connecting',
    3: 'disconnecting'
  };
  
  // Get more detailed information about the connection
  const connectionDetails = {
    status: stateMap[state] || 'unknown',
    database: mongoose.connection.name || 'none',
    host: mongoose.connection.host || 'unknown',
    port: mongoose.connection.port || 'unknown',
    models: Object.keys(mongoose.models) || [],
    connectionURI: mongoURI.replace(/\/\/([^:]+):([^@]+)@/, '//***:***@') // Hide credentials
  };
  
  res.json(connectionDetails);
});

// Start server
app.listen(port, () => {
  console.log(`Web app running at http://localhost:${port}`);
});