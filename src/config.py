"""
Configuration file for Traffic Violation Detection System
"""

# Detection Classes (matching data.yaml)
CLASSES = {
    0: 'helmet',
    1: 'no_helmet',
    2: 'mobile_usage',
    3: 'traffic_violation',
    4: 'overspeed',
    5: 'number_plate'
}

# Speed Limits (km/h)
SPEED_LIMITS = {
    'urban': 40,
    'highway': 80,
    'school_zone': 30,
    'default': 50
}

# Detection Thresholds
DETECTION_CONFIG = {
    'confidence_threshold': 0.35,
    'nms_threshold': 0.45,
    'model_size': 640,
    'max_detections': 100
}

# Tracking Configuration
TRACKING_CONFIG = {
    'max_age': 30,  # frames
    'min_hits': 3,  # minimum detections before tracking
    'iou_threshold': 0.3
}

# Speed Estimation
SPEED_CONFIG = {
    'pixels_per_meter': 50,  # calibration factor
    'smoothing_frames': 5,   # frames to average speed
    'min_track_frames': 3    # minimum frames for speed calculation
}

# OCR Configuration
OCR_CONFIG = {
    'confidence_threshold': 0.6,
    'preprocessing': True,
    'upscale_factor': 2
}

# Violation Detection
VIOLATION_CONFIG = {
    'helmet_overlap_threshold': 0.3,  # IoU threshold for helmet-rider overlap
    'phone_overlap_threshold': 0.2,   # IoU threshold for phone-rider overlap
    'violation_debounce_frames': 3,   # frames to confirm violation
    'plate_confidence_threshold': 0.7,
    'speed_violation_threshold': 1.2,  # 20% over speed limit
    'traffic_signal_threshold': 0.5   # confidence for traffic signal violations
}

# Database Configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'traffic_violations',
    'user': 'postgres',
    'password': 'password'
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': True
}

# File Paths
PATHS = {
    'models': 'models/',
    'data': 'data/',
    'output': 'output/',
    'logs': 'logs/'
}

# Camera Calibration
CAMERA_CONFIG = {
    'fps': 30,
    'width': 1920,
    'height': 1080,
    'calibration_points': [
        # Define calibration points for speed estimation
        # Format: [(x1, y1), (x2, y2), distance_in_meters]
    ]
}
