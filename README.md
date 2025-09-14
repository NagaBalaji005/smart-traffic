# 🚦 Traffic Violation Detection System

An AI-powered traffic violation detection system using YOLOv8, PostgreSQL, and FastAPI.

## 🎯 Features

- **Helmet Detection**: Detects riders with/without helmets
- **Mobile Usage Detection**: Identifies drivers using phones
- **Traffic Violations**: Detects traffic signal violations
- **Speed Monitoring**: Identifies overspeeding vehicles
- **Number Plate Recognition**: Extracts license plate numbers
- **Real-time Dashboard**: Web interface for monitoring violations
- **Database Storage**: PostgreSQL integration for violation records

## 🚀 Quick Start

### 1. Start the System
```bash
python start.py
```

### 2. Access Dashboard
Open your browser and go to: **http://localhost:8081/dashboard**

### 3. Process Videos
```bash
python simple_video_processor.py dataset/raw/traffic_video_original.mp4 --output output
```

### 4. Run Full Demo
```bash
python demo_full_system.py dataset/raw/traffic_video_original.mp4 --output demo_output
```

## 📁 Project Structure

```
final-year-project/
├── start.py                          # Main startup script
├── simple_server.py                  # Web server with API
├── simple_video_processor.py         # Video processing script
├── demo_full_system.py               # Complete system demo
├── dashboard/
│   └── index.html                    # Web dashboard
├── models/
│   └── traffic_violations_best.pt    # Trained YOLOv8 model
├── dataset/
│   ├── data.yaml                     # Dataset configuration
│   ├── raw/                          # Input videos
│   ├── train/                        # Training images & labels
│   └── val/                          # Validation images & labels
├── src/                              # Core system modules
│   ├── config.py                     # Configuration settings
│   ├── detector.py                   # YOLOv8 detection
│   ├── database.py                   # Database operations
│   └── ...
└── requirements.txt                  # Python dependencies
```

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Database** (Optional):
   ```bash
   python setup_database.py
   ```

## 📊 API Endpoints

- **Dashboard**: `http://localhost:8081/dashboard`
- **Statistics**: `http://localhost:8081/stats`
- **Violations**: `http://localhost:8081/violations`
- **Upload Video**: `http://localhost:8081/upload-video`
- **API Docs**: `http://localhost:8081/docs`

## 🎬 Usage Examples

### Process a Video File
```bash
python simple_video_processor.py path/to/video.mp4 --output results
```

### Run Complete Demo
```bash
python demo_full_system.py path/to/video.mp4 --output demo_results
```

## 🔧 Configuration

Edit `src/config.py` to modify:
- Detection thresholds
- Model settings
- Database connection
- API configuration

## 📈 Performance

- **Processing Speed**: ~8-10 FPS on CPU
- **Detection Accuracy**: Custom trained model
- **Supported Formats**: MP4, AVI, MOV
- **Resolution**: Up to 4K supported

## 🎯 Detection Classes

1. **helmet** - Riders wearing helmets
2. **no_helmet** - Riders without helmets
3. **mobile_usage** - Drivers using phones
4. **traffic_violation** - Traffic signal violations
5. **overspeed** - Speeding vehicles
6. **number_plate** - License plates

## 🚨 Troubleshooting

### Server Won't Start
- Check if port 8081 is available
- Ensure all dependencies are installed
- Verify model file exists: `models/traffic_violations_best.pt`

### Database Connection Issues
- Install PostgreSQL
- Run `python setup_database.py`
- Check database credentials in `.env` file

### Video Processing Errors
- Ensure video file exists and is readable
- Check video format (MP4 recommended)
- Verify sufficient disk space for output

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**🎉 Your AI-powered traffic violation detection system is ready to use!**