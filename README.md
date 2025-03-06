# Human Eye UI Attention Predictor

A sophisticated web application that predicts user attention patterns on UI elements using advanced computer vision and machine learning techniques. This project combines a modern Next.js frontend with a powerful FastAPI backend to provide real-time UI attention predictions.

## 🌟 Features

- **Real-time UI Attention Prediction**: Upload UI screenshots and get instant attention heatmaps
- **User Context-Aware**: Takes into account user age, tech-savviness, and platform preferences
- **Multi-Platform Support**: Works across different platforms (web, mobile, desktop)
- **Modern Tech Stack**: Built with Next.js, FastAPI, and advanced ML models
- **Beautiful UI**: Clean and intuitive user interface
- **API-First Design**: RESTful API endpoints for easy integration

## 👁️ Human Eye Modeling Approach

Our system employs a sophisticated multi-layered approach to model human eye behavior:

### 1. Age-Based Eye Scanning Patterns
- **Spotted Pattern** (Teenagers): Scattered, non-linear exploration with high visual acuity
- **Z-Pattern** (Young Adults): Quick zigzag assessment pattern for rapid information gathering
- **F-Pattern** (Middle-aged): Horizontal top scanning followed by vertical left scanning
- **Layered Pattern** (Elderly): Methodical, layer-by-layer reading with increased focus time

### 2. Visual Attention Rules
- **Contrast Sensitivity**: Analyzes color contrast in LAB color space for better perceptual accuracy
- **Color Intensity**: Evaluates color vibrancy and saturation
- **Element Isolation**: Measures visual separation from surrounding elements
- **Size and Position**: Considers element dimensions and screen position

### 3. Platform-Specific Hotspots
- **Android**: Status bar, navigation buttons, system UI elements
- **iOS**: Top bar, home indicator, system controls
- **Desktop**: Menu bars, taskbar, system tray

### 4. Mathematical Models
- **Normal Distribution**: Age-based pattern probabilities using Gaussian distributions
- **Attention Scoring**: Weighted combination of:
  - Visual attraction score (contrast, color, isolation)
  - Base position score (screen location)
  - Task relevance score (semantic similarity)
- **Proximity Merging**: Clustering of nearby attention points
- **Confidence Thresholding**: Filtering predictions based on confidence scores

### 5. Machine Learning Integration
- **Sentence Transformers**: For semantic understanding of UI elements and tasks
- **Visual Language Models**: For zero-shot UI element classification
- **Platform Detection**: Automated platform identification from UI characteristics

## 🔄 Versioning

### Current Version (v0)
The current version implements a rule-based approach to UI attention prediction, combining:
- Statistical models for age-based patterns
- Visual perception rules
- Platform-specific knowledge
- Basic ML models for semantic understanding

### Upcoming Version (v1)
We're working on a major upgrade that will leverage advanced machine learning models:
- **Eye-Tracking Data Integration**: Training on real eye-tracking datasets
- **Deep Learning Models**: 
  - Attention prediction using transformer architectures
  - Dynamic pattern recognition for user behavior
  - Real-time adaptation to user preferences
- **Advanced Computer Vision**:
  - Object detection for UI elements
  - Scene understanding for context
  - Visual hierarchy analysis
- **Personalization Engine**:
  - User-specific attention patterns
  - Learning from interaction history
  - Adaptive confidence scoring

Stay tuned for v1 release, which will bring significant improvements in prediction accuracy and personalization!

## 🏗️ Project Structure

```
human-eye/
├── frontend/                 # Next.js frontend application
│   ├── app/                 # Next.js app directory
│   ├── components/          # Reusable React components
│   └── public/             # Static assets
├── models/                  # ML models and utilities
│   ├── predictor.py        # Main prediction logic
│   ├── ui_attention_predictor.py  # UI attention model
│   └── utils.py            # Helper functions
├── api.py                  # FastAPI backend server
├── requirements.txt        # Python dependencies
└── .env.example           # Environment variables template
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA-capable GPU (recommended for faster predictions)

### Backend Setup

1. Create and activate a virtual environment:
```bash
python -m venv virtual
source virtual/bin/activate  # On Windows: virtual\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Start the backend server:
```bash
python api.py
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env.local
# Edit .env.local with your configuration
```

4. Start the development server:
```bash
npm run dev
```

## 🔧 API Endpoints

### POST /predict
Predicts UI attention patterns based on the uploaded image and user context.

**Parameters:**
- `file`: UI screenshot image
- `age`: User age (integer)
- `task`: User's intended task
- `tech_saviness`: User's tech-savviness level (1-5)
- `platform`: Target platform (web/mobile/desktop)
- `debug`: Enable debug mode (optional)

**Response:**
- JSON object containing attention heatmap and predictions

## 🛠️ Technologies Used

### Frontend
- Next.js 13+
- TypeScript
- Tailwind CSS
- React

### Backend
- FastAPI
- PyTorch
- OpenCV
- NumPy
- Pillow

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers. 