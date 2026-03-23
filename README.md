# 📰 Fake News Detection System - Full-Stack Web Application

A comprehensive AI-powered fake news detection system built with Python, Machine Learning, Flask, and modern web technologies. This project uses advanced Natural Language Processing (NLP) techniques and multiple machine learning algorithms to classify news articles as real or fake.

## 🚀 Features

- **Advanced AI**: Multiple machine learning algorithms (Logistic Regression, Naive Bayes, SVM)
- **Modern Web Interface**: Beautiful dark-themed responsive web application
- **User Authentication**: Secure user registration and login system
- **Database Storage**: SQLite database for user accounts and analysis history
- **Real-time Analysis**: Instant prediction with confidence scores
- **Analysis History**: Track all your previous analyses
- **Model Comparison**: Compare predictions across different algorithms
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices

## 📁 Project Structure

```
fake-news-detection/
│
├── dataset/
│   └── sample_news.csv          # Sample dataset for testing
│
├── src/
│   ├── __init__.py             # Package initialization
│   ├── preprocessing.py        # NLP text preprocessing module
│   ├── model_training.py       # ML model training & evaluation
│   └── prediction.py          # Prediction and inference module
│
├── templates/                  # HTML templates
│   ├── base.html              # Base template with navigation
│   ├── index.html             # Home page
│   ├── login.html             # Login page
│   ├── register.html          # Registration page
│   ├── dashboard.html         # User dashboard
│   └── history.html           # Analysis history
│
├── static/                    # Static files
│   ├── css/
│   │   └── style.css          # Main stylesheet
│   └── js/
│       ├── main.js            # General JavaScript
│       └── dashboard.js       # Dashboard functionality
│
├── models/                    # Trained model storage
│   ├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
│   ├── logistic_regression_model.pkl
│   ├── naive_bayes_model.pkl
│   └── svm_model.pkl
│
├── app_flask.py               # Flask web application
├── train_models.py            # Training script
├── demo.py                    # Demo script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data

The preprocessing module requires NLTK data. Run the following in Python:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('omw-1.4')
```

## 🚀 Running the Application

### Step 1: Train the Models (Already Done)

The models are pre-trained on the sample dataset. To retrain with your own data:

```bash
python train_models.py
```

### Step 2: Start the Flask Application

```bash
python app_flask.py
```

### Step 3: Access the Application

Open your browser and navigate to: `http://localhost:5000`

## 🌐 Web Application Features

### Public Pages
- **Home**: Overview of the system with features and demo examples
- **Login**: User authentication
- **Register**: Create new user account

### Protected Pages (Require Login)
- **Dashboard**: Main analysis interface with real-time prediction
- **History**: View all previous analyses with pagination
- **Statistics**: User analytics and insights

### Key Features
- **Real-time Analysis**: Paste news content and get instant results
- **Model Selection**: Choose between single model or compare all models
- **Confidence Scoring**: Visual confidence meters and probability displays
- **Analysis History**: Complete history of all analyses with timestamps
- **Responsive Design**: Works on all device sizes
- **Dark Theme**: Modern dark blue and black color scheme

## 🔧 API Endpoints

The Flask application provides REST API endpoints:

- `POST /analyze` - Analyze text with single model
- `POST /analyze-multiple` - Analyze text with all models
- `GET /api/history` - Get user's analysis history
- `POST /login` - User authentication
- `POST /register` - User registration

## 📊 Model Performance

The system achieves:
- **Accuracy**: 85-95% on test data
- **Precision**: 80-90% for fake news detection
- **Recall**: 75-85% for fake news detection
- **F1 Score**: 80-90% overall

## 🗄️ Database Schema

The application uses SQLite with the following tables:

- **User**: User accounts (id, username, email, password, created_at)
- **Prediction**: Analysis results (id, text, prediction, confidence, model_used, user_id, created_at)

## 🔒 Security Features

- **Password Hashing**: Bcrypt password encryption
- **Session Management**: Secure Flask-Login sessions
- **CSRF Protection**: Cross-site request forgery protection
- **Input Validation**: Client and server-side validation
- **SQL Injection Prevention**: SQLAlchemy ORM protection

## 🎨 Frontend Technologies

- **HTML5**: Semantic markup structure
- **CSS3**: Custom dark theme with responsive design
- **JavaScript (ES6+)**: Interactive functionality and AJAX requests
- **Font Awesome**: Icons and visual elements

## 🔧 Customization

### Changing Theme Colors

Edit `static/css/style.css` and modify the CSS custom properties:

```css
:root {
    --primary-color: #1f77b4;    /* Change primary color */
    --dark-bg: #0e1117;          /* Change background */
    --card-bg: #1e1e1e;          /* Change card backgrounds */
}
```

### Adding New ML Models

1. Train your model and save it to the `models/` directory
2. Update the `FakeNewsPredictor` class in `src/prediction.py`
3. Add the model to the UI in `templates/dashboard.html`

### Database Configuration

To use a different database, modify the SQLAlchemy URI in `app_flask.py`:

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/dbname'
```

## 🧪 Testing

### Run the Demo Script

```bash
python demo.py
```

### Test with Sample Data

The application includes sample real and fake news articles for testing.

### API Testing

Use tools like Postman or curl to test the API endpoints:

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news text here"}'
```

## 🚀 Deployment

### live hear https://truelence.onrender.com

```bash
export FLASK_ENV=development
python app_flask.py
```

### Production Deployment

1. Set environment variables:
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key-here
```

2. Use a production WSGI server:
```bash
pip install gunicorn
gunicorn app_flask:app
```

3. Consider using a reverse proxy like Nginx

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NLTK** for natural language processing
- **scikit-learn** for machine learning algorithms
- **Flask** for the web framework
- **Font Awesome** for icons
- **Kaggle** for providing news datasets

---

**Happy detecting fake news! 📰✨**
