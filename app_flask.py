"""
Fake News Detection - Flask Web Application

A full-stack web application for fake news detection with user authentication,
database storage, and modern frontend.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf.csrf import CSRFProtect
from datetime import datetime
import os
from src.prediction import FakeNewsPredictor
import json

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fake_news.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['WTF_CSRF_TIME_LIMIT'] = None  # No time limit for CSRF tokens

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Initialize predictor
predictor = FakeNewsPredictor()

# Database Models
class User(db.Model, UserMixin):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='author', lazy=True)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

class Prediction(db.Model):
    """Prediction model to store analysis results."""
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    probability_real = db.Column(db.Float, nullable=False, default=0.0)
    probability_fake = db.Column(db.Float, nullable=False, default=0.0)
    model_used = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Prediction('{self.prediction}', '{self.confidence}')"

@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login."""
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    """Home page route."""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page route."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login unsuccessful. Please check email and password', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page route."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation checks
        if not username or not email or not password:
            flash('All fields are required', 'danger')
            return render_template('register.html')
        
        if len(username) < 3 or len(username) > 20:
            flash('Username must be between 3 and 20 characters', 'danger')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'danger')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')

        # Check if user already exists (case-insensitive for email)
        existing_user = User.query.filter(
            (User.email.ilike(email)) | (User.username.ilike(username))
        ).first()
        
        if existing_user:
            if existing_user.email.lower() == email.lower():
                flash('Email already registered. Please use a different email or try logging in.', 'warning')
            else:
                flash('Username already taken. Please choose a different username.', 'warning')
            return render_template('register.html')

        try:
            hashed_password = bcrypt.generate_password_hash(password)
            if isinstance(hashed_password, bytes):
                hashed_password = hashed_password.decode('utf-8')
            user = User(username=username, email=email, password=hashed_password)

            db.session.add(user)
            db.session.commit()

            flash('Account created successfully! You can now log in.', 'success')
            return redirect(url_for('login'))
        
        except Exception as e:
            db.session.rollback()
            print(f"Registration error: {e}")
            flash('An error occurred during registration. Please try again.', 'danger')
            return render_template('register.html')

    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard page for logged-in users."""
    # Get user's recent predictions
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                 .order_by(Prediction.created_at.desc())\
                                 .limit(10).all()

    return render_template('dashboard.html', predictions=predictions)

@app.route('/logout')
def logout():
    """Logout route."""
    logout_user()
    return redirect(url_for('home'))

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    """API endpoint for news analysis."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if len(text) < 10:
            return jsonify({'error': 'Text too short. Please provide at least 10 characters.'}), 400

        # Make prediction
        result = predictor.predict_single(text)

        # Save prediction to database with probability scores
        prediction = Prediction(
            text=text,
            prediction=result['prediction'],
            confidence=result['confidence'],
            probability_real=result['probability_real'],
            probability_fake=result['probability_fake'],
            model_used=result['model_used'],
            user_id=current_user.id
        )

        db.session.add(prediction)
        db.session.commit()

        return jsonify({
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 4),
            'probability_real': round(result['probability_real'], 4),
            'probability_fake': round(result['probability_fake'], 4),
            'model_used': result['model_used'],
            'prediction_id': prediction.id
        })

    except Exception as e:
        print(f"Error in analyze: {e}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

@app.route('/analyze-multiple', methods=['POST'])
@login_required
def analyze_multiple():
    """API endpoint for analyzing with multiple models."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Get predictions from all models
        results = predictor.predict_multiple_models(text)
        
        if not results:
            return jsonify({'error': 'No models available for prediction'}), 500

        # Save the primary prediction (Logistic Regression) to database
        primary_result = results[0]  # Now results is a list
        prediction = Prediction(
            text=text,
            prediction=primary_result['prediction'],
            confidence=primary_result['confidence'],
            probability_real=primary_result['probability_real'],
            probability_fake=primary_result['probability_fake'],
            model_used='Multiple Models Ensemble',
            user_id=current_user.id
        )
        db.session.add(prediction)
        db.session.commit()

        # Format results for response
        formatted_results = []
        for result in results:
            formatted_results.append({
                'model_used': result['model_used'],
                'prediction': result['prediction'],
                'confidence': round(result['confidence'], 4),
                'probability_real': round(result['probability_real'], 4),
                'probability_fake': round(result['probability_fake'], 4)
            })

        return jsonify({
            'predictions': formatted_results,
            'primary_prediction': primary_result['prediction'],
            'prediction_id': prediction.id
        })

    except Exception as e:
        print(f"Error in analyze_multiple: {e}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

@app.route('/history')
@login_required
def history():
    """Get user's prediction history."""
    page = request.args.get('page', 1, type=int)
    per_page = 20

    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                 .order_by(Prediction.created_at.desc())\
                                 .paginate(page=page, per_page=per_page)

    return render_template('history.html', predictions=predictions)

@app.route('/api/history')
@login_required
def api_history():
    """API endpoint for prediction history."""
    page = request.args.get('page', 1, type=int)
    per_page = 20

    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                 .order_by(Prediction.created_at.desc())\
                                 .paginate(page=page, per_page=per_page)

    return jsonify({
        'predictions': [{
            'id': p.id,
            'text': p.text[:100] + '...' if len(p.text) > 100 else p.text,
            'prediction': p.prediction,
            'confidence': p.confidence,
            'model_used': p.model_used,
            'created_at': p.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } for p in predictions.items],
        'total_pages': predictions.pages,
        'current_page': predictions.page,
        'has_next': predictions.has_next,
        'has_prev': predictions.has_prev
    })

@app.route('/delete-history/<int:prediction_id>', methods=['DELETE'])
@login_required
def delete_history(prediction_id):
    """Delete a specific prediction from history."""
    try:
        print(f"Delete request for prediction {prediction_id} from user {current_user.id}")
        
        prediction = Prediction.query.filter_by(
            id=prediction_id,
            user_id=current_user.id
        ).first()

        if not prediction:
            print(f"Prediction {prediction_id} not found or belongs to different user")
            return jsonify({'success': False, 'error': 'Prediction not found'}), 404

        db.session.delete(prediction)
        db.session.commit()
        
        print(f"Successfully deleted prediction {prediction_id}")
        return jsonify({'success': True, 'message': 'Prediction deleted successfully'})

    except Exception as e:
        db.session.rollback()
        print(f"Error deleting prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Failed to delete prediction'}), 500

@app.route('/stats')
@login_required
def stats():
    """User statistics page."""
    # Get user's prediction statistics
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()

    real_predictions = Prediction.query.filter_by(user_id=current_user.id, prediction='Real News').count()
    fake_predictions = Prediction.query.filter_by(user_id=current_user.id, prediction='Fake News').count()

    # Get recent activity (last 30 days)
    from datetime import datetime, timedelta
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_predictions = Prediction.query.filter(
        Prediction.user_id == current_user.id,
        Prediction.created_at >= thirty_days_ago
    ).count()

    return render_template('stats.html',
                         total_predictions=total_predictions,
                         real_predictions=real_predictions,
                         fake_predictions=fake_predictions,
                         recent_predictions=recent_predictions)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """404 error handler."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler."""
    db.session.rollback()
    return render_template('500.html'), 500

# Create database tables
def create_tables():
    """Create database tables."""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

if __name__ == '__main__':
    create_tables()
    print("🚀 Starting Fake News Detection Flask Application...")
    print("📱 Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)