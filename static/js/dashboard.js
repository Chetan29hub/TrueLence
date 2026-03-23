/**
 * Fake News Detection - Dashboard JavaScript
 * Handles news analysis functionality and UI updates
 */

document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const newsText = document.getElementById('newsText');
    const modelSelect = document.getElementById('modelSelect');
    const resultsSection = document.getElementById('resultsSection');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const charCount = document.getElementById('charCount');

    // Character count functionality
    if (newsText && charCount) {
        newsText.addEventListener('input', function() {
            const count = newsText.value.length;
            charCount.textContent = `${count}/5000`;

            if (count > 5000) {
                charCount.style.color = '#ef4444';
            } else if (count > 4500) {
                charCount.style.color = '#f59e0b';
            } else {
                charCount.style.color = '#888888';
            }
        });
    }

    // Clear button functionality
    if (clearBtn && newsText) {
        clearBtn.addEventListener('click', function() {
            newsText.value = '';
            if (charCount) charCount.textContent = '0/5000';
            hideResults();
        });
    }

    // Analyze button functionality
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', function() {
            const text = newsText.value.trim();

            if (!text) {
                showAlert('Please enter some text to analyze.', 'warning');
                return;
            }

            if (text.length < 10) {
                showAlert('Please enter at least 10 characters for analysis.', 'warning');
                return;
            }

            if (text.length > 5000) {
                showAlert('Text is too long. Please limit to 5000 characters.', 'warning');
                return;
            }

            const modelType = modelSelect.value;
            analyzeText(text, modelType);
        });
    }

    // Enter key to analyze
    if (newsText) {
        newsText.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                analyzeBtn.click();
            }
        });
    }
});

function analyzeText(text, modelType) {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');

    // Show loading state
    showLoading(analyzeBtn, '<i class="fas fa-search"></i> Analyzing...');
    showSpinner();

    const endpoint = modelType === 'multiple' ? '/analyze-multiple' : '/analyze';

    makeRequest(endpoint, {
        method: 'POST',
        body: { text: text }
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }

        hideSpinner();

        if (modelType === 'multiple') {
            displayMultipleResults(data);
        } else {
            displaySingleResult(data);
        }

        showResults();
        showAlert('Analysis completed successfully!', 'success');
    })
    .catch(error => {
        hideSpinner();
        console.error('Analysis error:', error);
        showAlert(error.message || 'Analysis failed. Please try again.', 'error');
    })
    .finally(() => {
        hideLoading(analyzeBtn, '<i class="fas fa-search"></i> Analyze News');
    });
}

function displaySingleResult(data) {
    const predictionBadge = document.getElementById('predictionBadge');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidencePercent = document.getElementById('confidencePercent');
    const probReal = document.getElementById('probReal');
    const probFake = document.getElementById('probFake');
    const modelUsed = document.getElementById('modelUsed');

    // Update prediction badge
    const isReal = data.prediction === 'Real News';
    predictionBadge.className = `prediction-badge ${isReal ? 'real' : 'fake'}`;
    predictionBadge.innerHTML = `<span id="predictionText">${data.prediction}</span>`;

    // Update confidence meter
    const confidencePercentValue = Math.round(data.confidence * 100);
    confidenceFill.style.width = `${confidencePercentValue}%`;
    confidenceFill.className = `confidence-fill ${isReal ? 'real' : 'fake'}`;
    confidencePercent.textContent = `${confidencePercentValue}%`;

    // Update probabilities
    probReal.textContent = `${(data.probability_real * 100).toFixed(1)}%`;
    probFake.textContent = `${(data.probability_fake * 100).toFixed(1)}%`;

    // Update model info
    modelUsed.textContent = data.model_used;

    // Show single result, hide multiple
    document.getElementById('singleResult').style.display = 'block';
    document.getElementById('multipleResults').style.display = 'none';
}

function displayMultipleResults(data) {
    const modelComparison = document.getElementById('modelComparison');
    modelComparison.innerHTML = '';

    // Handle both old format (object) and new format (array in predictions property)
    const predictions = data.predictions || Object.entries(data);
    
    if (Array.isArray(data.predictions)) {
        // New format from Flask
        data.predictions.forEach(result => {
            const isReal = result.prediction === 'Real News';
            const confidencePercent = Math.round(result.confidence * 100);

            const resultItem = document.createElement('div');
            resultItem.className = 'model-result-item';
            resultItem.innerHTML = `
                <div class="model-name">${result.model_used}</div>
                <div class="prediction ${isReal ? 'real' : 'fake'}">${result.prediction}</div>
                <div class="confidence">${confidencePercent}% confidence</div>
                <div class="probabilities">
                    <span class="prob-real">Real: ${(result.probability_real * 100).toFixed(1)}%</span>
                    <span class="prob-fake">Fake: ${(result.probability_fake * 100).toFixed(1)}%</span>
                </div>
            `;
            modelComparison.appendChild(resultItem);
        });
    } else {
        // Old format (object)
        Object.entries(data).forEach(([modelName, result]) => {
            if (result.model_used) {  // Skip if it's not a prediction object
                const isReal = result.prediction === 'Real News';
                const confidencePercent = Math.round(result.confidence * 100);

                const resultItem = document.createElement('div');
                resultItem.className = 'model-result-item';
                resultItem.innerHTML = `
                    <div class="model-name">${result.model_used}</div>
                    <div class="prediction ${isReal ? 'real' : 'fake'}">${result.prediction}</div>
                    <div class="confidence">${confidencePercent}% confidence</div>
                    <div class="probabilities">
                        <span class="prob-real">Real: ${(result.probability_real * 100).toFixed(1)}%</span>
                        <span class="prob-fake">Fake: ${(result.probability_fake * 100).toFixed(1)}%</span>
                    </div>
                `;
                modelComparison.appendChild(resultItem);
            }
        });
    }

    // Show multiple results, hide single
    document.getElementById('singleResult').style.display = 'none';
    document.getElementById('multipleResults').style.display = 'block';
}

function showResults() {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    // Add cool animation
    resultsSection.classList.remove('bounce-in');
    resultsSection.classList.remove('fade-in');
    void resultsSection.offsetWidth; // trigger reflow
    resultsSection.classList.add('bounce-in');
    
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideResults() {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'none';
}

function showSpinner() {
    const loadingSpinner = document.getElementById('loadingSpinner');
    loadingSpinner.style.display = 'block';
}

function hideSpinner() {
    const loadingSpinner = document.getElementById('loadingSpinner');
    loadingSpinner.style.display = 'none';
}

function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => alert.remove());

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
        ${message}
    `;

    const dashboardGrid = document.querySelector('.dashboard-grid');
    if (dashboardGrid && dashboardGrid.parentNode) {
        dashboardGrid.parentNode.insertBefore(alertDiv, dashboardGrid);
    } else {
        const container = document.querySelector('.container') || document.body;
        container.insertBefore(alertDiv, container.firstChild);
    }

    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}