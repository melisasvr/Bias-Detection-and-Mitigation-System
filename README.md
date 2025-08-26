# 🔍 PerspectiveAI - Bias Detection & Mitigation System
- An advanced AI-powered system for detecting and mitigating bias in text documents, including news articles, policy documents, and research papers.
- Built with Python and featuring a modern web interface for interactive analysis.

## 🎯 Features
### Multi-layered Bias Detection
- **Explicit Bias Detection**: Identifies direct mentions of demographic groups and stereotypical language
- **Implicit Bias Analysis**: Examines word associations and contextual patterns
- **Sentiment Bias Analysis**: Measures sentiment differences across demographic groups
- **Statistical Analysis**: Provides quantitative metrics and confidence scores

### Bias Categories
- 👥 **Gender Bias**: Pronouns, stereotypes, profession assumptions
- 🌍 **Racial/Ethnic Bias**: Coded language, stereotypes, implicit associations
- 📅 **Age Bias**: Generational stereotypes and assumptions
- 🕊️ **Religious Bias**: Faith-based stereotypes and language patterns

### Mitigation System
- **Alternative Phrasing**: Suggests bias-free language alternatives
- **Gender-Neutral Terms**: Provides inclusive terminology options
- **Context-Aware Recommendations**: Explains why changes are suggested
- **Before/After Comparison**: Visual diff of original vs. improved text

### Interactive Web Interface
- **Real-time Analysis**: Instant bias detection as you type
- **Visual Dashboard**: Color-coded results with progress bars and charts
- **Sample Text Library**: Pre-loaded examples for different bias types
- **Export Functionality**: Generate reports and save analysis results

## 📋 Requirements

### Python Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
nltk>=3.6
scikit-learn>=1.0.0
spacy>=3.4.0
textblob>=0.17.0
flask>=2.0.0
flask-cors>=3.0.0
```

### Additional Requirements
- Python 3.7 or higher
- spaCy English model: `python -m spacy download en_core_web_sm`

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/perspectiveai-bias-detection.git
   cd perspectiveai-bias-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

5. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 💻 Usage

### Command Line Interface

Run the basic demo:
```bash
python bias_detection.py
```

This will analyze sample texts and generate a console report showing:
- Bias indicators found
- Sentiment analysis by demographic group
- Word association patterns
- Mitigation suggestions

### Web Interface
1. **Start the web server**
   ```bash
   python simple_server.py
   ```

2. **Open your browser** to `http://localhost:8000`
3. **Analyze text**:
   - Paste your text into the input field
   - Click "Analyze for Bias"
   - Review the comprehensive visual report

### Programmatic Usage

```python
from bias_detection import BiasDetectionSystem

# Initialize the system
detector = BiasDetectionSystem()

# Analyze a single text
text = "Your text to analyze here"
results = detector.comprehensive_bias_analysis([text])

# Generate a report
report = detector.generate_bias_report(results)
print(report)

# Get mitigation suggestions
suggestions = detector.suggest_mitigation(text)
print(suggestions['modified_text'])
```

## 📊 Output Examples

### Console Output
```
=== BIAS DETECTION AND MITIGATION REPORT ===

📊 SUMMARY:
   • Total texts analyzed: 6
   • Texts with detected bias: 5
   • Total bias indicators found: 7
   • Bias types detected: gender, race, age

🎯 EXPLICIT BIAS DETECTED:
   GENDER:
     - 'emotional': 2 occurrences
     - 'nurturing': 1 occurrences

💭 SENTIMENT BIAS ANALYSIS:
   women: negative (score: -0.200, n=15)
   men: neutral (score: 0.100, n=12)
```

### Web Interface Features
- **Visual Dashboard**: Interactive charts and progress bars
- **Color-coded Tags**: Different colors for bias types (gender=pink, race=orange, age=blue)
- **Sentiment Bars**: Visual representation of sentiment bias
- **Side-by-side Comparison**: Original vs. improved text highlighting
- **Detailed Suggestions**: Context-aware mitigation recommendations

## 🎨 Web Interface Components

### Analysis Summary Dashboard
- Total bias indicators found
- Number of bias types detected  
- Available mitigation suggestions
- Overall bias detection rate

### Explicit Bias Detection
- Color-coded tags for each bias type
- Word frequency analysis
- Category classification (explicit, stereotypes, coded language)

### Sentiment Analysis
- Progress bars showing sentiment scores (-1.0 to +1.0)
- Color coding: Green (positive), Red (negative), Gray (neutral)
- Sample size indicators for statistical confidence

### Word Association Mapping
- Shows implicit bias through word co-occurrence
- Frequency counts for associated terms
- Helps identify unconscious language patterns

### Mitigation Suggestions
- Before/after text comparison
- Contextual explanations for each suggestion
- Type classification (gendered language, coded terms, etc.)

## 🔧 Configuration

### Custom Bias Indicators

You can extend the bias detection by modifying the `bias_indicators` dictionary in `BiasDetectionSystem`:

```python
self.bias_indicators['custom_category'] = {
    'explicit': ['term1', 'term2'],
    'stereotypes': ['stereotype1', 'stereotype2'],
    'coded_language': ['coded_term1', 'coded_term2']
}
```

### Mitigation Rules

Add custom mitigation suggestions:

```python
self.mitigation_suggestions['custom_replacements'] = {
    'problematic_term': 'better_alternative',
    'biased_phrase': 'neutral_phrase'
}
```

## 📁 Project Structure

```
perspectiveai-bias-detection/
├── bias_detection.py          # Core bias detection system
├── index.html                 # Web interface
├── requirements.txt           # Python dependencies
├── README.md                 # This file
```

## 🧪 Testing
Run the demo with sample texts:
```bash
python bias_detection.py
```

Test specific bias types:
```python
detector = BiasDetectionSystem()

# Test gender bias
result = detector.detect_explicit_bias("The nurse was caring, as women typically are.", 'gender')
print(result)

# Test sentiment bias
texts = ["He's a strong leader", "She's too emotional for leadership"]
sentiment_results = detector.detect_sentiment_bias(texts, [['he', 'him'], ['she', 'her']])
print(sentiment_results)
```

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📈 Roadmap
### Phase 2: Advanced Features
- [ ] Integration with BERT/transformer models
- [ ] Real-time batch processing
- [ ] Custom domain-specific bias detection
- [ ] Multi-language support
- [ ] API endpoints for external integration

### Phase 3: Enhanced Analytics
- [ ] Statistical significance testing
- [ ] Bias trend analysis over time
- [ ] Comparative analysis between documents
- [ ] Machine learning model fine-tuning

### Phase 4: Enterprise Features
- [ ] User authentication and roles
- [ ] Database integration for result storage
- [ ] Automated report generation
- [ ] Integration with document management systems

## ⚖️ Ethical Considerations
- This tool is designed to help identify potential bias, but should not be the sole determinant of bias in text. Consider:
- **Human Review**: Always have human experts review automated bias detection results
- **Context Matters**: The same words may be appropriate in some contexts but not others
- **Cultural Sensitivity**: Bias definitions may vary across cultures and communities
- **Continuous Learning**: Language and social understanding evolve; update detection rules regularly

## 📄 License
- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Built with open-source libraries: NLTK, spaCy, scikit-learn
- Inspired by research in computational linguistics and bias detection
- Special thanks to the bias research community for establishing detection methodologies

## 📞 Support
For questions, issues, or contributions:
- Create an issue on GitHub

---

**⚠️ Disclaimer**: This tool is for research and educational purposes. Results should be validated by domain experts before making decisions based on bias detection outcomes.
