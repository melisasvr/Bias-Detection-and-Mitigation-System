import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class BiasDetectionSystem:
    def __init__(self):
        """Initialize the bias detection system with predefined bias indicators."""
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
        
        # Load spaCy model (install with: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.stop_words = set(stopwords.words('english'))
        
        # Bias indicators dictionary
        self.bias_indicators = {
            'gender': {
                'explicit': ['he', 'she', 'him', 'her', 'his', 'hers', 'man', 'woman', 'male', 'female', 
                           'boy', 'girl', 'gentleman', 'lady', 'guys', 'gals'],
                'stereotypes': ['emotional', 'aggressive', 'nurturing', 'bossy', 'hysterical', 'shrill',
                              'assertive', 'dramatic', 'moody', 'hormonal', 'irrational'],
                'professions': ['nurse', 'teacher', 'secretary', 'engineer', 'doctor', 'CEO', 'pilot',
                              'programmer', 'scientist', 'homemaker']
            },
            'race': {
                'explicit': ['black', 'white', 'asian', 'hispanic', 'latino', 'caucasian', 'african',
                           'european', 'native', 'indigenous'],
                'stereotypes': ['articulate', 'well-spoken', 'exotic', 'urban', 'ghetto', 'thuggish',
                              'primitive', 'savage', 'civilized', 'cultured'],
                'coded_language': ['inner city', 'urban youth', 'welfare queen', 'model minority',
                                 'diversity hire', 'quota']
            },
            'age': {
                'explicit': ['young', 'old', 'elderly', 'senior', 'millennial', 'boomer', 'teenager'],
                'stereotypes': ['tech-savvy', 'out of touch', 'entitled', 'lazy', 'experienced',
                              'set in ways', 'innovative', 'traditional']
            },
            'religion': {
                'explicit': ['christian', 'muslim', 'jewish', 'hindu', 'buddhist', 'atheist', 'religious'],
                'stereotypes': ['fundamentalist', 'extremist', 'devout', 'fanatical', 'godless']
            }
        }
        
        # Bias mitigation suggestions
        self.mitigation_suggestions = {
            'gendered_pronouns': {
                'he/she': 'they',
                'him/her': 'them',
                'his/her': 'their',
                'himself/herself': 'themselves'
            },
            'gendered_terms': {
                'chairman': 'chairperson',
                'mankind': 'humanity',
                'manpower': 'workforce',
                'guys': 'everyone/folks/team',
                'policeman': 'police officer',
                'fireman': 'firefighter'
            },
            'problematic_phrases': {
                'articulate for a': 'articulate',
                'surprisingly well-spoken': 'well-spoken',
                'exotic looking': 'distinctive appearance',
                'ghetto': 'low-income area',
                'primitive culture': 'traditional culture'
            }
        }

    def preprocess_text(self, text):
        """Preprocess text for analysis."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        
        return tokens

    def detect_explicit_bias(self, text, bias_type='all'):
        """Detect explicit bias indicators in text."""
        results = {}
        tokens = self.preprocess_text(text)
        
        bias_types = [bias_type] if bias_type != 'all' else self.bias_indicators.keys()
        
        for b_type in bias_types:
            if b_type in self.bias_indicators:
                found_indicators = []
                for category, indicators in self.bias_indicators[b_type].items():
                    found = [token for token in tokens if token in [ind.lower() for ind in indicators]]
                    if found:
                        found_indicators.extend([(ind, category) for ind in found])
                
                if found_indicators:
                    results[b_type] = found_indicators
        
        return results

    def analyze_word_associations(self, texts, target_words, context_window=5):
        """Analyze word associations to detect implicit bias."""
        associations = defaultdict(list)
        
        for text in texts:
            if not isinstance(text, str):
                continue
                
            tokens = word_tokenize(text.lower())
            
            for i, token in enumerate(tokens):
                if token in [tw.lower() for tw in target_words]:
                    # Get context words within window
                    start = max(0, i - context_window)
                    end = min(len(tokens), i + context_window + 1)
                    context = tokens[start:end]
                    
                    # Remove the target word itself
                    context = [w for w in context if w != token and w.isalpha()]
                    associations[token].extend(context)
        
        # Calculate association frequencies
        association_scores = {}
        for word, contexts in associations.items():
            association_scores[word] = Counter(contexts).most_common(10)
        
        return association_scores

    def detect_sentiment_bias(self, texts, demographic_groups):
        """Detect sentiment bias towards different demographic groups."""
        sentiment_scores = {}
        
        for group in demographic_groups:
            group_sentiments = []
            
            for text in texts:
                if not isinstance(text, str):
                    continue
                    
                # Check if text mentions the demographic group
                if any(term.lower() in text.lower() for term in group):
                    blob = TextBlob(text)
                    group_sentiments.append(blob.sentiment.polarity)
            
            if group_sentiments:
                sentiment_scores[str(group)] = {
                    'mean_sentiment': np.mean(group_sentiments),
                    'std_sentiment': np.std(group_sentiments),
                    'count': len(group_sentiments)
                }
        
        return sentiment_scores

    def suggest_mitigation(self, text):
        """Suggest bias mitigation for the given text."""
        suggestions = []
        modified_text = text
        
        # Check for gendered pronouns and terms
        for category, replacements in self.mitigation_suggestions.items():
            for original, replacement in replacements.items():
                if original.lower() in text.lower():
                    suggestions.append({
                        'type': category,
                        'original': original,
                        'suggestion': replacement,
                        'context': f"Consider replacing '{original}' with '{replacement}'"
                    })
                    # Apply replacement to modified text
                    modified_text = re.sub(re.escape(original), replacement, 
                                         modified_text, flags=re.IGNORECASE)
        
        return {
            'suggestions': suggestions,
            'modified_text': modified_text,
            'original_text': text
        }

    def comprehensive_bias_analysis(self, texts, output_detailed=True):
        """Perform comprehensive bias analysis on a collection of texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        results = {
            'explicit_bias': {},
            'word_associations': {},
            'sentiment_bias': {},
            'mitigation_suggestions': [],
            'summary_stats': {}
        }
        
        # 1. Explicit bias detection
        all_explicit_bias = defaultdict(list)
        for text in texts:
            bias_detected = self.detect_explicit_bias(text)
            for bias_type, indicators in bias_detected.items():
                all_explicit_bias[bias_type].extend(indicators)
        
        results['explicit_bias'] = dict(all_explicit_bias)
        
        # 2. Word association analysis for common demographic terms
        demographic_terms = ['women', 'men', 'black', 'white', 'asian', 'hispanic', 'young', 'old']
        results['word_associations'] = self.analyze_word_associations(texts, demographic_terms)
        
        # 3. Sentiment bias analysis
        demographic_groups = [
            ['women', 'female', 'she', 'her'],
            ['men', 'male', 'he', 'him'],
            ['black', 'african american'],
            ['white', 'caucasian'],
            ['asian'],
            ['hispanic', 'latino']
        ]
        results['sentiment_bias'] = self.detect_sentiment_bias(texts, demographic_groups)
        
        # 4. Generate mitigation suggestions for each text
        for i, text in enumerate(texts):
            mitigation = self.suggest_mitigation(text)
            if mitigation['suggestions']:
                results['mitigation_suggestions'].append({
                    'text_index': i,
                    'mitigation': mitigation
                })
        
        # 5. Summary statistics
        results['summary_stats'] = {
            'total_texts_analyzed': len(texts),
            'texts_with_explicit_bias': len([t for t in texts if self.detect_explicit_bias(t)]),
            'total_bias_indicators_found': sum(len(indicators) for indicators in all_explicit_bias.values()),
            'bias_types_detected': list(all_explicit_bias.keys())
        }
        
        return results

    def generate_bias_report(self, analysis_results):
        """Generate a human-readable bias analysis report."""
        report = []
        report.append("=== BIAS DETECTION AND MITIGATION REPORT ===\n")
        
        # Summary
        stats = analysis_results['summary_stats']
        report.append(f"📊 SUMMARY:")
        report.append(f"   • Total texts analyzed: {stats['total_texts_analyzed']}")
        report.append(f"   • Texts with detected bias: {stats['texts_with_explicit_bias']}")
        report.append(f"   • Total bias indicators found: {stats['total_bias_indicators_found']}")
        report.append(f"   • Bias types detected: {', '.join(stats['bias_types_detected'])}\n")
        
        # Explicit bias findings
        if analysis_results['explicit_bias']:
            report.append("🎯 EXPLICIT BIAS DETECTED:")
            for bias_type, indicators in analysis_results['explicit_bias'].items():
                report.append(f"   {bias_type.upper()}:")
                indicator_counts = Counter([ind[0] for ind in indicators])
                for indicator, count in indicator_counts.most_common(5):
                    report.append(f"     - '{indicator}': {count} occurrences")
            report.append("")
        
        # Sentiment bias
        if analysis_results['sentiment_bias']:
            report.append("💭 SENTIMENT BIAS ANALYSIS:")
            for group, scores in analysis_results['sentiment_bias'].items():
                sentiment = scores['mean_sentiment']
                sentiment_desc = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
                report.append(f"   {group}: {sentiment_desc} (score: {sentiment:.3f}, n={scores['count']})")
            report.append("")
        
        # Word associations
        if analysis_results['word_associations']:
            report.append("🔗 WORD ASSOCIATIONS:")
            for word, associations in analysis_results['word_associations'].items():
                if associations:
                    top_associations = [f"{assoc[0]} ({assoc[1]})" for assoc in associations[:3]]
                    report.append(f"   '{word}' → {', '.join(top_associations)}")
            report.append("")
        
        # Mitigation suggestions
        if analysis_results['mitigation_suggestions']:
            report.append("💡 MITIGATION SUGGESTIONS:")
            for suggestion_group in analysis_results['mitigation_suggestions'][:5]:  # Show first 5
                report.append(f"   Text {suggestion_group['text_index'] + 1}:")
                for suggestion in suggestion_group['mitigation']['suggestions']:
                    report.append(f"     • {suggestion['context']}")
            report.append("")
        
        report.append("=== END OF REPORT ===")
        
        return "\n".join(report)


# Example usage and testing
def demo_bias_detection():
    """Demonstrate the bias detection system with sample texts."""
    
    # Initialize the system
    bias_detector = BiasDetectionSystem()
    
    # Sample texts with various types of bias
    sample_texts = [
        "The nurse was very caring, as women typically are in healthcare roles.",
        "He's surprisingly articulate for someone from that neighborhood.",
        "The CEO, a strong leader and visionary, guided the company through tough times.",
        "She's probably too emotional to handle the pressure of this executive position.",
        "The young programmer was innovative, unlike the older employees who resist change.",
        "Our diverse team includes several minorities who bring unique perspectives."
    ]
    
    print("🚀 BIAS DETECTION SYSTEM DEMO\n")
    print("Sample texts to analyze:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    print("\n" + "="*60 + "\n")
    
    # Perform comprehensive analysis
    results = bias_detector.comprehensive_bias_analysis(sample_texts)
    
    # Generate and display report
    report = bias_detector.generate_bias_report(results)
    print(report)
    
    # Show detailed mitigation example
    print("\n📝 DETAILED MITIGATION EXAMPLE:")
    example_text = sample_texts[0]
    mitigation = bias_detector.suggest_mitigation(example_text)
    
    print(f"Original: {mitigation['original_text']}")
    print(f"Modified: {mitigation['modified_text']}")
    print("\nSuggestions:")
    for suggestion in mitigation['suggestions']:
        print(f"  • {suggestion['context']}")

if __name__ == "__main__":
    demo_bias_detection()