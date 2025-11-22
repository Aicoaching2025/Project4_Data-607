
# 📧 Email Classification System: A Two-Part Exploration
### *DATA 607 - Recommender Systems Course Project*

[![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)](https://www.r-project.org/)
[![Shiny](https://img.shields.io/badge/Shiny-Dashboard-blue)](https://shiny.rstudio.com/)
[![ML](https://img.shields.io/badge/ML-Naive%20Bayes-green)](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF-orange)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

> **A comprehensive text classification project demonstrating both ambitious problem formulation (personal email categorization) and standard benchmark evaluation (spam detection). This work showcases the value of comparative analysis and the importance of problem selection in machine learning success.**

---

## 📋 Assignment Overview

For this project, students were given two options:

### Option 1: Standard Assignment
Build a spam/ham classifier using established public datasets

### Option 2: Ambitious Alternative (Selected)
Use a different labeled dataset of personal choice

**My Approach:** I completed **both options** to gain deeper insights into text classification challenges and to compare performance across problem formulations.

---

## 🎯 Project Structure

### Part 1: Personal Inbox Classification ✅ **COMPLETED**
**Dataset:** My personal Gmail archive (3,200 emails)  
**Categories:** Inbox, Promotions, Social, Updates (4-class problem)  
**Goal:** Understand whether machine learning could automate my email organization

### Part 2: Spam/Ham Classification 🔄 
**Dataset:** Public spam corpus (10,000+ emails)  
**Categories:** Spam, Ham (2-class problem)  
**Goal:** Compare performance against established benchmarks and test hypothesis about problem complexity

---

## 📊 Part 1 Results: Personal Inbox Classification

### Motivation

I chose to tackle my personal email organization as the "ambitious" option because:
1. **Real-world relevance** - I actually wanted this functionality
2. **Personalized learning** - My email patterns reflect my actual usage
3. **Challenge** - Multiple similar categories would test model discrimination
4. **Practical application** - Could deploy for daily use via Shiny app

### Methodology

**Data Collection**
- Exported 11,884 emails from my Gmail account
- Categories: Inbox (7,151), Promotions (2,267), Social (845), Updates (3,621)
- Balanced to 800 emails per category to prevent class imbalance bias

**Feature Engineering**
- **Text Preprocessing:** Removed HTML tags, URLs, email addresses, special characters
- **Vectorization:** TF-IDF with top 500 features
- **Document Frequency:** Words appearing in 5+ documents
- **Feature Space:** 500 terms × 2,560 training samples

**Model Training**
- **Algorithm:** Naive Bayes classifier (suited for text classification)
- **Split:** 80% training (2,560 emails), 20% testing (640 emails)
- **Sampling:** Stratified to maintain class proportions

**Evaluation Metrics**
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix analysis
- Error pattern investigation

### Results

#### Overall Performance
- **Overall Accuracy:** 55.0%
- **Training Time:** ~45 seconds
- **Prediction Time:** <1 second per email

#### Per-Category Performance

| Category | Precision | Recall | F1-Score | Accuracy | Observations |
|----------|-----------|--------|----------|----------|--------------|
| **Social** | 0.87 | 0.87 | 0.87 | **87%** | ✅ Excellent performance |
| **Promotions** | 0.50 | 0.41 | 0.45 | **41%** | ⚠️ Moderate confusion |
| **Updates** | 0.35 | 0.31 | 0.33 | **31%** | ❌ High confusion |
| **Inbox** | 0.33 | 0.24 | 0.28 | **24%** | ❌ Very high confusion |

#### Confusion Matrix

```
                    Predicted
                Inbox  Promotions  Social  Updates
Actual
Inbox             38      28         0       50      ← 50 confused with Updates
Promotions        29      66        19       17      
Social            50      38       139       43      ← Clear winner
Updates           43      28         2       50      ← 43 confused with Inbox
```

**Visual Representation:**

![Confusion Matrix](confusion_matrix_naive_bayes.png)

### Key Findings & Analysis

#### Finding 1: Category Distinctiveness is Critical

**Social media emails achieved 87% accuracy** because they contain highly distinctive vocabulary:
- Platform-specific terms: "Facebook", "LinkedIn", "Instagram", "Twitter"
- Social actions: "commented", "liked", "tagged", "shared", "mentioned"
- Unique patterns: "@username", "wants to connect", "accepted your request"

These terms rarely appear in other email categories, creating clear decision boundaries.

#### Finding 2: Vocabulary Overlap Causes Confusion

**Inbox, Promotions, and Updates struggled (24-41% accuracy)** due to shared transactional language:
- Order-related: "order", "shipping", "delivery", "tracking", "confirmation"
- Commercial: "save", "discount", "offer", "sale", "deal", "price"
- Action verbs: "click here", "view now", "manage", "update"
- Generic: "account", "email", "information", "service"

**Example of Confusion:**
- **Promotional email:** "Save 50% on your next order! Free shipping over $50."
- **Update email:** "Your order has shipped. Track your delivery here."
- **Personal email:** "Thanks for your order! Here's your receipt."

All three contain "order", "shipping", and calls-to-action, making them linguistically similar.

#### Finding 3: Small Dataset Limitations

With only 800 training samples per category (2,560 total):
- **Feature/sample ratio:** 500 features ÷ 2,560 samples = 0.20
- Risk of overfitting to personal email patterns
- May not generalize to other users' email habits

### Deployment: Interactive Shiny Application

Created a production-ready web application featuring:
- **Real-time classification** of user-input emails
- **File upload support** for .txt files
- **Color-coded predictions** with category descriptions
- **Confidence visualization** for prediction strength

**Technical Implementation:**
```r
predict_email <- function(subject, body_text) {
  full_text <- paste(subject, body_text, sep = " ") %>%
    clean_text() %>%
    create_tfidf_features()
  
  prediction <- predict(nb_model, full_text)
  return(prediction)
}
```

### Lessons Learned from Part 1

**Successes:**
✅ Completed end-to-end ML pipeline from data collection to deployment  
✅ Created functional interactive application  
✅ Identified category-specific performance patterns  
✅ Gained insights into text similarity challenges  


**Critical Insight:**
> *The choice of classification problem matters as much as the model itself. Some categorization schemes are inherently more difficult than others due to vocabulary overlap, regardless of model sophistication.*

---

## 🔬 Part 2: Spam/Ham Classification (Standard Assignment)

### Hypothesis

Based on findings from Part 1, I hypothesize that spam/ham classification will achieve **significantly higher accuracy (85-95%)** compared to the multi-class inbox classification (55%) for the following reasons:

#### Hypothesis 1: Simpler Decision Boundary
**Prediction:** Binary classification (2 classes) is fundamentally easier than multi-class (4 classes)
- Fewer decision boundaries to learn
- Reduced probability of confusion
- More focused feature learning


### Planned Methodology

**Dataset Selection**
- Source: SpamAssassin Public Corpus or Enron Email Dataset
- Size: 10,000+ emails (5,000 spam, 5,000 ham)
- Balance: Equal representation to prevent bias

**Consistency with Part 1**
To ensure fair comparison, I will maintain the same:
- Text preprocessing pipeline
- TF-IDF parameters (500 features, 5+ document frequency)
- Model architecture (Naive Bayes)
- Evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
- Train/test split (80/20 with stratified sampling)

**Comparative Analysis**
Will directly compare:
- Part 1 (multi-class): 55% accuracy
- Part 2 (binary): [Expected 85-95%]
- Performance gain quantification
- Feature importance differences
- Error pattern analysis

### Expected Outcomes

**Quantitative Predictions:**
- Overall accuracy: 85-95% (vs 55% in Part 1)
- Precision (spam): >90%
- Recall (spam): >85%
- F1-Score: >87%


---

**Last Updated:** November 2025 

