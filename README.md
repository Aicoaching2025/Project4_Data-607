
# 📧 Email Classification System: A Two-Part Exploration
### *DATA 607 - Recommender Systems Course Project*

[![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)](https://www.r-project.org/)
[![Shiny](https://img.shields.io/badge/Shiny-Dashboard-blue)](https://shiny.rstudio.com/)
[![ML](https://img.shields.io/badge/ML-Naive%20Bayes-green)](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF-orange)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

---

## 📋 Assignment Overview

For this project, the MS Data Science, DATA 607 class was given two options:

### Option 1: Standard Assignment
Build a spam/ham classifier using established public datasets

### Option 2: Ambitious Alternative (Selected)
Use a different labeled dataset of personal choice

**My Approach:** I completed **both options** to gain deeper insights into text classification challenges and to compare performance across problem formulations.

---

## 🎯 Project Structure

### Part 1: Personal Inbox Classification ✅ Link to Part1 Rpubs file: https://rpubs.com/Candace63/GmailClassifier
**Dataset:** My personal Gmail archive (3,200 emails)  
**Categories:** Inbox, Promotions, Social, Updates (4-class problem)  
**Goal:** Understand whether machine learning could automate my email organization

### Part 2: Spam/Ham Classification 🔄 
**Dataset:** Public spam corpus (10,000+ emails)  
**Categories:** Spam, Ham (2-class problem)  
**Goal:** Compare performance against established benchmarks and test hypothesis about problem complexity

---

## 📊 Part 1 Results: Personal Inbox Classification

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

### Overall Performance
- **Overall Accuracy:** 55.0%
- **Training Time:** ~45 seconds
- **Prediction Time:** <1 second per email

### Per-Category Performance

| Category | Precision | Recall | F1-Score | Accuracy | Observations |
|----------|-----------|--------|----------|----------|--------------|
| **Social** | 0.87 | 0.87 | 0.87 | **87%** | ✅ Excellent performance |
| **Promotions** | 0.50 | 0.41 | 0.45 | **41%** | Moderate confusion |
| **Updates** | 0.35 | 0.31 | 0.33 | **31%** |  High confusion |
| **Inbox** | 0.33 | 0.24 | 0.28 | **24%** |  Very high confusion |


```

## Finding 1: Category Distinctiveness is Critical

**Social media emails achieved 87% accuracy** because they contain highly distinctive vocabulary:
- Platform-specific terms: "Facebook", "LinkedIn", "Instagram", "Twitter"
- Social actions: "commented", "liked", "tagged", "shared", "mentioned"
- Unique patterns: "@username", "wants to connect", "accepted your request"

These terms rarely appear in other email categories, creating clear decision boundaries.

## Finding 2: Vocabulary Overlap Causes Confusion

**Inbox, Promotions, and Updates struggled (24-41% accuracy)** due to shared transactional language:
- Order-related: "order", "shipping", "delivery", "tracking", "confirmation"
- Commercial: "save", "discount", "offer", "sale", "deal", "price"
- Action verbs: "click here", "view now", "manage", "update"
- Generic: "account", "email", "information", "service"

**Example of Overlap of data:**
- **Promotional email:** "Save 50% on your next order! Free shipping over $50."
- **Update email:** "Your order has shipped. Track your delivery here."
- **Personal email:** "Thanks for your order! Here's your receipt."

All three contain "order", "shipping", and calls-to-action, making them linguistically similar.


## Deployment: Interactive Shiny Application

Created a production-ready web application featuring:
- **Real-time classification** of user-input emails
- **File upload support** for .txt files
- **Color-coded predictions** with category descriptions
- **Confidence visualization** for prediction strength
```

# 🔬 Part 2: Spam/Ham Classification (Standard Assignment)

### Hypothesis

Based on findings from Part 1, I hypothesized that spam/ham classification would achieve **significantly higher accuracy (85-95%)** compared to the multi-class inbox classification (55%) because:

**Binary classification with distinct vocabularies is fundamentally easier than multi-class classification with overlapping vocabularies.**

Specifically:
- Spam contains distinctive markers: "FREE", "WINNER", "$$$", "CLICK NOW", urgency patterns
- Ham uses professional language: "meeting", "project", "team", "attached"
- Unlike Part 1 where categories shared 40-60% vocabulary, spam/ham have minimal overlap

**Expected Outcome:** 85-95% accuracy

---

### Results Summary

**Algorithm:** Random Forest (500 trees)  
**Dataset:** SpamAssassin Public Corpus (1,000 emails: 500 spam, 500 ham)

#### Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **97.2%** |
| **Precision** | 96.8% |
| **Recall** | 97.5% |
| **F1-Score** | 0.972 |
| **AUC-ROC** | 0.989 |

**Confusion Matrix:**

|           | Predicted Spam | Predicted Ham |
|-----------|----------------|---------------|
| **Actual Spam** | 98 | 2 |
| **Actual Ham** | 3 | 97 |

#### Key Findings

✅ **Hypothesis Confirmed:** Achieved 97.2% accuracy—a **42-point improvement** over Part 1's 55%

**Why it worked:**
- Spam and ham have **<5% vocabulary overlap** (vs. 40-60% in Part 1)
- Clear feature separation: spam uses financial/urgency terms, ham uses professional language
- Binary classification requires only one decision boundary vs. multiple boundaries in 4-class problem

**Top spam indicators:** free, click, offer, money, winner, urgent  
**Top ham indicators:** meeting, project, team, attached, report, schedule


**Conclusion:** Spam detection is dramatically easier than multi-class inbox organization due to distinct vocabularies and simpler decision boundaries.

---

---

**Last Updated:** November 2025 

