World Important Events – Ancient to Modern

Data exploration, sentiment analysis, and predictive modeling on historical events dataset

1. Overview

This project analyzes a dataset of world important events across history. The workflow includes:

Data exploration & preprocessing

Visualization of trends (countries, populations, event descriptions)

Sentiment analysis using VADER

Calculation of an Impact Score (affected vs. total population)

Predictive modeling with multiple ML algorithms

The aim is to demonstrate how informatics methods can be applied to large, descriptive datasets for insights and predictive analytics.

2. Dataset

The dataset (Project_data.xlsx) contains:

Country – location of the event

Affected_Population – count of affected individuals

Total_Population – reference population

Text_description – narrative of the event

Additional fields related to event impact

3. Project Files

Intro_Informatics_Project.ipynb – main Jupyter Notebook with full workflow

Project_data.xlsx – input dataset

wordcloud.png – generated word cloud of event descriptions

impact_score_plot.png – log-log regression of population vs. affected

evaluation_results.json – ML performance metrics

README.md – project documentation

4. Requirements

Install dependencies with:

pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud


Also download the VADER lexicon for sentiment analysis:

import nltk
nltk.download('vader_lexicon')

5. Usage
A) Load and Inspect Dataset
import pandas as pd

df = pd.read_excel("Project_data.xlsx")
print(df.head())
df.info()

B) Explore Distributions
# Top 20 countries by number of events
import seaborn as sns
import matplotlib.pyplot as plt

country_counts = df['Country'].value_counts().nlargest(20)
plt.figure(figsize=(10,6))
sns.barplot(x=country_counts.values, y=country_counts.index)
plt.title("Top 20 Countries by Number of Events")
plt.show()

C) Word Cloud of Event Descriptions
from wordcloud import WordCloud

text = " ".join(str(desc) for desc in df["Text_description"])
wordcloud = WordCloud(background_color="white").generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for Event Descriptions")
plt.show()

D) Impact Score & Sentiment
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer

df["Population_Affected"] = pd.to_numeric(df["Population_Affected"], errors="coerce")
df["Total_Population"] = pd.to_numeric(df["Total_Population"], errors="coerce")

# Impact score
df["Impact Score"] = (df["Population_Affected"] / df["Total_Population"]) * 100

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
df["Sentiment"] = df["Text_description"].astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])

E) Predictive Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dummy outcome for demonstration
df["Outcome"] = np.random.choice([0,1], size=len(df))
X = df[["Impact Score", "Sentiment"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

6. Results

Exploration: Identified top countries & populations most represented in the dataset

Word Cloud: Highlighted frequent terms in historical descriptions

Impact Score: Quantified population-level effect of events

Sentiment Analysis: Assigned polarity values to event narratives

Predictive Models: Logistic Regression, SVM, Decision Tree, Random Forest tested for classification tasks

7. Future Work

Use real historical labels instead of dummy outcomes for classification

Expand visualizations with interactive dashboards (Plotly, Power BI, Tableau)

Apply advanced NLP (topic modeling, transformers) for event classification

8. Author

Sai Srilekha Aluru
MS Health Informatics | Pharm D
📧 saisrilekhaaluru@gmail.com

🔗 LinkedIn
