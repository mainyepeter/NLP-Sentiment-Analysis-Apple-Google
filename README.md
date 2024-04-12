

<img src="image-1.png" alt="alt text" width="800">

# **Sentiment Analysis of Tweets on Tech Brands: Apple And Google**

# Group Members
1. Joyce Chepng'eno
2. Meshael Oduor
3. Peter Mainye
4. Ronald Kipng'etich
5. Milton Kabute
6. Esther Omulokoli


# Project Overview
The project aimed to perform sentiment analysis on tweets related to tech brands such as Apple and Google. The goal was to develop a model capable of accurately classifying the sentiment expressed in tweets as positive, negative, or neutral.

# **Business Understanding**
Implementing sentiment analysis for tech brands offers valuable insights into customer perceptions, preferences, and sentiments expressed on social media platforms. By leveraging these insights, businesses can make data-driven decisions to enhance brand reputation, improve customer satisfaction, and gain a competitive edge in the market. The primary objective of implementing sentiment analysis for tech brands is to gain actionable insights into customer perceptions, opinions, and sentiments expressed on social media platforms. By analyzing tweets related to tech brands such as Apple and Google, businesses can understand how their products are being perceived by the public, identify areas for improvement, and make informed decisions to enhance brand reputation and customer satisfaction.

# Research Question

    What is the overall sentiment towards our brand/products?
Understanding the general sentiment can help assess brand perception and identify potential areas for improvement or areas of strength.
    Which specific products or features are receiving positive/negative feedback?
Identifying sentiments towards specific products or features allows businesses to prioritize areas for enhancement or capitalize on strengths.
    How does sentiment vary across different customer segments or demographics?
Analyzing sentiment variations across different customer segments provides insights into audience preferences and helps tailor marketing strategies accordingly.
    Are there any emerging trends or topics driving sentiment?
Monitoring emerging trends and topics influencing sentiment enables businesses to stay ahead of market shifts and adapt their strategies accordingly.
    How does our brand sentiment compare to competitors?
Benchmarking brand sentiment against competitors helps identify competitive advantages and areas where improvements are needed.



# **Problem Statement**
We aim to address the need for actionable insights into customer sentiments towards tech brands, particularly Apple and Google, as expressed on social media platforms. Despite the abundance of user-generated content on platforms like Twitter, businesses often struggle to extract meaningful insights from this data due to its unstructured nature and sheer volume. By addressing the challenges associated with extracting insights from social media data, our proposed solution aims to empower businesses in the tech industry to gain a deeper understanding of customer sentiments, identify areas for improvement, and make data-driven decisions to drive growth and maintain competitive advantage.

# Main Objective
The main objective of the project is to analyze the sentiment of tweets towards various brands and products. By categorizing the sentiment as positive, negative, or neutral and identifying the specific targets of emotions, the aim is to gain insights into consumer perceptions and attitudes.

# Specific Objectives
Classify each tweet as expressing positive, negative, or neutral sentiments towards brands and products.
Identify the specific brands or products targeted by the emotional content in each tweet.
Analyze the overall sentiment distribution across brands and products.
Explore patterns and trends in consumer sentiment over time or in response to specific events or marketing campaigns.


# **Data Understanding**
The dataset contains 9093 rows of tweets related to multiple brands and products. Each tweet is annotated by contributors to indicate the sentiment expressed (positive, negative, or neutral) and specify the brand or product targeted by the emotion. The dataset used for this project was sourced from CrowdFlower via data.world. It contained 9000 tweets mentioning various tech brands and their associated sentiments.

# Data Preparation
Data preparation is a crucial step in any data analysis or machine learning project, ensuring that the data is in a suitable format for analysis and modeling. In this project, the data preparation involved several steps, including loading the data, handling missing values, cleaning text data, and renaming columns to improve readability and consistency.

# Loading the Data
The project started with loading the dataset containing tweets mentioning tech brands like Apple and Google from a CSV file into a pandas DataFrame. The pd.read_csv() function was used for this purpose.

Utilize Natural Language Processing techniques to construct a machine learning model for automated sentiment analysis of tweets related to Google and Apple products.
Collect and analyze sentiment data from Twitter regarding Apple and Google stocks.
Augment traditional financial analysis with sentiment analysis to identify potential investment opportunities or risks.
Use sentiment analysis to inform investment decisions, providing an alternative perspective on which stocks to invest in based on social sentiment.

# **Data Source**
The dataset, sourced from CrowdFlower via data.world, comprises over 9,000 tweets with sentiment ratings labeled as positive, negative, or neutral. The tweets were collected during the South by Southwest conference, primarily discussing Google and Apple products.

# **Data Description**
The dataset contains three columns:

tweet: Contains the text of the tweet, facilitating sentiment analysis based on the words.

product: Indicates whether the expressed emotion pertains to a specific product.

sentiment: Serves as a quick indicator of brand-related sentiment, allowing for efficient initial filtering of relevant data.

# **Data Preparation**

1. **Loading Data:** The dataset containing tech brand tweets was loaded using pd.read_csv().
2. **Handling Missing Values:** Missing values were addressed using appropriate strategies, such as imputation or removal.
3. **Renaming Columns:** Column names were simplified for clarity and consistency.
4. **Text Cleaning:** Text data underwent cleaning to remove noise, punctuation, and irrelevant information.
5. **Feature Engineering:** Additional features were engineered, like sentiment score, to enhance analysis and modeling. Sentiments were categorized as positive, negative, or neutral, dropping ambiguous entries.

# **EDA (Exploratory Data Analysis)**
The EDA focused on gaining insights into customer sentiments towards Google and Apple products. Key findings include the majority of tweets expressing neutral sentiment and the necessity for data cleaning due to missing values.

# Counts of Apple and Google Products

![alt text](image-2.png)

# Products Distribution 

![alt text](image-3.png)

# Sentiments 

![alt text](image-4.png)

# Most Common Words in the Tweets

![alt text](image-5.png)

# Cloudword in the Tweet Column

![alt text](image-6.png)

# Distribution of the Sentiment Classes

![alt text](image-7.png)

# Visualisation of the best perfoming Models

![alt text](image-8.png)

# Visualisation of Random Forest Model

![alt text](image-9.png)

# Visualisation of Naive Bayes Classification

![alt text](image-10.png)

# Visualisation of the Confusion Matrix

![alt text](image-11.png)

# Visualisation of Multiclass  
# SVM

![alt text](image-12.png)

# Naive Bayes
![alt text](image-13.png)

# Random Forest
![alt text](image-14.png)

### Model Evaluation 
# SVM
![alt text](image-15.png)

# Naive Bayes
![alt text](image-16.png)

# Random Forest
![alt text](image-17.png)

# **Modeling**
Trained multiple machine learning models, including Logistic Regression, Support Vector Machines (SVM), Naive Bayes, Random Forest, and Gradient Boosting.
Applied hyperparameter tuning using techniques like Grid Search to optimize model performance.
Utilized ensemble methods like Random Forest to improve classification accuracy.
Experimented with multi-class classification using Logistic Regression and evaluated performance metrics.


# **Model Evaluation**
Evaluated models using standard classification metrics such as accuracy, precision, recall, and F1-score.
Conducted cross-validation to assess model generalization performance.
Plotted ROC curves and calculated AUC scores to evaluate binary classification models.
Compared the performance of different models to select the best-performing one.


# **Results & Insights**
Model Performance:

1. **Naive Bayes Model:**
   - Accuracy: 84.5%
   - Precision: 85.2%
   - Recall: 98.3%
   - F1 Score: 91.3%
   - ROC AUC Score: 57.8%
   
2. **Random Forest Model:**
   - Accuracy: 86.3%
   - Precision: 86.4%
   - Recall: 99.1%
   - F1 Score: 92.3%
   - ROC AUC Score: 61.6%
   
3. **SVM Model:**
   - Accuracy: 85.9%
   - Precision: 85.8%
   - Recall: 99.5%
   - F1 Score: 92.1%
   - ROC AUC Score: 59.7%

Overall, the Random Forest model outperformed the Naive Bayes and SVM models in terms of accuracy, precision, and F1 score. It achieved the highest recall, indicating its ability to effectively capture positive instances. However, all models had relatively low ROC AUC scores, suggesting moderate discriminative ability in distinguishing between positive and negative classes.

# **Recommendation**
Collect more data to improve model performance, especially for less-represented brands/products.
Experiment with advanced NLP techniques like word embeddings or pre-trained language models.
Explore techniques for handling imbalanced datasets, such as oversampling or using weighted loss functions.
Implement sentiment analysis in real-time to monitor brand sentiment on social media platforms.

# Data Report
Link to the [Data Report] (https://docs.google.com/document/d/1hsGQkaLxKqXB_GnSjE_NDKT6AoRRcj14/edit#heading=h.ivmxn7lel0yk)

