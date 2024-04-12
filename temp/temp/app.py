#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load your dataset
df = pd.read_csv('./mail_data.csv')

# Preprocess data
data = df.where((pd.notnull(df)), '')

# Map 'spam' to 0 and 'ham' to 1 in the 'Category' column
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

# Split into features (x) and target (y)
x = data['Message']
y = data['Category']

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)

# Initialize TF-IDF Vectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Transform text data into TF-IDF features
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

# Convert target to integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(x_train_features, y_train)

@app.route('/', methods=['GET', 'POST'])
def email_classifier():
    if request.method == 'POST':
        # Get the text input from the form
        input_mail = [request.form['email_text']]
        
        # Transform the input text into TF-IDF features
        input_data_features = feature_extraction.transform(input_mail)
        
        # Make prediction using the trained model
        prediction = model.predict(input_data_features)
        
        # Determine the output based on the prediction
        if prediction[0] == 1:
            result = "Ham mail"
        else:
            result = "Spam mail"
        
        return render_template('index.html', result=result)
    
    # Render the home page with the form for text input
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
