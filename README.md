How to Use
Single Text Analysis
Launch the application.
Navigate to the "Analyze Single Text Input" section.
Enter your text in the input box.
Click on "Analyze Text" to get sentiment results from both VADER and RoBERTa models.
Batch Analysis
Navigate to the "Batch Analysis from File" section.
Upload a CSV file containing text data.
Select the column containing text for analysis.
Click on "Analyze Dataset" to process the data. Results will display in a table.
Download the results as a CSV file for further use.
Models Used
VADER
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based tool specialized for social media sentiment analysis.
RoBERTa
A transformer-based deep learning model optimized for robust sentiment classification.
Sample Output
Single Text Input
Displays sentiment scores (Negative, Neutral, Positive) for both models.
Provides a bar chart comparing the scores from VADER and RoBERTa.
Batch Analysis
Processes a dataset and displays sentiment scores for each entry.
Visualizes the overall sentiment distribution using bar charts.
Dependencies
This project uses the following Python libraries:

Streamlit: For building the web application.
Pandas: For handling and analyzing data.
Matplotlib: For creating visualizations.
Transformers: For using the RoBERTa model.
VADER: For rule-based sentiment analysis.
Scipy: For softmax calculations.
