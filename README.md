🤖 AI/ML Studio Pro

A complete no-code machine learning platform built with Streamlit that enables users to train, compare, and deploy machine learning models without writing any code.

✨ Features

📥 Data Import
  - Upload CSV files
  - Import datasets from URLs
  - Automatic data preprocessing

🎯 Supervised Learning
  - Classification Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, SVM, Naive Bayes, K-Nearest Neighbors
  - Regression Models: Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, SVR, K-Nearest Neighbors

🔍 Clustering Analysis
  - K-Means
  - DBSCAN
  - Agglomerative Clustering
  - Gaussian Mixture
  - Mean Shift

📝 NLP Tasks
  - Text Classification
  - Sentiment Analysis
  - Topic Modeling (LDA)
  - TF-IDF and Count Vectorization

📊 Model Comparison
  - Interactive visualizations
  - Performance metrics comparison
  - Export results to CSV

🧪 Testing & Deployment
  - Real-time predictions
  - Model performance metrics
  - Easy-to-use interface

🚀 Installation

1. Clone this repository or download the files

2. Install the required dependencies:

pip install -r requirements.txt


💻 Usage

1. Start the application:

streamlit run "no code ml.py"


2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Follow the workflow:
   - Import Data: Upload your CSV file or import from a URL
   - Choose Task: Select Supervised Learning, Clustering, or NLP
   - Train Models: Select algorithms and train them on your data
   - Compare: Analyze model performance with interactive charts
   - Test: Make predictions on new data

📋 Requirements

- Python 3.8 or higher
- streamlit==1.31.0
- pandas==2.2.0
- numpy==1.26.3
- scikit-learn==1.4.0
- plotly==5.18.0
- requests==2.31.0

📊 Supported Data Formats

- CSV files with headers
- Numerical and categorical features
- Text data for NLP tasks

🎨 Features in Detail

Data Preprocessing
- Automatic handling of missing values
- Label encoding for categorical variables
- Feature scaling with StandardScaler
- Train-test split with customizable ratios

Model Training
- Multiple models can be trained simultaneously
- Automatic hyperparameter configuration
- Progress tracking during training
- Model persistence in session state

Evaluation Metrics

Classification:
- Accuracy
- Precision
- Recall
- F1 Score

Regression:
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

Clustering:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

🎯 Example Workflow

1. Import Data: Upload iris.csv
2. Select Supervised Learning → Classification
3. Choose Target: Select the species column
4. Select Features: Choose all other columns
5. Train Models: Select Random Forest and Logistic Regression
6. Compare: View performance metrics and visualizations
7. Test: Make predictions on new flower measurements

🛠️ Troubleshooting

Issue: Models not training
- Ensure your dataset has no missing target values
- Check that feature columns are selected

Issue: Import errors
- Verify all dependencies are installed: pip install -r requirements.txt
- Check Python version compatibility

Issue: Application not loading
- Clear browser cache
- Restart the Streamlit server
- Check for port conflicts

📝 Notes

- The application stores trained models in session state
- Closing the browser tab will clear all trained models
- Large datasets may take longer to process
- For best results, ensure your data is clean and properly formatted

🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

📄 License

This project is open source and available for educational and commercial use.

🙏 Acknowledgments

Built with:
- Streamlit (https://streamlit.io/) - Web framework
- Scikit-learn (https://scikit-learn.org/) - Machine learning library
- Plotly (https://plotly.com/) - Interactive visualizations

---

Made with ❤️ using Streamlit
