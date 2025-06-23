# 🌾 Smart Agriculture Analysis & Crop Recommendation

## Project Description

Welcome to the **Smart Agriculture Analysis & Crop Recommendation System**! This interactive web application is designed to empower farmers, agronomists, and agricultural enthusiasts with data-driven insights to optimize crop selection and management. Built using Streamlit, this platform offers a comprehensive suite of tools for:

* **Understanding Agricultural Data**: Gain deep insights into your soil and climatic conditions.
* **Intelligent Crop Recommendation**: Leverage machine learning to predict the most suitable crops based on specific environmental parameters.
* **Proactive Environmental Forecasting**: Anticipate future trends in key environmental factors to plan more effectively.

Our goal is to foster sustainable and more profitable farming practices by making advanced agricultural analytics accessible and user-friendly.

---

## ✨ Key Features

* **📊 Comprehensive Data Upload & Preprocessing**:
    * Effortless upload of CSV or Excel datasets.
    * Automated data cleaning, missing value imputation, and data type validation to ensure robust analysis.
* **🔍 Intuitive Exploratory Data Analysis (EDA)**:
    * Visualize feature distributions (histograms, count plots) for a quick overview.
    * Generate correlation matrices to understand relationships between numerical features.
    * Create custom scatter, box, and violin plots to explore specific variable interactions.
* **🤖 Robust Machine Learning for Crop Recommendation**:
    * **Model Comparison**: Train and evaluate multiple classification models including Random Forest, Logistic Regression, and Support Vector Machines to find the best fit for your data.
    * **Performance Metrics**: Access detailed evaluation reports, including Accuracy, Precision, Recall, and a comprehensive Classification Report for each crop type.
    * **Confusion Matrix**: Visualize model performance and identify misclassifications.
    * **Feature Importance**: Understand which soil and climate factors are most influential in determining crop suitability.
* **🚀 Interactive Crop Prediction & Insights**:
    * Input custom soil nutrient levels (N, P, K), pH, temperature, humidity, and rainfall to get instant crop recommendations.
    * View prediction confidence scores (probabilities) for all possible crop types, aiding in decision-making.
* **📈 Time-Series Forecasting for Environmental Factors**:
    * Utilize ARIMA (AutoRegressive Integrated Moving Average) models to forecast future trends of crucial environmental parameters like temperature, humidity, or rainfall.
    * Analyze Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots to help in selecting optimal ARIMA model orders (p, d, q).
    * Review forecast metrics (MAE, RMSE, MAPE) to gauge model accuracy on historical data.
 


📋 Dataset Requirements
For the application to function optimally and provide accurate recommendations, your input dataset should include the following types of columns. The application provides an interface to map your dataset's specific column names to these required features.

Soil Nutrient Levels:

Nitrogen (N): Numeric values representing nitrogen content.
Phosphorus (P): Numeric values representing phosphorus content.
Potassium (K): Numeric values representing potassium content.
Soil Properties:

pH Value: Numeric values representing the soil's pH level (typically 0-14).
Climatic Factors:

Temperature: Numeric values for average temperature (e.g., in Celsius).
Humidity: Numeric values for relative humidity (e.g., as a percentage).
Rainfall: Numeric values for rainfall (e.g., in mm).
Target Variable (Crop Type):

Crop Type/Name: Categorical (textual) values representing the name of the crop (e.g., "Rice", "Maize", "Wheat", "Coffee", "Mango"). This is the variable the model will learn to predict.
Optional for Time-Series Forecasting:

Date/Time Column: A column containing dates or timestamps (e.g., Date, Timestamp, Observation_Date). This is crucial for time-series analysis. If this column is not provided or cannot be parsed, the application will use the row order as a proxy for time, which might not reflect true temporal patterns.
Example Dataset:
A small sample dataset (e.g., sample_data.csv or sample_data.xlsx) can be found in the data/ directory of this repository. You can use this to quickly test the application's functionalities.

💡 Usage Tips
Initial Data Load: Upon starting the app, use the sidebar's file uploader to load your dataset.
Column Mapping is Key: Navigate to the "Data & Preprocessing" tab and carefully select the correct columns from your uploaded data that correspond to the required factors. This step is critical for accurate analysis.
Experiment with Models: In the "Model Training & Evaluation" tab, feel free to compare different machine learning models or train a single one to see their performance characteristics.
Forecast with Caution: For time-series forecasting, ensure your date column is well-formatted and that you understand how ARIMA orders (p,d,q) are selected using ACF/PACF plots.


## 🚀 How to Run the Application

After completing the installation, you can launch the Streamlit application:

```bash
streamlit run app.py


📄 License
This project is open-source and licensed under the MIT License. See the LICENSE file in the repository for full details.

📧 Contact
For any questions, feedback, or collaborations, please feel free to connect!
