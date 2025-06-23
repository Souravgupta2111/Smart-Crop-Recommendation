import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# --- Configuration ---
CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'max_data_size_mb': 50,  # Maximum file size in MB
    'max_rows': 10000,  # Maximum number of rows
    'cv_folds': 5,  # Cross-validation folds
    'forecast_points': 50,
    'pair_plot_max_features': 5,
    'default_arima_freq': 'D' # Default frequency for ARIMA if not inferred
}

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="🌾 Smart Agriculture Analysis & Crop Recommendation",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🌾 Smart Agriculture Analysis & Crop Recommendation")

# --- Helper Functions ---
def clean_column_name(name):
    """Cleans column names by removing special characters and converting to lowercase."""
    return re.sub(r'[^A-Za-z0-9_]+', '', str(name)).lower()

@st.cache_data
def load_and_clean_data(file_content, file_name):
    """Load and clean data with caching."""
    try:
        if file_name.endswith('.csv'):
            data = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        else:
            data = pd.read_excel(io.BytesIO(file_content))

        # Clean column names
        data.columns = [clean_column_name(col) for col in data.columns]
        return data, None
    except Exception as e:
        return None, f"Failed to load data: {str(e)}"

def validate_file_size(file):
    """Validate file size and data dimensions."""
    file_size_mb = len(file.getvalue()) / (1024 * 1024)

    if file_size_mb > CONFIG['max_data_size_mb']:
        return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({CONFIG['max_data_size_mb']}MB)"

    return True, "File size OK"

def validate_data_quality(data, numerical_features):
    """Validate data quality and return issues."""
    issues = []

    # Check data size
    if len(data) > CONFIG['max_rows']:
        issues.append(f"Dataset has {len(data)} rows, which exceeds the maximum of {CONFIG['max_rows']}. Consider sampling your data.")

    # Check for insufficient variation
    for col in numerical_features:
        if col in data.columns:
            if data[col].nunique() < 2:
                issues.append(f"Column '{col}' has insufficient variation (only {data[col].nunique()} unique values)")

            # Check for excessive missing values
            missing_pct = (data[col].isnull().sum() / len(data)) * 100
            if missing_pct > 50:
                issues.append(f"Column '{col}' has {missing_pct:.1f}% missing values")

    return issues

def display_data_quality_metrics(data, numerical_features):
    """Display data quality overview."""
    st.subheader("📊 Data Quality Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Samples", len(data))

    with col2:
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        st.metric("Missing Data %", f"{missing_pct:.2f}%")

    with col3:
        duplicates = data.duplicated().sum()
        st.metric("Duplicate Rows", duplicates)

    with col4:
        memory_usage = data.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Memory Usage (MB)", f"{memory_usage:.2f}")

def create_preprocessing_pipeline(numerical_features, apply_scaling):
    """Create preprocessing pipeline."""
    numerical_transformer = StandardScaler() if apply_scaling else 'passthrough'
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def safe_model_training(model, X_train, y_train):
    """Safely train model with error handling."""
    try:
        with st.spinner('Training model...'):
            model.fit(X_train, y_train)
        return model, None
    except Exception as e:
        error_msg = f"Model training failed: {str(e)}"
        if "convergence" in str(e).lower():
            error_msg += "\n💡 Try enabling feature scaling or reducing data complexity."
        elif "memory" in str(e).lower():
            error_msg += "\n💡 Try reducing dataset size or using a simpler model."
        return None, error_msg

def compare_models(models_dict, X_train, X_test, y_train_encoded, y_test_encoded):
    """Compare multiple models and return results."""
    results = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, model) in enumerate(models_dict.items()):
        status_text.text(f"Training {name}...")

        try:
            model.fit(X_train, y_train_encoded)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test_encoded, y_pred)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=CONFIG['cv_folds'])

            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
        except Exception as e:
            results[name] = {
                'accuracy': 0,
                'cv_mean': 0,
                'cv_std': 0,
                'error': str(e),
                'model': None
            }

        progress_bar.progress((i + 1) / len(models_dict))

    progress_bar.empty()
    status_text.empty()

    return results

def display_feature_importance(model, feature_names):
    """Display feature importance if available."""
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        st.subheader("🔍 Feature Importance Analysis")

        importances = model.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
            ax.set_title('Feature Importance in Crop Prediction')
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.dataframe(importance_df.style.format({'Importance': "{:.4f}"}))

        # Insights
        most_important = importance_df.iloc[0]['Feature']
        least_important = importance_df.iloc[-1]['Feature']
        st.info(f"🔑 Most influential factor: **{most_important.capitalize()}**\n\n"
                f"🔹 Least influential factor: **{least_important.capitalize()}**")

@st.cache_data
def create_visualizations(data, numerical_features, crop_type_col):
    """Create cached visualizations."""
    figures = {}

    # Distribution plots
    num_plots = len(numerical_features) + 1 # +1 for the target variable countplot
    cols_per_row = 3
    rows = int(np.ceil(num_plots / cols_per_row))

    fig_dist, axes_dist = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 6, rows * 4))
    axes_dist = axes_dist.flatten()

    for idx, ax in enumerate(axes_dist):
        if idx < len(numerical_features):
            feature = numerical_features[idx]
            sns.histplot(data[feature], kde=True, ax=ax, color='skyblue')
            ax.set_title(f'Distribution of {feature.capitalize()}')
        elif idx == len(numerical_features):
            sns.countplot(y=data[crop_type_col], ax=ax, palette='viridis',
                         order=data[crop_type_col].value_counts().index)
            ax.set_title(f'Counts of {crop_type_col.capitalize()}')
        else:
            fig_dist.delaxes(ax) # Hide unused subplots if any

    plt.tight_layout()
    figures['distributions'] = fig_dist

    # Correlation matrix
    correlation_data = data[numerical_features].copy()
    corr_matrix = correlation_data.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
               linewidths=.5, ax=ax_corr, center=0)
    ax_corr.set_title("Correlation Matrix of Numerical Features")
    figures['correlation'] = fig_corr

    # Removed the pair plot generation block
    figures['pair_plot'] = None # Explicitly set to None as it's no longer generated


    return figures

# ARIMA forecasting function
def run_arima_forecast(time_series_data_series, forecast_steps=CONFIG['forecast_points']):
    """
    Runs ARIMA forecasting on a given time-series.
    time_series_data_series: A pandas Series with a DateTime index.
    forecast_steps: Number of future steps to forecast.
    """
    if time_series_data_series.empty or not isinstance(time_series_data_series.index, pd.DatetimeIndex):
        return None, "Time series data must have a DateTime index and not be empty."

    try:
        # Plot ACF and PACF to help user determine p, d, q
        st.subheader("📚 ACF and PACF Plots for ARIMA Parameter Selection")
        st.info("Analyze these plots to help determine the (p,d,q) orders for the ARIMA model. "
                "You might need to difference your data first if it's not stationary.")

        fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
        plot_acf(time_series_data_series.dropna(), ax=ax_acf, lags=min(20, len(time_series_data_series) // 2 - 1))
        ax_acf.set_title("Autocorrelation Function (ACF)")
        st.pyplot(fig_acf)
        plt.close(fig_acf)

        fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
        plot_pacf(time_series_data_series.dropna(), ax=ax_pacf, lags=min(20, len(time_series_data_series) // 2 - 1))
        ax_pacf.set_title("Partial Autocorrelation Function (PACF)")
        st.pyplot(fig_pacf)
        plt.close(fig_pacf)
        
        st.markdown("---")
        st.markdown("#### Manually Set ARIMA Orders (p, d, q)")
        st.info("Based on the ACF/PACF plots and your understanding of the time series, set the orders below.")
        
        col_p, col_d, col_q = st.columns(3)
        with col_p:
            p_order = st.number_input("Order (p) - AR part:", min_value=0, value=1, step=1, key='arima_p')
        with col_d:
            d_order = st.number_input("Order (d) - Differencing part:", min_value=0, value=1, step=1, key='arima_d')
        with col_q:
            q_order = st.number_input("Order (q) - MA part:", min_value=0, value=1, step=1, key='arima_q')

        # Fit the ARIMA model
        with st.spinner("Fitting ARIMA model... This may take a moment."):
            model = ARIMA(time_series_data_series, order=(p_order, d_order, q_order))
            model_fit = model.fit()

        forecast_start_idx = len(time_series_data_series)
        forecast_end_idx = forecast_start_idx + forecast_steps - 1
        forecast_result = model_fit.predict(start=forecast_start_idx, end=forecast_end_idx)


        # Create a date index for the forecast
        last_date = time_series_data_series.index[-1]
        try:
            inferred_freq = pd.infer_freq(time_series_data_series.index)
        except ValueError:
            inferred_freq = CONFIG['default_arima_freq'] # Default to daily if cannot infer

        forecast_index = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=inferred_freq)[1:]
        
        # Align forecast_result with forecast_index
        if len(forecast_result) != len(forecast_index):
            st.warning("Forecast result length does not match expected index length. Adjusting forecast index.")
            forecast_index = pd.date_range(start=last_date, periods=len(forecast_result) + 1, freq=inferred_freq)[1:]


        forecast_df = pd.DataFrame({'Forecast': forecast_result}, index=forecast_index)

        in_sample_predictions = model_fit.predict(start=0, end=len(time_series_data_series) - 1)
        
        if len(time_series_data_series) != len(in_sample_predictions):
            in_sample_predictions = model_fit.fittedvalues[:len(time_series_data_series)] # Trim if longer


        mae = mean_absolute_error(time_series_data_series, in_sample_predictions)
        rmse = np.sqrt(mean_squared_error(time_series_data_series, in_sample_predictions))
        
        # Avoid division by zero if time_series_data_series has zero values for MAPE
        mape = np.mean(np.abs((time_series_data_series - in_sample_predictions) / time_series_data_series.replace(0, np.nan).dropna())) * 100 if not time_series_data_series.replace(0, np.nan).dropna().empty else float('nan')


        metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

        return forecast_df, metrics, None
    except Exception as e:
        return None, None, f"ARIMA forecasting failed: {str(e)}"

# Main Application Logic

# Sidebar
st.sidebar.header("🛠 Upload Your Dataset")
file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

# Initial welcome message if no file is uploaded
if not file:
    st.markdown("""
    ## Welcome to Smart Agriculture Analysis! 🌾

    This comprehensive platform helps you analyze agricultural data and get intelligent crop recommendations.

    ### 🚀 What you can do:

    #### 📊 **Data Analysis**
    - Upload your agricultural dataset (CSV or Excel)
    - Automatic data cleaning and preprocessing
    - Comprehensive data quality assessment
    - Rich visualizations and insights

    #### 🤖 **Machine Learning**
    - Compare multiple ML models automatically
    - Random Forest, Logistic Regression, and SVM classifiers
    - Cross-validation and performance metrics
    - Feature importance analysis

    #### 🎯 **Predictions & Insights**
    - Interactive crop recommendations
    - Sensitivity analysis for environmental factors
    - Batch processing for multiple predictions
    - Growing tips and agricultural insights

    #### 📈 **Advanced Features**
    - Model performance comparison
    - Prediction confidence scores
    - **Time-Series Forecasting for Environmental Factors (Manual ARIMA)**
    - Data-driven improvement suggestions

    ### 📂 Getting Started
    1. **Upload your dataset** using the sidebar
    2. **Map your columns** to soil and climate features
    3. **Choose your analysis mode** (compare models or select one)
    4. **Get recommendations** and insights!

    ### 📋 Data Requirements
    Your dataset should include:
    - **Soil nutrients:** Nitrogen (N), Phosphorus (P), Potassium (K)
    - **Soil properties:** pH value
    - **Climate factors:** Temperature, Humidity, Rainfall
    - **Target variable:** Crop type/name
    - **Optional for Forecasting:** A `Date` or `Time` column for time-series analysis. If not provided, forecasting will use row order as a proxy for time.

    ---
    **Ready to start?** Upload your dataset using the file uploader in the sidebar! 👆
    """)

# File processing and application tabs
if file:
    size_valid, size_msg = validate_file_size(file)
    if not size_valid:
        st.error(f"❌ {size_msg}")
        st.stop()

    file_content = file.getvalue()
    data, load_error = load_and_clean_data(file_content, file.name)

    if load_error:
        st.error(f"❌ {load_error}")
        st.stop()

    all_columns = data.columns.tolist()

    # Create tabs for the main application sections
    tab_data, tab_eda, tab_model, tab_predict, tab_ts_forecast = st.tabs(
        ["Data & Preprocessing", "Exploratory Analysis", "Model Training & Evaluation",
         "Predictions & Tools", "Time-Series Forecasting"]
    )

    with tab_data:
        st.header("⚙️ Data Loading & Preprocessing")
        display_data_quality_metrics(data, all_columns)

        st.subheader("📋 Dataset Preview")
        with st.expander("Click to view raw data preview"):
            st.dataframe(data.head())

        st.subheader("📊 Map Your Dataset Columns")
        st.markdown("Please select the columns corresponding to your soil, climate, and the crop type.")

        # Use columns for a more compact layout for mapping
        map_col1, map_col2, map_col3 = st.columns(3)

        with map_col1:
            n_col = st.selectbox("🧪 Nitrogen (N) Column", all_columns, help="Select the column representing Nitrogen content in soil.")
            p_col = st.selectbox("🧪 Phosphorus (P) Column", all_columns, help="Select the column representing Phosphorus content in soil.")
            k_col = st.selectbox("🧪 Potassium (K) Column", all_columns, help="Select the column representing Potassium content in soil.")

        with map_col2:
            temp_col = st.selectbox("🌡 Temperature Column", all_columns, help="Select the column representing Temperature.")
            hum_col = st.selectbox("💧 Humidity Column", all_columns, help="Select the column representing Humidity.")
            ph_col = st.selectbox("pH Value Column", all_columns, help="Select the column representing pH value of the soil.")

        with map_col3:
            rain_col = st.selectbox("🌧 Rainfall Column", all_columns, help="Select the column representing Rainfall.")
            crop_type_col = st.selectbox("🌱 Crop Type (Target) Column", all_columns, help="Select the column representing the type of crop (categorical target).")
            date_col = st.selectbox("🗓 Date/Time Column (Optional for Forecasting)", ['None'] + all_columns, help="Select a column for time-series analysis (e.g., 'date', 'timestamp'). If 'None', row order will be used as time steps.")

        # Validate column selection
        required_cols = [n_col, p_col, k_col, temp_col, hum_col, ph_col, rain_col, crop_type_col]
        if all(col is not None for col in required_cols) and len(set(required_cols)) == len(required_cols):
            try:
                # Define features and target
                numerical_features = [n_col, p_col, k_col, temp_col, hum_col, ph_col, rain_col]
                target_feature = crop_type_col

                # Validate data quality
                quality_issues = validate_data_quality(data, numerical_features)
                if quality_issues:
                    st.warning("⚠️ Data Quality Issues Detected:")
                    for issue in quality_issues:
                        st.write(f"• {issue}")

                    if st.button("Continue Despite Issues in Data & Preprocessing"):
                        st.info("Proceeding with analysis...")
                    else:
                        st.stop()
                else:
                    st.success("✅ Data quality looks good!")


                # Create copies to avoid warnings
                X = data[numerical_features].copy()
                y = data[target_feature].copy()

                st.markdown("---")
                st.subheader("🧹 Data Cleaning and Preprocessing Steps")

                # Handle missing values
                st.markdown("##### Handling Missing Values")
                initial_rows = len(X)

                for col in numerical_features:
                    if X[col].isnull().any():
                        mean_val = X[col].mean()
                        X[col].fillna(mean_val, inplace=True)
                        st.info(f"Filled missing values in '{col}' with mean ({mean_val:.2f})")

                # Handle missing target values
                if y.isnull().any():
                    st.warning(f"Target column '{target_feature}' has missing values. Dropping rows.")
                    y.dropna(inplace=True)
                    X = X.loc[y.index]
                    st.info(f"Removed {initial_rows - len(X)} rows due to missing target values.")

                # Data type validation
                st.markdown("##### Validating Data Types")
                for col in numerical_features:
                    if not pd.api.types.is_numeric_dtype(X[col]):
                        st.error(f"Selected feature '{col}' is not numeric. Please check your data.")
                        st.stop()
                st.info("All selected numerical features are valid.")

                # Validate target column
                if not pd.api.types.is_string_dtype(y) and not pd.api.types.is_object_dtype(y):
                    st.error(f"Target column '{target_feature}' is not categorical. Please select a column with crop names.")
                    st.stop()
                y = y.str.lower().str.strip()
                st.info(f"Target column '{target_feature}' processed successfully.")

                # Process Date/Time column if selected, or create a synthetic one
                df_ts_ready = None
                if date_col and date_col != 'None':
                    try:
                        # Attempt to parse as datetime, coercing errors to NaT
                        data['__ts_date__'] = pd.to_datetime(data[date_col], errors='coerce')
                        if data['__ts_date__'].isnull().any():
                            st.warning(f"Some values in '{date_col}' could not be parsed as dates and will be ignored for time-series analysis.")
                            data.dropna(subset=['__ts_date__'], inplace=True)
                        if not data['__ts_date__'].empty:
                            data = data.sort_values('__ts_date__').set_index('__ts_date__')
                            df_ts_ready = data[numerical_features] # Prepare data for time-series forecasting
                            st.info(f"Date/Time column '{date_col}' processed. Data ready for time-series analysis.")
                        else:
                            st.warning("No valid date entries found after parsing. Time-series analysis will be skipped.")
                            date_col = 'None' # Reset if no valid dates
                    except Exception as e:
                        st.warning(f"Could not process Date/Time column '{date_col}': {e}. Time-series analysis will be skipped.")
                        date_col = 'None' # Reset if parsing fails
                else:
                    st.info("No Date/Time column selected. Time-series forecasting will be performed using row order as a proxy for time. This may not reflect true temporal patterns.")
                    # Create a synthetic DatetimeIndex based on row numbers
                    temp_data_for_ts = data[numerical_features].copy()
                    temp_data_for_ts.index = pd.date_range(start='2023-01-01', periods=len(temp_data_for_ts), freq='D')
                    df_ts_ready = temp_data_for_ts
                    date_col = '__synthetic_date__' # Set a dummy name for internal use

                # Feature scaling option
                st.markdown("##### Feature Scaling")
                apply_scaling = st.checkbox("Apply Feature Scaling (StandardScaler)", value=True,
                                          help="Recommended for Logistic Regression and SVC.")
                preprocessor = create_preprocessing_pipeline(numerical_features, apply_scaling)

                # Encode target labels
                st.markdown("##### Encoding Crop Type Labels")
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                st.info(f"Encoded {len(label_encoder.classes_)} unique crop types.")

                with st.expander("View Crop Type Encoding Map"):
                    encoding_df = pd.DataFrame({
                        'Crop Type': label_encoder.classes_,
                        'Encoded Value': range(len(label_encoder.classes_))
                    })
                    st.dataframe(encoding_df)

                # Split data
                st.markdown("##### Splitting Data (80% Train, 20% Test)")
                X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
                    X, y_encoded,
                    test_size=CONFIG['test_size'],
                    random_state=CONFIG['random_state'],
                    stratify=y_encoded
                )
                st.info(f"Dataset split: {len(X_train)} training, {len(X_test)} testing samples")

                # --- Data Exploration Tab ---
                with tab_eda:
                    st.header("🔍 Exploratory Data Analysis (EDA)")

                    # Create cached visualizations
                    with st.spinner("Generating visualizations..."):
                        viz_figures = create_visualizations(data, numerical_features, crop_type_col)

                    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Feature Distributions", "Correlation Analysis", "Custom Plots"])

                    with eda_tab1:
                        st.markdown("##### Distribution of Features")
                        st.pyplot(viz_figures['distributions'])
                        plt.close(viz_figures['distributions'])

                    with eda_tab2:
                        st.markdown("##### Correlation Matrix")
                        st.pyplot(viz_figures['correlation'])
                        plt.close(viz_figures['correlation'])

                    with eda_tab3: # This is now the custom plots tab
                        st.markdown("##### Create Custom Plots")
                        plot_col1, plot_col2, plot_col3 = st.columns(3)

                        with plot_col1:
                            plot_type = st.selectbox("Plot Type",
                                                   ["Scatter Plot", "Box Plot", "Violin Plot"], key='plot_type')

                        with plot_col2:
                            x_feature = st.selectbox("X-axis Feature",
                                                   numerical_features + [crop_type_col], key='x_feature')

                        with plot_col3:
                            if plot_type in ["Scatter Plot", "Box Plot", "Violin Plot"]:
                                # Ensure y_feature makes sense for the plot type. If X-axis is categorical (crop_type_col), Y-axis should be numerical.
                                if x_feature == crop_type_col:
                                    y_options = [f for f in numerical_features if f != x_feature]
                                    if not y_options:
                                        y_options = [None]
                                    y_feature = st.selectbox("Y-axis Feature", y_options, key='y_feature_cat_x')
                                else: # X-axis is numerical
                                    y_options = [None] + numerical_features + [crop_type_col]
                                    y_feature = st.selectbox("Y-axis Feature", y_options, key='y_feature_num_x')
                            else:
                                y_feature = None

                        if st.button("Generate Plot", key='generate_plot_btn'):
                            try:
                                if not x_feature:
                                    st.warning("Please select an X-axis feature.")
                                elif plot_type in ["Scatter Plot", "Box Plot", "Violin Plot"] and not y_feature:
                                    st.warning("Please select a Y-axis feature for this plot type.")
                                else:
                                    fig_custom, ax_custom = plt.subplots(figsize=(10, 6))

                                    if plot_type == "Scatter Plot":
                                        sns.scatterplot(x=data[x_feature], y=data[y_feature],
                                                      hue=data[crop_type_col], ax=ax_custom, alpha=0.7)
                                        ax_custom.set_title(f'{x_feature.capitalize()} vs {y_feature.capitalize()}')

                                    elif plot_type == "Box Plot":
                                        sns.boxplot(x=data[x_feature], y=data[y_feature], ax=ax_custom)
                                        ax_custom.set_title(f'{x_feature.capitalize()} vs {y_feature.capitalize()}')

                                    elif plot_type == "Violin Plot":
                                        sns.violinplot(x=data[x_feature], y=data[y_feature], ax=ax_custom)
                                        ax_custom.set_title(f'{x_feature.capitalize()} vs {y_feature.capitalize()}')

                                    st.pyplot(fig_custom)
                                    plt.close(fig_custom)

                            except Exception as e:
                                st.error(f"Error generating plot: {e}")

                # --- Model Training & Evaluation Tab ---
                with tab_model:
                    st.header("🤖 Model Training & Evaluation")

                    comparison_mode = st.radio(
                        "Choose Analysis Mode:",
                        ["Compare All Models", "Train Single Model"],
                        help="Compare all models to find the best one, or train a specific model", key='model_mode'
                    )

                    model = None
                    best_model_name = "N/A"
                    best_accuracy = 0.0
                    report_dict = {}

                    if comparison_mode == "Compare All Models":
                        st.info("Training and comparing all available models...")

                        models_dict = {
                            "Random Forest": Pipeline([
                                ('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier(random_state=CONFIG['random_state']))
                            ]),
                            "Logistic Regression": Pipeline([
                                ('preprocessor', preprocessor),
                                ('classifier', LogisticRegression(random_state=CONFIG['random_state'], max_iter=1000))
                            ]),
                            "Support Vector Classifier": Pipeline([
                                ('preprocessor', preprocessor),
                                ('classifier', SVC(random_state=CONFIG['random_state'], probability=True))
                            ])
                        }

                        model_results = compare_models(models_dict, X_train, X_test, y_train_encoded, y_test_encoded)

                        st.markdown("#### 📊 Model Comparison Results")
                        comparison_data = []
                        for name, results in model_results.items():
                            if 'error' not in results:
                                comparison_data.append({
                                    'Model': name,
                                    'Test Accuracy': f"{results['accuracy']:.4f}",
                                    'CV Mean': f"{results['cv_mean']:.4f}",
                                    'CV Std': f"{results['cv_std']:.4f}"
                                })
                            else:
                                comparison_data.append({
                                    'Model': name,
                                    'Test Accuracy': 'Error',
                                    'CV Mean': 'Error',
                                    'CV Std': results['error']
                                })

                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df)

                        valid_results = {k: v for k, v in model_results.items() if 'error' not in v}
                        if valid_results:
                            best_model_name = max(valid_results.keys(), key=lambda k: valid_results[k]['accuracy'])
                            model = valid_results[best_model_name]['model']
                            best_accuracy = valid_results[best_model_name]['accuracy']

                            st.success(f"🏆 Best Model: **{best_model_name}** (Accuracy: {best_accuracy:.4f})")
                        else:
                            st.error("All models failed to train. Please check your data.")
                            st.stop()

                    else: # Single model mode
                        model_choice = st.selectbox("Select Model",
                                                  ["Random Forest Classifier", "Logistic Regression", "Support Vector Classifier"], key='single_model_choice')

                        if model_choice == "Random Forest Classifier":
                            classifier = RandomForestClassifier(random_state=CONFIG['random_state'])
                        elif model_choice == "Logistic Regression":
                            classifier = LogisticRegression(random_state=CONFIG['random_state'], max_iter=1000)
                        else:
                            classifier = SVC(random_state=CONFIG['random_state'], probability=True)

                        model = Pipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', classifier)
                        ])

                        model, train_error = safe_model_training(model, X_train, y_train_encoded)
                        if train_error:
                            st.error(train_error)
                            st.stop()

                        st.success("Model training complete!")

                        best_model_name = model_choice
                        y_pred_encoded = model.predict(X_test)
                        best_accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

                    # Model Evaluation (common to both modes)
                    st.markdown("---")
                    st.subheader("📈 Model Performance Analysis")
                    y_pred_encoded = model.predict(X_test)
                    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
                    y_test_labels = label_encoder.inverse_transform(y_test_encoded)

                    # Performance metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{best_accuracy:.4f}")
                    with col2:
                        precision_avg = classification_report(y_test_encoded, y_pred_encoded, output_dict=True)['weighted avg']['precision']
                        st.metric("Precision (Weighted)", f"{precision_avg:.4f}")
                    with col3:
                        recall_avg = classification_report(y_test_encoded, y_pred_encoded, output_dict=True)['weighted avg']['recall']
                        st.metric("Recall (Weighted)", f"{recall_avg:.4f}")

                    # Detailed classification report
                    with st.expander("View Classification Report"):
                        report_dict = classification_report(y_test_encoded, y_pred_encoded, target_names=label_encoder.classes_, output_dict=True)
                        report_df = pd.DataFrame(report_dict).transpose()
                        st.dataframe(report_df.style.format(precision=4))

                    # Confusion Matrix
                    with st.expander("View Confusion Matrix"):
                        cm = confusion_matrix(y_test_encoded, y_pred_encoded)
                        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax_cm)
                        ax_cm.set_xlabel('Predicted Label')
                        ax_cm.set_ylabel('True Label')
                        ax_cm.set_title('Confusion Matrix')
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)

                    # Feature Importance
                    display_feature_importance(model, numerical_features)

                # --- Time-Series Forecasting Tab ---
                with tab_ts_forecast:
                    st.header("📈 Time-Series Forecasting for Environmental Factors (Manual ARIMA)")
                    st.markdown("Forecast future trends of environmental factors using ARIMA model. **If no Date/Time column is selected, forecasting will be based on the row order of your data, not actual calendar time.**")
                    if df_ts_ready is not None and not df_ts_ready.empty:
                        ts_feature_to_forecast = st.selectbox(
                            "Select environmental factor to forecast:", numerical_features, key='ts_feature_select'
                        )
                        forecast_steps_input = st.slider(
                            "Number of future points to forecast:", min_value=10, max_value=200, value=CONFIG['forecast_points'], key='forecast_steps_slider'
                        )

                        if st.button("🔮 Run ARIMA Forecast", key='run_arima_btn'):
                            with st.spinner(f"Running ARIMA forecast for {ts_feature_to_forecast}..."):
                                time_series_to_forecast = df_ts_ready[ts_feature_to_forecast]
                                
                                st.warning("Could not infer frequency of the time series or frequency is not set. Attempting to resample to daily frequency and interpolate. Ensure your date column is suitable for this.")
                                try:
                                    time_series_to_forecast = time_series_to_forecast.asfreq(CONFIG['default_arima_freq']).interpolate(method='linear')
                                    time_series_to_forecast = time_series_to_forecast.fillna(time_series_to_forecast.mean())

                                    if time_series_to_forecast.empty:
                                        st.error("Time series became empty after frequency resampling and interpolation. Cannot proceed.")
                                        st.stop()

                                    st.info("Resampling and interpolation applied for forecasting.")
                                except Exception as e:
                                    st.error(f"Could not resample time series: {e}. Ensure your date data is appropriate for resampling.")
                                    st.stop()

                                forecast_df, metrics, arima_error = run_arima_forecast(
                                    time_series_to_forecast, forecast_steps_input
                                )
                                if arima_error:
                                    st.error(f"❌ {arima_error}")
                                else:
                                    st.success(f"✅ Forecast for {ts_feature_to_forecast.capitalize()} completed!")

                                    # Plotting the forecast
                                    fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
                                    ax_forecast.plot(time_series_to_forecast.index, time_series_to_forecast, label='Historical Data', color='blue')
                                    ax_forecast.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red', linestyle='--')
                                    ax_forecast.set_title(f'ARIMA Forecast for {ts_feature_to_forecast.capitalize()}')
                                    ax_forecast.set_xlabel('Date')
                                    ax_forecast.set_ylabel(ts_feature_to_forecast.capitalize())
                                    ax_forecast.legend()
                                    ax_forecast.grid(True, alpha=0.3)
                                    st.pyplot(fig_forecast)
                                    plt.close(fig_forecast)

                                    st.markdown("##### 📈 Forecast Metrics (In-sample performance)")
                                    col_mae, col_rmse, col_mape = st.columns(3)
                                    with col_mae:
                                        st.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.2f}")
                                    with col_rmse:
                                        st.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.2f}")
                                    with col_mape:
                                        st.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE']:.2f}%" if not np.isnan(metrics['MAPE']) else "N/A")
                                    st.info("These metrics reflect the model's fit on the *historical* data. For more rigorous evaluation, consider splitting your time-series data into train/test sets before forecasting.")

                                    st.markdown("##### 🔮 Forecasted Values")
                                    st.dataframe(forecast_df)
                    else:
                        st.info("Please select a 'Date/Time Column' in the 'Data & Preprocessing' tab to enable time-series forecasting. Ensure your data has a time component and sufficient entries.")

                # --- Predictions & Tools Tab ---
                with tab_predict:
                    st.header("🚀 Predictions & Tools")
                    st.subheader("Interactive Crop Recommendation")
                    st.markdown("Enter soil and climate conditions to get personalized crop recommendations.")

                    with st.form("prediction_form"):
                        pred_col1, pred_col2, pred_col3 = st.columns(3)
                        with pred_col1:
                            nitrogen = st.number_input("Nitrogen (N)", value=float(X[n_col].mean()), min_value=0.0, help="Nitrogen content in soil", key='pred_n')
                            phosphorus = st.number_input("Phosphorus (P)", value=float(X[p_col].mean()), min_value=0.0, help="Phosphorus content in soil", key='pred_p')
                            potassium = st.number_input("Potassium (K)", value=float(X[k_col].mean()), min_value=0.0, help="Potassium content in soil", key='pred_k')
                        with pred_col2:
                            temperature = st.number_input("Temperature (°C)", value=float(X[temp_col].mean()), help="Average temperature", key='pred_temp')
                            humidity = st.number_input("Humidity (%)", value=float(X[hum_col].mean()), min_value=0.0, max_value=100.0, help="Relative humidity", key='pred_hum')
                            ph = st.number_input("pH Value", value=float(X[ph_col].mean()), min_value=0.0, max_value=14.0, help="Soil pH level", key='pred_ph')
                        with pred_col3:
                            rainfall = st.number_input("Rainfall (mm)", value=float(X[rain_col].mean()), min_value=0.0, help="Annual rainfall", key='pred_rain')
                        predict_button = st.form_submit_button("🌱 Get Recommendation", use_container_width=True)

                        if predict_button:
                            custom_input = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]], columns=numerical_features)
                            predicted_encoded = model.predict(custom_input)[0]
                            predicted_crop = label_encoder.inverse_transform([predicted_encoded])[0]
                            st.success(f"🌾 **Recommended Crop: {predicted_crop.capitalize()}**")

                            if hasattr(model.named_steps['classifier'], 'predict_proba'):
                                probabilities = model.predict_proba(custom_input)[0]
                                prob_df = pd.DataFrame({
                                    'Crop Type': [crop.capitalize() for crop in label_encoder.classes_],
                                    'Probability': probabilities
                                }).sort_values(by='Probability', ascending=False)
                                st.markdown("##### 📊 Prediction Confidence")
                                st.dataframe(prob_df.style.format({'Probability': "{:.2%}"}))

            except Exception as e:
                st.error(f"An error occurred during column mapping or initial processing: {e}")
                st.exception(e) # Display full traceback for debugging
        else:
            st.warning("Please select all required columns (Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH Value, Rainfall, Crop Type) and ensure they are unique.")
            st.info("The Date/Time column is optional but recommended for time-series forecasting.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
🌱 **Smart Agriculture Analysis**
Version 2.0 - Enhanced Edition

Built with Streamlit, scikit-learn, statsmodels, and agricultural expertise.
""")
