import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from datetime import datetime, timedelta
import joblib
import os
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes.utils import summary_data_from_transaction_data
import warnings
warnings.filterwarnings('ignore')

# App title and description
st.set_page_config(page_title="Revenue Optimization & Customer Retention Engine", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

st.title("AI-Powered Predictive Revenue Optimization & Customer Retention Engine")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", 
                        ["Data Overview", 
                         "Customer Segmentation", 
                         "Churn Prediction", 
                         "Revenue Forecast", 
                         "Product Recommendations", 
                         "Customer Lifetime Value"])

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=['csv'])

# Function to load and preprocess data
import chardet

@st.cache_data
def load_and_process_data(file):
    if file is not None:
        # Read a sample of the file to detect encoding
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        file.seek(0)  # Reset file pointer after reading
        
        df = pd.read_csv(file, encoding=encoding)
        
        # Basic cleaning
        df = df.dropna(subset=['CustomerID'])
        df['CustomerID'] = df['CustomerID'].astype(int)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Remove canceled orders (if invoice no contains 'C')
        df = df[~df['InvoiceNo'].astype(str).str.contains('C', na=False)]
        
        # Calculate total price
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        # Filter out returns (negative quantities) and zero prices
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        
        return df
    return None

# Load data if file is uploaded
if uploaded_file is not None:
    df = load_and_process_data(uploaded_file)
    
    if df is not None:
        # Display data overview page
        if page == "Data Overview":
            st.header("Data Overview")
            
            # Display dataframe preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Display data information
            st.subheader("Data Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Number of Records:** {df.shape[0]}")
                st.write(f"**Number of Customers:** {df['CustomerID'].nunique()}")
                st.write(f"**Number of Products:** {df['StockCode'].nunique()}")
                st.write(f"**Date Range:** {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
            
            with col2:
                st.write(f"**Total Revenue:** ${df['TotalPrice'].sum():,.2f}")
                st.write(f"**Average Order Value:** ${df.groupby('InvoiceNo')['TotalPrice'].sum().mean():,.2f}")
                st.write(f"**Number of Countries:** {df['Country'].nunique()}")
            
            # Time series revenue trend
            st.subheader("Revenue Trend Over Time")
            
            daily_revenue = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().reset_index()
            daily_revenue.columns = ['Date', 'Revenue']
            
            fig = px.line(daily_revenue, x='Date', y='Revenue', title='Daily Revenue')
            st.plotly_chart(fig, use_container_width=True)
            
            # Top countries and products
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Countries by Revenue")
                country_revenue = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
                fig = px.bar(country_revenue, x='Country', y='TotalPrice', title='Revenue by Country')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top Products by Sales")
                product_revenue = df.groupby(['StockCode', 'Description'])['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
                fig = px.bar(product_revenue, x='TotalPrice', y='Description', orientation='h', title='Top Products by Revenue')
                st.plotly_chart(fig, use_container_width=True)
            
        # Customer Segmentation (RFM Analysis)
        elif page == "Customer Segmentation":
            st.header("Customer Segmentation (RFM Analysis)")
            
            # Calculate RFM
            # Get max date to use as snapshot date
            snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
            
            # Compute RFM metrics
            rfm = df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
                'InvoiceNo': 'nunique',  # Frequency
                'TotalPrice': 'sum'  # Monetary Value
            }).reset_index()
            
            rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
            
            # Scale RFM data for clustering
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
            
            # Determine optimal number of clusters
            if st.checkbox("Find Optimal Clusters (K-means Elbow Method)"):
                inertia = []
                k_range = range(1, 11)
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(rfm_scaled)
                    inertia.append(kmeans.inertia_)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(k_range), y=inertia, mode='lines+markers'))
                fig.update_layout(title='Elbow Method for Optimal k',
                                xaxis_title='Number of Clusters (k)',
                                yaxis_title='Inertia')
                st.plotly_chart(fig, use_container_width=True)
            
            # Default clusters = 4 or let user select
            num_clusters = st.slider("Select Number of Customer Segments", min_value=2, max_value=10, value=4)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
            
            # Analyze clusters
            cluster_analysis = rfm.groupby('Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'CustomerID': 'count'
            }).reset_index()
            
            cluster_analysis.columns = ['Cluster', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Customer_Count']
            cluster_analysis['Percentage'] = (cluster_analysis['Customer_Count'] / cluster_analysis['Customer_Count'].sum() * 100).round(2)
            
            # Sort by monetary value to assign segment names
            cluster_analysis = cluster_analysis.sort_values(by='Avg_Monetary', ascending=False)
            cluster_analysis['Segment'] = ['High Value', 'Loyal', 'Potential Loyalists', 'At Risk'] + \
                                        ['Segment ' + str(i+5) for i in range(num_clusters-4)] if num_clusters > 4 else \
                                        ['High Value', 'Loyal', 'Potential Loyalists', 'At Risk'][:num_clusters]
            
            # Create a mapping between cluster numbers and segment names
            segment_map = dict(zip(cluster_analysis['Cluster'], cluster_analysis['Segment']))
            rfm['Segment'] = rfm['Cluster'].map(segment_map)
            
            # Display segment information
            st.subheader("Customer Segments Overview")
            st.dataframe(cluster_analysis)
            
            # Visualize clusters
            st.subheader("Segment Visualization")
            
            tab1, tab2, tab3 = st.tabs(["3D Scatter Plot", "Segment Size", "Segment Metrics"])
            
            with tab1:
                fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                                  color='Segment', hover_name='CustomerID',
                                  labels={'Recency': 'Recency (days)', 'Frequency': 'Frequency', 'Monetary': 'Monetary Value'},
                                  title='3D RFM Segmentation')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = px.pie(cluster_analysis, values='Customer_Count', names='Segment', 
                           title='Customer Segment Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Radar chart of normalized values for each segment
                radar_data = cluster_analysis.copy()
                
                # Normalize the metrics for radar chart
                for col in ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']:
                    if col == 'Avg_Recency':  # Lower recency is better, so invert it
                        radar_data[col + '_Norm'] = 1 - ((radar_data[col] - radar_data[col].min()) / 
                                                       (radar_data[col].max() - radar_data[col].min()))
                    else:
                        radar_data[col + '_Norm'] = (radar_data[col] - radar_data[col].min()) / \
                                                   (radar_data[col].max() - radar_data[col].min())
                
                fig = go.Figure()
                
                for i, segment in enumerate(radar_data['Segment']):
                    fig.add_trace(go.Scatterpolar(
                        r=[radar_data.iloc[i]['Avg_Recency_Norm'], 
                           radar_data.iloc[i]['Avg_Frequency_Norm'], 
                           radar_data.iloc[i]['Avg_Monetary_Norm']],
                        theta=['Recency', 'Frequency', 'Monetary'],
                        fill='toself',
                        name=segment
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    title='Segment Metrics Comparison'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Export segments
            if st.button("Download Customer Segments"):
                temp_file = "customer_segments.csv"
                rfm.to_csv(temp_file, index=False)
                with open(temp_file, 'rb') as f:
                    st.download_button("Download CSV", f, file_name="customer_segments.csv", mime="text/csv")
                os.remove(temp_file)
                
        # Churn Prediction
        elif page == "Churn Prediction":
            st.header("Customer Churn Prediction")
            
            # Define churn period
            snapshot_date = df['InvoiceDate'].max()
            churn_period = st.slider("Define Churn Period (days):", 30, 180, 90)
            
            # Calculate features for churn model
            # First, get the date boundary for historical data and prediction period
            boundary_date = snapshot_date - timedelta(days=churn_period)
            
            # Historical period data (training data)
            hist_df = df[df['InvoiceDate'] <= boundary_date]
            
            # Prediction period data (to determine if customers churned)
            pred_df = df[df['InvoiceDate'] > boundary_date]
            
            # Create customer features from historical data
            customer_features = hist_df.groupby('CustomerID').agg({
                'InvoiceDate': [
                    lambda x: (boundary_date - x.max()).days,  # Recency
                    lambda x: (x.max() - x.min()).days if len(x) > 1 else 0,  # Customer age
                    'count'  # Total transactions
                ],
                'InvoiceNo': 'nunique',  # Number of orders
                'TotalPrice': ['sum', 'mean', 'std'],  # Monetary metrics
                'Quantity': ['sum', 'mean', 'std']  # Product metrics
            })
            
            # Flatten the column hierarchy
            customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns.values]
            customer_features.reset_index(inplace=True)
            
            # Rename columns for clarity
            customer_features.rename(columns={
                'InvoiceDate_<lambda_0>': 'Recency',
                'InvoiceDate_<lambda_1>': 'CustomerAge',
                'InvoiceDate_count': 'TotalTransactions',
                'InvoiceNo_nunique': 'OrderCount',
                'TotalPrice_sum': 'TotalSpend',
                'TotalPrice_mean': 'AvgOrderValue',
                'TotalPrice_std': 'OrderValueStd',
                'Quantity_sum': 'TotalItems',
                'Quantity_mean': 'AvgItems',
                'Quantity_std': 'ItemsStd'
            }, inplace=True)
            
            # Handle NaN values from std calculations for customers with single orders
            customer_features.fillna(0, inplace=True)
            
            # Add frequency metrics
            customer_features['PurchaseFrequency'] = customer_features['OrderCount'] / (customer_features['CustomerAge'] + 1)
            
            # Get list of customers who made purchases in the prediction period
            active_customers = pred_df['CustomerID'].unique()
            
            # Add churn label: 1 if customer didn't purchase in prediction period
            customer_features['Churned'] = ~customer_features['CustomerID'].isin(active_customers)
            
            # Display churn overview
            st.subheader("Churn Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                churn_rate = customer_features['Churned'].mean() * 100
                st.metric("Churn Rate", f"{churn_rate:.2f}%")
                st.write(f"Period: {boundary_date.date()} to {snapshot_date.date()} ({churn_period} days)")
                
                fig = px.pie(
                    values=[customer_features['Churned'].sum(), len(customer_features) - customer_features['Churned'].sum()],
                    names=['Churned', 'Retained'],
                    title='Customer Churn Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Churned vs retained customers by recency
                fig = px.histogram(
                    customer_features,
                    x='Recency',
                    color='Churned',
                    barmode='overlay',
                    title='Churn by Recency'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Train churn prediction model
            st.subheader("Churn Prediction Model")
            
            # Prepare features and target
            X = customer_features.drop(['CustomerID', 'Churned'], axis=1)
            y = customer_features['Churned']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            
            # Train model if user clicks button
            if st.button("Train Churn Prediction Model"):
                with st.spinner("Training model..."):
                    # Train a Random Forest model
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Model performance
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
                    
                    # Display confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Feature importance
                    importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### Confusion Matrix")
                        fig = px.imshow(
                            cm, 
                            text_auto=True, 
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=['Retained', 'Churned'],
                            y=['Retained', 'Churned']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with col2:
                        st.write("### Feature Importance")
                        fig = px.bar(
                            importance.head(10),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 10 Features for Churn Prediction'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Classification report
                    st.write("### Model Performance Metrics")
                    st.text(classification_report(y_test, y_pred))
                    
                    # Save model
                    joblib.dump(model, 'churn_model.pkl')
                    
                    # Churn prediction for all customers
                    customer_features['Churn_Probability'] = model.predict_proba(X)[:, 1]
                    
                    # Display customers at risk
                    st.subheader("Customers at Risk of Churning")
                    risk_threshold = st.slider("Churn Risk Threshold", 0.0, 1.0, 0.7)
                    
                    at_risk = customer_features[customer_features['Churn_Probability'] >= risk_threshold]
                    at_risk = at_risk.sort_values('Churn_Probability', ascending=False)
                    
                    # Add customer value metric
                    at_risk['CLV_Estimation'] = at_risk['TotalSpend'] * (1 / (1 + at_risk['Churn_Probability']))
                    at_risk['Prioritization_Score'] = at_risk['CLV_Estimation'] * at_risk['Churn_Probability']
                    
                    # Select columns to display
                    display_cols = ['CustomerID', 'Churn_Probability', 'Recency', 'TotalSpend', 
                                  'OrderCount', 'CLV_Estimation', 'Prioritization_Score']
                    
                    st.write(f"Found {len(at_risk)} customers with churn risk >= {risk_threshold}")
                    st.dataframe(at_risk[display_cols])
                    
                    # Download at-risk customer list
                    if not at_risk.empty:
                        temp_file = "at_risk_customers.csv"
                        at_risk[display_cols].to_csv(temp_file, index=False)
                        with open(temp_file, 'rb') as f:
                            st.download_button("Download At-Risk Customers", f, 
                                             file_name="at_risk_customers.csv", mime="text/csv")
                        os.remove(temp_file)
            
        # Revenue Forecast
        elif page == "Revenue Forecast":
            st.header("Revenue Forecasting")
            
            # Prepare time series data
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['Date'] = df['InvoiceDate'].dt.date
            
            # Daily revenue
            daily_revenue = df.groupby('Date')['TotalPrice'].sum().reset_index()
            daily_revenue.columns = ['Date', 'Revenue']
            daily_revenue['Date'] = pd.to_datetime(daily_revenue['Date'])
            
            # Add time features
            daily_revenue['Day'] = daily_revenue['Date'].dt.day
            daily_revenue['Month'] = daily_revenue['Date'].dt.month
            daily_revenue['Year'] = daily_revenue['Date'].dt.year
            daily_revenue['DayOfWeek'] = daily_revenue['Date'].dt.dayofweek
            daily_revenue['WeekOfYear'] = daily_revenue['Date'].dt.isocalendar().week
            
            # Add lag features
            for lag in [1, 7, 14, 30]:
                daily_revenue[f'Revenue_Lag_{lag}'] = daily_revenue['Revenue'].shift(lag)
            
            # Add rolling statistics
            for window in [7, 14, 30]:
                daily_revenue[f'Revenue_Rolling_Mean_{window}'] = daily_revenue['Revenue'].rolling(window=window).mean()
                daily_revenue[f'Revenue_Rolling_Std_{window}'] = daily_revenue['Revenue'].rolling(window=window).std()
            
            # Fill NaN values
            daily_revenue = daily_revenue.dropna()
            
            # Display time series data
            st.subheader("Historical Revenue Trend")
            fig = px.line(daily_revenue, x='Date', y='Revenue')
            st.plotly_chart(fig, use_container_width=True)
            
            # Train revenue forecasting model
            st.subheader("Revenue Forecast Model")
            
            # Features and target for model
            X = daily_revenue.drop(['Date', 'Revenue'], axis=1)
            y = daily_revenue['Revenue']
            
            # Split data (use time-based split for time series)
            split_point = int(len(daily_revenue) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            # Train model button
            if st.button("Train Revenue Forecast Model"):
                with st.spinner("Training forecast model..."):
                    # Train Random Forest Regressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Model performance
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = model.score(X_test, y_test)
                    
                    st.success(f"Model trained successfully! RMSE: ${rmse:.2f}, R²: {r2:.2%}")
                    
                    # Feature importance
                    importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Visualize actual vs predicted
                    results = pd.DataFrame({
                        'Date': daily_revenue['Date'].iloc[split_point:].reset_index(drop=True),
                        'Actual': y_test.reset_index(drop=True),
                        'Predicted': y_pred
                    })
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Actual vs Predicted Revenue")
                        fig = px.line(results, x='Date', y=['Actual', 'Predicted'],
                                    labels={'value': 'Revenue', 'variable': 'Type'},
                                    title='Actual vs Predicted Revenue')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("### Top 10 Predictive Features")
                        fig = px.bar(
                            importance.head(10),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Feature Importance'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Save model for future use
                    joblib.dump(model, 'revenue_forecast_model.pkl')
                    
                    # Future forecast section
                    st.subheader("Future Revenue Forecast")
                    
                    # Let user select forecast horizon
                    forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
                    
                    # Generate dates for forecast
                    last_date = daily_revenue['Date'].max()
                    forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                    
                    # Create dataframe for forecast
                    forecast_df = pd.DataFrame({'Date': forecast_dates})
                    forecast_df['Day'] = forecast_df['Date'].dt.day
                    forecast_df['Month'] = forecast_df['Date'].dt.month
                    forecast_df['Year'] = forecast_df['Date'].dt.year
                    forecast_df['DayOfWeek'] = forecast_df['Date'].dt.dayofweek
                    forecast_df['WeekOfYear'] = forecast_df['Date'].dt.isocalendar().week
                    
                    # This is a simplified approach - in a real application, you would need
                    # a more sophisticated method to generate lag features for future dates
                    # Here, we'll use the most recent values as a proxy
                    
                    for lag in [1, 7, 14, 30]:
                        forecast_df[f'Revenue_Lag_{lag}'] = daily_revenue['Revenue'].iloc[-lag]
                    
                    for window in [7, 14, 30]:
                        forecast_df[f'Revenue_Rolling_Mean_{window}'] = daily_revenue['Revenue'].tail(window).mean()
                        forecast_df[f'Revenue_Rolling_Std_{window}'] = daily_revenue['Revenue'].tail(window).std()
                    
                    # Make predictions
                    forecast_df['Predicted_Revenue'] = model.predict(forecast_df.drop('Date', axis=1))
                    
                    # Visualize forecast
                    combined_df = pd.concat([
                        daily_revenue[['Date', 'Revenue']].rename(columns={'Revenue': 'Historical'}),
                        forecast_df[['Date', 'Predicted_Revenue']].rename(columns={'Predicted_Revenue': 'Forecast'})
                    ])
                    
                    # Plot with confidence interval
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=daily_revenue['Date'],
                        y=daily_revenue['Revenue'],
                        mode='lines',
                        name='Historical Revenue',
                        line=dict(color='blue')
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Predicted_Revenue'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Confidence interval (using a simple approach for demonstration)
                    std_error = np.sqrt(mean_squared_error(y_test, y_pred))
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Predicted_Revenue'] + 1.96 * std_error,
                        mode='lines',
                        name='Upper Bound (95%)',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Predicted_Revenue'] - 1.96 * std_error,
                        mode='lines',
                        name='Lower Bound (95%)',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.1)',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title='Revenue Forecast with 95% Confidence Interval',
                        xaxis_title='Date',
                        yaxis_title='Revenue',
                        legend=dict(y=0.99, x=0.01)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    st.write("### Forecast Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total Forecasted Revenue", 
                            f"${forecast_df['Predicted_Revenue'].sum():,.2f}",
                            delta=f"{(forecast_df['Predicted_Revenue'].sum() / (daily_revenue['Revenue'].tail(forecast_days).sum()) - 1) * 100:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Avg Daily Revenue (Forecast)", 
                            f"${forecast_df['Predicted_Revenue'].mean():,.2f}",
                            delta=f"{(forecast_df['Predicted_Revenue'].mean() / daily_revenue['Revenue'].mean() - 1) * 100:.1f}%"
                        )
                    
                    with col3:
                        peak_date = forecast_df.loc[forecast_df['Predicted_Revenue'].idxmax(), 'Date']
                        st.metric(
                            "Peak Revenue Date", 
                            f"{peak_date.strftime('%Y-%m-%d')}",
                            delta=f"${forecast_df['Predicted_Revenue'].max():,.2f}"
                        )
                    
                    # Download forecast data
                    temp_file = "revenue_forecast.csv"
                    forecast_df[['Date', 'Predicted_Revenue']].to_csv(temp_file, index=False)
                    with open(temp_file, 'rb') as f:
                        st.download_button("Download Forecast Data", f, 
                                         file_name="revenue_forecast.csv", mime="text/csv")
                    os.remove(temp_file)
        
        # Product Recommendations
        elif page == "Product Recommendations":
            st.header("Product Recommendations Engine")
            
            # Create product association rules
            st.subheader("Product Association Analysis")
            
            # Get transactions data in basket format
            transactions = df.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().fillna(0)
            transactions = transactions.applymap(lambda x: 1 if x > 0 else 0)
            
            # Calculate product co-occurrence matrix
            product_co_occurrence = transactions.T.dot(transactions)
            np.fill_diagonal(product_co_occurrence.values, 0)  # Remove self-associations
            
            # Create product lookup for descriptions
            product_lookup = df[['StockCode', 'Description']].drop_duplicates()
            product_lookup = product_lookup.set_index('StockCode')
            
            # Show top product associations
            if st.checkbox("Show Product Association Matrix"):
                # Get top products by sales for a cleaner visualization
                top_products = df.groupby('StockCode')['Quantity'].sum().nlargest(20).index
                
                # Filter co-occurrence matrix
                filtered_matrix = product_co_occurrence.loc[top_products, top_products]
                
                # Add product descriptions
                filtered_matrix.index = [f"{idx} - {product_lookup.loc[idx, 'Description'][:20]}..." 
                                        if idx in product_lookup.index else idx 
                                        for idx in filtered_matrix.index]
                filtered_matrix.columns = filtered_matrix.index
                
                # Plot heatmap
                fig = px.imshow(
                    filtered_matrix,
                    text_auto=True,
                    title="Product Co-occurrence Matrix (Top 20 Products)",
                    color_continuous_scale='blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # User-based product recommendations
            st.subheader("Customer Product Recommendations")
            
            # Get all customers
            all_customers = df['CustomerID'].unique()
            
            # Create a customer selector
            selected_customer = st.selectbox("Select a Customer ID", all_customers)
            
            if selected_customer:
                # Get customer purchase history
                customer_purchases = df[df['CustomerID'] == selected_customer]
                
                customer_products = customer_purchases['StockCode'].unique()
                
                # Display customer info
                st.write(f"### Customer {selected_customer} Purchase History")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Purchases", f"${customer_purchases['TotalPrice'].sum():.2f}")
                with col2:
                    st.metric("Number of Orders", len(customer_purchases['InvoiceNo'].unique()))
                with col3:
                    st.metric("Unique Products", len(customer_products))
                
                # Show top purchased products
                top_customer_products = customer_purchases.groupby('StockCode').agg({
                    'Quantity': 'sum',
                    'TotalPrice': 'sum',
                    'Description': 'first'
                }).sort_values('Quantity', ascending=False).head(5)
                
                st.write("#### Top Purchased Products")
                st.dataframe(top_customer_products[['Description', 'Quantity', 'TotalPrice']])
                
                # Generate recommendations
                st.write("### Product Recommendations")
                
                # Create a simple recommendation algorithm based on product co-occurrence
                recommendations = pd.Series(dtype='float64')
                
                for product in customer_products:
                    if product in product_co_occurrence.index:
                        # Get co-occurrence scores for this product
                        similar_products = product_co_occurrence[product]
                        # Add to recommendations
                        recommendations = recommendations.add(similar_products, fill_value=0)
                
                # Remove already purchased products
                recommendations = recommendations[~recommendations.index.isin(customer_products)]
                
                # Get top recommendations
                top_recommendations = recommendations.nlargest(10)
                
                # Format recommendations
                if not top_recommendations.empty:
                    rec_df = pd.DataFrame({
                        'StockCode': top_recommendations.index,
                        'Score': top_recommendations.values
                    })
                    
                    # Add product descriptions
                    rec_df = rec_df.merge(product_lookup.reset_index(), on='StockCode', how='left')
                    
                    # Add product metrics
                    product_metrics = df.groupby('StockCode').agg({
                        'Quantity': 'sum',
                        'TotalPrice': 'sum'
                    }).reset_index()
                    
                    rec_df = rec_df.merge(product_metrics, on='StockCode', how='left')
                    
                    # Calculate popularity percentile
                    product_popularity = df.groupby('StockCode')['Quantity'].count()
                    product_popularity = product_popularity.rank(pct=True)
                    rec_df['Popularity'] = rec_df['StockCode'].map(product_popularity)
                    
                    # Display recommendations
                    st.dataframe(rec_df[['StockCode', 'Description', 'Score', 'Quantity', 'TotalPrice', 'Popularity']])
                    
                    # Visualization
                    fig = px.bar(
                        rec_df.head(5),
                        x='Score',
                        y='Description',
                        orientation='h',
                        title='Top 5 Recommended Products'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No recommendations available for this customer.")
                
                # Advanced recommendations button
                if st.button("Generate Advanced Recommendations"):
                    with st.spinner("Generating personalized recommendations..."):
                        # Create a customer-product matrix
                        customer_product_matrix = df.pivot_table(
                            index='CustomerID',
                            columns='StockCode',
                            values='Quantity',
                            aggfunc='sum',
                            fill_value=0
                        )
                        
                        # Convert to binary (purchased or not)
                        customer_product_matrix = customer_product_matrix.applymap(lambda x: 1 if x > 0 else 0)
                        
                        # Calculate customer similarity
                        from sklearn.metrics.pairwise import cosine_similarity
                        
                        # Check if the customer is in the matrix
                        if selected_customer in customer_product_matrix.index:
                            # Get customer vector
                            customer_vector = customer_product_matrix.loc[selected_customer]
                            
                            # Calculate similarity to other customers
                            customer_similarities = []
                            for cust in customer_product_matrix.index:
                                if cust != selected_customer:
                                    other_vector = customer_product_matrix.loc[cust]
                                    similarity = cosine_similarity(
                                        customer_vector.values.reshape(1, -1),
                                        other_vector.values.reshape(1, -1)
                                    )[0][0]
                                    customer_similarities.append((cust, similarity))
                            
                            # Find similar customers
                            similar_customers = sorted(customer_similarities, key=lambda x: x[1], reverse=True)[:10]
                            
                            # Get products from similar customers that our customer hasn't purchased
                            similar_customer_ids = [cust for cust, _ in similar_customers]
                            similar_customer_purchases = df[df['CustomerID'].isin(similar_customer_ids)]
                            
                            # Products the similar customers bought
                            similar_products = similar_customer_purchases['StockCode'].unique()
                            
                            # Filter out products our customer already bought
                            new_products = [p for p in similar_products if p not in customer_products]
                            
                            # Calculate a weighted score for each product based on similar customer purchases
                            product_scores = {}
                            
                            for product in new_products:
                                score = 0
                                for cust, similarity in similar_customers:
                                    # Check if this customer bought the product
                                    if product in df[df['CustomerID'] == cust]['StockCode'].values:
                                        score += similarity
                                product_scores[product] = score
                            
                            # Sort products by score
                            recommended_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                            
                            # Display results
                            if recommended_products:
                                st.write("### Collaborative Filtering Recommendations")
                                
                                # Convert to DataFrame
                                collab_rec_df = pd.DataFrame(recommended_products, columns=['StockCode', 'Score'])
                                
                                # Add product info
                                collab_rec_df = collab_rec_df.merge(product_lookup.reset_index(), on='StockCode', how='left')
                                
                                # Add product metrics
                                collab_rec_df = collab_rec_df.merge(product_metrics, on='StockCode', how='left')
                                
                                st.dataframe(collab_rec_df[['StockCode', 'Description', 'Score', 'Quantity', 'TotalPrice']])
                                
                                # Show similar customers
                                st.write("### Similar Customers")
                                similar_cust_df = pd.DataFrame(similar_customers, columns=['CustomerID', 'Similarity'])
                                st.dataframe(similar_cust_df)
                            else:
                                st.write("No collaborative filtering recommendations available.")
                        else:
                            st.write("Customer not found in the matrix. Try another customer.")
        
        # Customer Lifetime Value
        elif page == "Customer Lifetime Value":
            st.header("Customer Lifetime Value Analysis")
            
            # Prepare data for CLV analysis
            # We'll use the BG/NBD model and Gamma-Gamma model from the lifetimes package
            
            # Format data for lifetimes package
            rfm_data = df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,  # Recency
                'InvoiceNo': 'nunique',  # Frequency
                'TotalPrice': ['sum', 'count', 'mean']  # Monetary metrics
            })
            
            # Flatten the column hierarchy
            rfm_data.columns = ['_'.join(col).strip() for col in rfm_data.columns.values]
            rfm_data.reset_index(inplace=True)
            
            # Rename columns for clarity
            rfm_data.rename(columns={
                'InvoiceDate_<lambda_0>': 'Recency',
                'InvoiceNo_nunique': 'Frequency',
                'TotalPrice_sum': 'MonetaryValue',
                'TotalPrice_count': 'T',
                'TotalPrice_mean': 'AvgOrderValue'
            }, inplace=True)
            
            # Calculate time since first purchase (T)
            first_purchase = df.groupby('CustomerID')['InvoiceDate'].min()
            last_purchase = df.groupby('CustomerID')['InvoiceDate'].max()
            rfm_data['T'] = (last_purchase - first_purchase).dt.days + 1  # Add 1 to avoid zero
            
            # Filter to include only customers with at least 2 purchases
            rfm_data = rfm_data[rfm_data['Frequency'] > 1]
            
            # Only proceed if we have enough data
            if len(rfm_data) > 10:
                # Calibrate BG/NBD model for purchase frequency
                st.subheader("Customer Lifetime Value Modeling")
                
                if st.checkbox("Run CLV Analysis"):
                    with st.spinner("Training CLV models..."):
                        # Create summary data
                        summary_data = pd.DataFrame({
                            'frequency': rfm_data['Frequency'] - 1,  # Frequency excluding first purchase
                            'recency': rfm_data['Recency'],
                            'T': rfm_data['T'],
                            'monetary_value': rfm_data['MonetaryValue'] / rfm_data['Frequency']  # Average per order
                        })
                        
                        # Fit BG/NBD model
                        bgf = BetaGeoFitter(penalizer_coef=0.01)
                        bgf.fit(summary_data['frequency'], summary_data['recency'], summary_data['T'])
                        
                        # Fit Gamma-Gamma model for monetary value
                        ggf = GammaGammaFitter(penalizer_coef=0.01)
                        ggf.fit(summary_data['frequency'], summary_data['monetary_value'])
                        
                        # Predict future transactions
                        time_horizon = st.slider("Prediction Horizon (days)", 30, 365, 180)
                        
                        # Predict number of future transactions
                        rfm_data['Predicted_Purchases'] = bgf.predict(
                            time_horizon,
                            rfm_data['Frequency'] - 1,
                            rfm_data['Recency'],
                            rfm_data['T']
                        )
                        
                        # Calculate probability of being alive
                        rfm_data['Prob_Alive'] = bgf.conditional_probability_alive(
                            rfm_data['Frequency'] - 1,
                            rfm_data['Recency'],
                            rfm_data['T']
                        )
                        
                        # Predict CLV using both models
                        rfm_data['CLV'] = ggf.customer_lifetime_value(
                            bgf,
                            rfm_data['Frequency'] - 1,
                            rfm_data['Recency'],
                            rfm_data['T'],
                            rfm_data['MonetaryValue'] / rfm_data['Frequency'],
                            time=time_horizon,
                            discount_rate=0.01  # Monthly discount rate ~ 12.7% annually
                        )
                        
                        # Display descriptive statistics
                        st.write(f"### CLV Predictions ({time_horizon} days)")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average CLV", f"${rfm_data['CLV'].mean():.2f}")
                        with col2:
                            st.metric("Median CLV", f"${rfm_data['CLV'].median():.2f}")
                        with col3:
                            st.metric("Total CLV", f"${rfm_data['CLV'].sum():.2f}")
                        
                        # Distribution of CLV
                        st.write("### Customer Lifetime Value Distribution")
                        fig = px.histogram(
                            rfm_data,
                            x='CLV',
                            nbins=50,
                            title='CLV Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # CLV vs Probability Alive
                        fig = px.scatter(
                            rfm_data,
                            x='Prob_Alive',
                            y='CLV',
                            color='Frequency',
                            size='MonetaryValue',
                            hover_data=['CustomerID'],
                            title='CLV vs Probability Alive'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Top customers by CLV
                        st.write("### Top Customers by Lifetime Value")
                        top_clv = rfm_data.sort_values('CLV', ascending=False).head(20)
                        st.dataframe(top_clv[['CustomerID', 'Frequency', 'MonetaryValue', 'Prob_Alive', 'Predicted_Purchases', 'CLV']])
                        
                        # Probability alive matrix
                        st.write("### Customer Retention Matrix")
                        fig = plt.figure(figsize=(10, 8))
                        plot_probability_alive_matrix(bgf)
                        st.pyplot(fig)
                        
                        # Customer segmentation based on CLV
                        st.write("### Customer Segmentation by CLV")
                        
                        # Create segments
                        rfm_data['CLV_Segment'] = pd.qcut(
                            rfm_data['CLV'],
                            q=4,
                            labels=['Low Value', 'Medium Value', 'High Value', 'Top Value']
                        )
                        
                        # Segment analysis
                        segment_analysis = rfm_data.groupby('CLV_Segment').agg({
                            'CustomerID': 'count',
                            'Frequency': 'mean',
                            'MonetaryValue': 'mean',
                            'Prob_Alive': 'mean',
                            'CLV': 'mean'
                        }).reset_index()
                        
                        segment_analysis.columns = ['Segment', 'Count', 'Avg_Frequency', 'Avg_Monetary', 'Avg_Prob_Alive', 'Avg_CLV']
                        
                        # Calculate percentages
                        segment_analysis['Percentage'] = segment_analysis['Count'] / segment_analysis['Count'].sum() * 100
                        segment_analysis['Total_CLV'] = segment_analysis['Avg_CLV'] * segment_analysis['Count']
                        segment_analysis['CLV_Percentage'] = segment_analysis['Total_CLV'] / segment_analysis['Total_CLV'].sum() * 100
                        
                        # Display segment analysis
                        st.dataframe(segment_analysis)
                        
                        # Visualize segments
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.pie(
                                segment_analysis,
                                values='Count',
                                names='Segment',
                                title='Customer Segment Distribution'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.pie(
                                segment_analysis,
                                values='Total_CLV',
                                names='Segment',
                                title='CLV Distribution by Segment'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download CLV data
                        temp_file = "customer_lifetime_value.csv"
                        rfm_data.to_csv(temp_file, index=False)
                        with open(temp_file, 'rb') as f:
                            st.download_button("Download CLV Data", f, 
                                             file_name="customer_lifetime_value.csv", mime="text/csv")
                        os.remove(temp_file)
            else:
                st.warning("Not enough repeat purchase data for CLV analysis. The model requires customers with at least 2 purchases.")
                
                # Show basic monetary analysis instead
                st.subheader("Customer Value Analysis")
                
                # Aggregate by customer
                customer_value = df.groupby('CustomerID').agg({
                    'InvoiceNo': 'nunique',
                    'TotalPrice': 'sum'
                }).reset_index()
                
                customer_value.columns = ['CustomerID', 'PurchaseCount', 'TotalSpend']
                customer_value['AvgOrderValue'] = customer_value['TotalSpend'] / customer_value['PurchaseCount']
                
                # Display customer value distribution
                fig = px.scatter(
                    customer_value,
                    x='PurchaseCount',
                    y='TotalSpend',
                    size='AvgOrderValue',
                    hover_data=['CustomerID'],
                    title='Customer Value Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top customers
                st.write("### Top Customers by Total Spend")
                st.dataframe(customer_value.sort_values('TotalSpend', ascending=False).head(20))
    else:
        st.info("Please upload a CSV file with your customer transaction data.")

else:
    # Show intro page when no file is uploaded
    st.write("""
    # Welcome to the AI-Powered Predictive Revenue Optimization & Customer Retention Engine
    
    This application helps businesses identify revenue opportunities, prevent customer churn, 
    and maximize customer lifetime value using advanced analytics and machine learning.
    
    ## Key Features
    
    - **Data Overview**: Analyze your sales and customer data with interactive visualizations
    - **Customer Segmentation**: Segment customers using RFM analysis and K-means clustering
    - **Churn Prediction**: Identify customers at risk of churning before they leave
    - **Revenue Forecast**: Predict future revenue with advanced time series modeling
    - **Product Recommendations**: Generate personalized product recommendations
    - **Customer Lifetime Value**: Calculate and optimize customer lifetime value
    
    ## How to Use
    
    1. Upload your customer transaction data using the file uploader in the sidebar
    2. Navigate between different analyses using the page selector
    3. Explore interactive visualizations and download insights for your business
    
    ## Required Data Format
    
    Your CSV file should contain the following columns:
    - InvoiceNo: Invoice number
    - StockCode: Product code
    - Description: Product description
    - Quantity: Quantity purchased
    - InvoiceDate: Date of purchase
    - UnitPrice: Price per unit
    - CustomerID: Customer identifier
    - Country: Customer's country
    
    ## Get Started
    
    Upload your data using the file uploader in the sidebar to begin analyzing your business!
    """)
    
    # Example data structure
    st.write("### Example Data Format")
    example_data = pd.DataFrame({
        'InvoiceNo': ['536365', '536365', '536366', '536367', '536368'],
        'StockCode': ['85123A', '71053', '84406B', '84029G', '84029E'],
        'Description': ['WHITE HANGING HEART T-LIGHT HOLDER', 'WHITE METAL LANTERN', 'CREAM CUPID HEARTS COAT HANGER', 'KNITTED UNION FLAG HOT WATER BOTTLE', 'RED WOOLLY HOTTIE WHITE HEART'],
        'Quantity': [6, 6, 8, 6, 6],
        'InvoiceDate': ['12/1/2010 8:26', '12/1/2010 8:26', '12/1/2010 8:28', '12/1/2010 8:34', '12/1/2010 8:34'],
        'UnitPrice': [2.55, 3.39, 2.75, 3.39, 3.39],
        'CustomerID': [17850, 17850, 17851, 13047, 13047],
        'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom']
    })
    
    st.dataframe(example_data)