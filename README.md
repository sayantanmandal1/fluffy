# 🤖 AI-Powered Predictive Revenue Optimization & Customer Retention Engine

Unlock the full potential of your customer base using **machine learning-powered analytics**. This interactive Streamlit app provides actionable insights to **reduce churn**, **increase revenue**, and **optimize customer lifetime value**.

---

## 🚀 Key Features

### 📊 Data Overview
- Upload transaction data and visualize trends across products, countries, and customer segments.

### 🔍 Customer Segmentation
- **RFM Analysis**: Understand customer value with Recency, Frequency, and Monetary metrics.
- **K-Means Clustering**: Segment customers into behavior-based cohorts.

### ⚠️ Churn Prediction
- Predict which customers are most likely to churn using supervised learning models.

### 📈 Revenue Forecasting
- Forecast future sales trends using **Time Series Modeling**.

### 🛒 Product Recommendations
- Generate personalized recommendations using item-based collaborative filtering.

### 💰 Customer Lifetime Value (CLV)
- Calculate expected future value from customers to inform marketing strategy.

---

## 📂 Required Data Format

Upload a `.csv` file with the following columns:

| Column Name | Description |
|-------------|-------------|
| `InvoiceNo` | Invoice number (unique transaction ID) |
| `StockCode` | Product code |
| `Description` | Product description |
| `Quantity` | Units purchased |
| `InvoiceDate` | Date of transaction |
| `UnitPrice` | Price per unit |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer's country of origin |

---

## 🧠 How It Works

1. **Upload your dataset** using the sidebar in the app.
2. Use the **page selector** to navigate through different modules.
3. Analyze interactive plots, predictions, and KPIs.
4. **Download insights** to inform business decisions.

---

## 🛠 How to Run the App Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## 🎯 Use Cases

- SaaS and Subscription-based business churn modeling
- E-commerce revenue optimization and targeting
- Product analytics and recommendation engine
- Business intelligence dashboards for SMEs

---

## 📈 Sample Outputs

> Churn Probability: **87%** for Customer #17821  
> Predicted Revenue Next Month: **$4,324.50**  
> Top Product Recommendation: **"Wireless Mouse - Black Edition"**

---

## 🔐 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more information.

---

## 👨‍💻 Author

**Sayantan Mandal**  
Empowering businesses through data-driven intelligence.

---

## 🌐 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fluffy.streamlit.app/)
