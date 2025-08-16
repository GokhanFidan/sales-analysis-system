"""
Advanced Sales Analysis System
Professional-grade data analysis for retail sales data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno as msno
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from operator import attrgetter

# Statistical and ML imports
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Market basket analysis
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import config

warnings.filterwarnings('ignore')

class SalesAnalyzer:
    """
    Comprehensive sales analysis system with advanced analytics capabilities
    """
    
    def __init__(self, data_path: str = None):
        """Initialize the SalesAnalyzer with data loading and setup"""
        self.data_path = data_path or config.DATA_FILE
        self.df = None
        self.processed_df = None
        self.customer_metrics = None
        self.setup_logging()
        self.setup_plotting()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),
            format=config.LOG_FORMAT,
            handlers=[
                logging.FileHandler(config.LOGS_DIR / 'sales_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_plotting(self):
        """Setup plotting style and configuration"""
        plt.style.use(config.PLOT_STYLE)
        sns.set_palette("husl")
        
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate the sales data with error handling"""
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            
            if not Path(self.data_path).exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
                
            self.df = pd.read_csv(self.data_path, encoding='latin1')
            
            # Data validation
            required_columns = ['Order ID', 'Customer ID', 'Sales', 'Profit', 'Order Date']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            self.logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the data with feature engineering"""
        try:
            self.logger.info("Starting data preprocessing")
            
            df = self.df.copy()
            
            # Date processing
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            df['Ship Date'] = pd.to_datetime(df['Ship Date'])
            
            # Feature engineering
            df['Days_to_Ship'] = (df['Ship Date'] - df['Order Date']).dt.days
            df['Profit_Margin'] = df['Profit'] / df['Sales']
            df['Order_Year'] = df['Order Date'].dt.year
            df['Order_Month'] = df['Order Date'].dt.month
            df['Order_Quarter'] = df['Order Date'].dt.quarter
            df['Day_of_Week'] = df['Order Date'].dt.day_name()
            df['Month_Name'] = df['Order Date'].dt.month_name()
            df['Is_Weekend'] = df['Order Date'].dt.weekday >= 5
            
            # Handle infinite and null values
            df = df.replace([np.inf, -np.inf], np.nan)
            df['Profit_Margin'] = df['Profit_Margin'].fillna(0)
            
            # Add business metrics
            df['Revenue_per_Quantity'] = df['Sales'] / df['Quantity']
            df['Discount_Impact'] = df['Discount'] * df['Sales']
            
            self.processed_df = df
            self.logger.info("Data preprocessing completed")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def perform_eda(self) -> Dict:
        """Perform comprehensive exploratory data analysis"""
        try:
            self.logger.info("Performing exploratory data analysis")
            
            eda_results = {}
            
            # Basic statistics
            eda_results['shape'] = self.processed_df.shape
            eda_results['columns'] = self.processed_df.columns.tolist()
            eda_results['data_types'] = self.processed_df.dtypes.to_dict()
            eda_results['missing_values'] = self.processed_df.isnull().sum().to_dict()
            eda_results['duplicates'] = self.processed_df.duplicated().sum()
            
            # Business metrics
            eda_results['total_revenue'] = self.processed_df['Sales'].sum()
            eda_results['total_profit'] = self.processed_df['Profit'].sum()
            eda_results['profit_margin'] = eda_results['total_profit'] / eda_results['total_revenue']
            eda_results['unique_customers'] = self.processed_df['Customer ID'].nunique()
            eda_results['unique_orders'] = self.processed_df['Order ID'].nunique()
            eda_results['avg_order_value'] = self.processed_df['Sales'].mean()
            
            # Date range
            eda_results['date_range'] = {
                'start': self.processed_df['Order Date'].min(),
                'end': self.processed_df['Order Date'].max(),
                'days': (self.processed_df['Order Date'].max() - self.processed_df['Order Date'].min()).days
            }
            
            self.logger.info("EDA completed successfully")
            return eda_results
            
        except Exception as e:
            self.logger.error(f"Error in EDA: {str(e)}")
            raise
            
    def analyze_correlations(self) -> pd.DataFrame:
        """Analyze correlations between numerical variables"""
        try:
            numerical_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 
                            'Days_to_Ship', 'Profit_Margin', 'Revenue_per_Quantity']
            
            correlation_matrix = self.processed_df[numerical_cols].corr()
            
            # Statistical significance testing
            significant_correlations = []
            for i, col1 in enumerate(numerical_cols):
                for j, col2 in enumerate(numerical_cols[i+1:], i+1):
                    corr, p_value = stats.pearsonr(
                        self.processed_df[col1].dropna(), 
                        self.processed_df[col2].dropna()
                    )
                    if p_value < config.SIGNIFICANCE_LEVEL:
                        significant_correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': corr,
                            'p_value': p_value
                        })
            
            self.significant_correlations = pd.DataFrame(significant_correlations)
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {str(e)}")
            raise
            
    def customer_segmentation_rfm(self) -> pd.DataFrame:
        """Perform RFM analysis for customer segmentation"""
        try:
            self.logger.info("Performing RFM customer segmentation")
            
            # Calculate RFM metrics
            customer_metrics = self.processed_df.groupby('Customer ID').agg({
                'Sales': ['sum', 'mean', 'count'],
                'Profit': 'sum',
                'Quantity': 'sum',
                'Order Date': ['min', 'max']
            }).reset_index()
            
            customer_metrics.columns = ['Customer_ID', 'Total_Sales', 'Avg_Order_Value', 
                                      'Order_Frequency', 'Total_Profit', 'Total_Quantity', 
                                      'First_Order', 'Last_Order']
            
            # Calculate recency
            analysis_date = self.processed_df['Order Date'].max()
            customer_metrics['Recency_Days'] = (analysis_date - customer_metrics['Last_Order']).dt.days
            customer_metrics['Customer_Lifetime_Days'] = (customer_metrics['Last_Order'] - customer_metrics['First_Order']).dt.days
            
            # RFM Scoring
            customer_metrics['R_Score'] = pd.qcut(customer_metrics['Recency_Days'].rank(method='first'), 5, labels=[5,4,3,2,1])
            customer_metrics['F_Score'] = pd.qcut(customer_metrics['Order_Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
            customer_metrics['M_Score'] = pd.qcut(customer_metrics['Total_Sales'].rank(method='first'), 5, labels=[1,2,3,4,5])
            
            customer_metrics['RFM_Score'] = (customer_metrics['R_Score'].astype(str) + 
                                           customer_metrics['F_Score'].astype(str) + 
                                           customer_metrics['M_Score'].astype(str))
            
            # Segment assignment
            def assign_segment(rfm_score):
                for segment, scores in config.CUSTOMER_SEGMENTS.items():
                    if rfm_score in scores:
                        return segment.replace('_', ' ').title()
                return 'Others'
            
            customer_metrics['Segment'] = customer_metrics['RFM_Score'].apply(assign_segment)
            
            self.customer_metrics = customer_metrics
            self.logger.info("RFM segmentation completed")
            return customer_metrics
            
        except Exception as e:
            self.logger.error(f"Error in RFM segmentation: {str(e)}")
            raise
            
    def advanced_clustering(self) -> pd.DataFrame:
        """Perform K-means clustering for customer segmentation"""
        try:
            if self.customer_metrics is None:
                self.customer_segmentation_rfm()
                
            clustering_features = ['Total_Sales', 'Order_Frequency', 'Total_Profit', 'Recency_Days']
            X = self.customer_metrics[clustering_features].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply K-means
            kmeans = KMeans(n_clusters=config.KMEANS_CLUSTERS, random_state=config.RANDOM_STATE, n_init=10)
            self.customer_metrics['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Calculate cluster statistics
            cluster_stats = self.customer_metrics.groupby('KMeans_Cluster')[clustering_features].mean()
            
            self.cluster_stats = cluster_stats
            self.scaler = scaler
            self.kmeans_model = kmeans
            
            return self.customer_metrics
            
        except Exception as e:
            self.logger.error(f"Error in K-means clustering: {str(e)}")
            raise
            
    def cohort_analysis(self) -> pd.DataFrame:
        """Perform cohort analysis to understand customer retention"""
        try:
            self.logger.info("Performing cohort analysis")
            
            df = self.processed_df.copy()
            
            # Create order period and cohort group
            df['Order_Period'] = df['Order Date'].dt.to_period('M')
            df['Cohort_Group'] = df.groupby('Customer ID')['Order Date'].transform('min').dt.to_period('M')
            
            # Calculate period number
            df['Period_Number'] = (df['Order_Period'] - df['Cohort_Group']).apply(attrgetter('n'))
            
            # Create cohort table
            cohort_data = df.groupby(['Cohort_Group', 'Period_Number'])['Customer ID'].nunique().reset_index()
            cohort_counts = cohort_data.pivot(index='Cohort_Group', 
                                             columns='Period_Number', 
                                             values='Customer ID')
            
            # Calculate cohort sizes and retention rates
            cohort_sizes = df.groupby('Cohort_Group')['Customer ID'].nunique()
            cohort_table = cohort_counts.divide(cohort_sizes, axis=0)
            
            self.cohort_table = cohort_table
            self.logger.info("Cohort analysis completed")
            return cohort_table
            
        except Exception as e:
            self.logger.error(f"Error in cohort analysis: {str(e)}")
            raise
            
    def market_basket_analysis(self) -> pd.DataFrame:
        """Perform market basket analysis to find product associations"""
        try:
            self.logger.info("Performing market basket analysis")
            
            # Create transaction matrix
            basket = (self.processed_df.groupby(['Order ID', 'Sub-Category'])['Quantity']
                     .sum().unstack().fillna(0))
            
            # Convert to binary matrix
            basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
            
            # Find frequent itemsets
            frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
            
            # Generate association rules
            if not frequent_itemsets.empty:
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                rules = rules.sort_values('lift', ascending=False)
                
                self.market_basket_rules = rules
                self.logger.info(f"Found {len(rules)} association rules")
                return rules
            else:
                self.logger.warning("No frequent itemsets found")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error in market basket analysis: {str(e)}")
            raise
            
    def predictive_modeling(self) -> Dict:
        """Build predictive models for sales forecasting"""
        try:
            self.logger.info("Building predictive models")
            
            # Prepare features for modeling
            model_df = self.processed_df.copy()
            
            # Encode categorical variables
            le_category = LabelEncoder()
            le_segment = LabelEncoder()
            le_region = LabelEncoder()
            
            model_df['Category_Encoded'] = le_category.fit_transform(model_df['Category'])
            model_df['Segment_Encoded'] = le_segment.fit_transform(model_df['Segment'])
            model_df['Region_Encoded'] = le_region.fit_transform(model_df['Region'])
            
            # Select features
            features = ['Quantity', 'Discount', 'Category_Encoded', 'Segment_Encoded', 
                       'Region_Encoded', 'Order_Month', 'Order_Quarter', 'Is_Weekend']
            
            X = model_df[features]
            y_sales = model_df['Sales']
            y_profit = model_df['Profit']
            
            # Split data
            X_train, X_test, y_sales_train, y_sales_test = train_test_split(
                X, y_sales, test_size=0.2, random_state=config.RANDOM_STATE
            )
            _, _, y_profit_train, y_profit_test = train_test_split(
                X, y_profit, test_size=0.2, random_state=config.RANDOM_STATE
            )
            
            # Train models
            sales_model = RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE)
            profit_model = RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE)
            
            sales_model.fit(X_train, y_sales_train)
            profit_model.fit(X_train, y_profit_train)
            
            # Evaluate models
            sales_pred = sales_model.predict(X_test)
            profit_pred = profit_model.predict(X_test)
            
            model_results = {
                'sales_model': {
                    'mae': mean_absolute_error(y_sales_test, sales_pred),
                    'mse': mean_squared_error(y_sales_test, sales_pred),
                    'r2': r2_score(y_sales_test, sales_pred),
                    'feature_importance': dict(zip(features, sales_model.feature_importances_))
                },
                'profit_model': {
                    'mae': mean_absolute_error(y_profit_test, profit_pred),
                    'mse': mean_squared_error(y_profit_test, profit_pred),
                    'r2': r2_score(y_profit_test, profit_pred),
                    'feature_importance': dict(zip(features, profit_model.feature_importances_))
                }
            }
            
            self.sales_model = sales_model
            self.profit_model = profit_model
            self.label_encoders = {
                'category': le_category,
                'segment': le_segment,
                'region': le_region
            }
            
            self.logger.info("Predictive modeling completed")
            return model_results
            
        except Exception as e:
            self.logger.error(f"Error in predictive modeling: {str(e)}")
            raise
            
    def calculate_kpis(self) -> Dict:
        """Calculate key performance indicators"""
        try:
            self.logger.info("Calculating KPIs")
            
            kpis = {}
            
            # Revenue KPIs
            kpis['total_revenue'] = self.processed_df['Sales'].sum()
            kpis['total_profit'] = self.processed_df['Profit'].sum()
            kpis['profit_margin'] = kpis['total_profit'] / kpis['total_revenue']
            kpis['avg_order_value'] = self.processed_df['Sales'].mean()
            
            # Customer KPIs
            kpis['total_customers'] = self.processed_df['Customer ID'].nunique()
            kpis['total_orders'] = self.processed_df['Order ID'].nunique()
            kpis['orders_per_customer'] = kpis['total_orders'] / kpis['total_customers']
            
            # Time-based KPIs
            monthly_sales = self.processed_df.groupby('Order_Month')['Sales'].sum()
            kpis['monthly_growth_rate'] = monthly_sales.pct_change().mean()
            
            # Product KPIs
            kpis['avg_discount'] = self.processed_df['Discount'].mean()
            kpis['avg_shipping_days'] = self.processed_df['Days_to_Ship'].mean()
            
            # Performance vs targets
            kpis['performance_vs_targets'] = {}
            for kpi, target in config.KPI_TARGETS.items():
                if kpi in kpis:
                    kpis['performance_vs_targets'][kpi] = {
                        'actual': kpis[kpi],
                        'target': target,
                        'performance': kpis[kpi] / target
                    }
            
            self.kpis = kpis
            self.logger.info("KPI calculation completed")
            return kpis
            
        except Exception as e:
            self.logger.error(f"Error calculating KPIs: {str(e)}")
            raise