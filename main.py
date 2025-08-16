"""
Advanced Sales Analysis System - Main Execution Script
Professional data analysis pipeline for retail sales data
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Import custom modules
from sales_analyzer import SalesAnalyzer
from visualizer import SalesVisualizer
import config

def setup_directories():
    """Create necessary directories for outputs and logs"""
    for directory in [config.OUTPUT_DIR, config.LOGS_DIR]:
        directory.mkdir(exist_ok=True)

def print_section_header(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)

def print_results_summary(results: dict):
    """Print formatted results summary"""
    print(f"\n📊 Analysis Results Summary:")
    print(f"   • Dataset Shape: {results.get('shape', 'N/A')}")
    print(f"   • Total Revenue: ${results.get('total_revenue', 0):,.2f}")
    print(f"   • Total Profit: ${results.get('total_profit', 0):,.2f}")
    print(f"   • Profit Margin: {results.get('profit_margin', 0)*100:.2f}%")
    print(f"   • Unique Customers: {results.get('unique_customers', 0):,}")
    print(f"   • Unique Orders: {results.get('unique_orders', 0):,}")
    print(f"   • Date Range: {results.get('date_range', {}).get('start', 'N/A')} to {results.get('date_range', {}).get('end', 'N/A')}")

def main():
    """Main execution function"""
    try:
        # Setup
        setup_directories()
        print_section_header("ADVANCED SALES ANALYSIS SYSTEM")
        print("🚀 Initializing comprehensive sales analysis...")
        
        # Initialize analyzer
        analyzer = SalesAnalyzer()
        visualizer = SalesVisualizer(analyzer)
        
        # 1. Data Loading and Validation
        print_section_header("DATA LOADING & VALIDATION")
        print("📁 Loading and validating sales data...")
        analyzer.load_and_validate_data()
        print(f"✅ Data loaded successfully: {analyzer.df.shape[0]:,} rows, {analyzer.df.shape[1]} columns")
        
        # 2. Data Preprocessing
        print_section_header("DATA PREPROCESSING")
        print("🔧 Preprocessing data and engineering features...")
        analyzer.preprocess_data()
        print("✅ Data preprocessing completed")
        
        # 3. Exploratory Data Analysis
        print_section_header("EXPLORATORY DATA ANALYSIS")
        print("🔍 Performing comprehensive exploratory data analysis...")
        eda_results = analyzer.perform_eda()
        print_results_summary(eda_results)
        
        # 4. Correlation Analysis
        print_section_header("CORRELATION ANALYSIS")
        print("📈 Analyzing correlations between variables...")
        correlation_matrix = analyzer.analyze_correlations()
        print("✅ Correlation analysis completed")
        
        # Create correlation heatmap
        visualizer.plot_advanced_correlation_heatmap(correlation_matrix)
        print("📊 Advanced correlation heatmap generated")
        
        # 5. Customer Segmentation
        print_section_header("CUSTOMER SEGMENTATION")
        print("👥 Performing RFM analysis and customer segmentation...")
        customer_metrics = analyzer.customer_segmentation_rfm()
        print(f"✅ {len(customer_metrics)} customers segmented")
        
        # Print segment distribution
        segment_distribution = customer_metrics['Segment'].value_counts()
        print("\n📊 Customer Segment Distribution:")
        for segment, count in segment_distribution.items():
            print(f"   • {segment}: {count:,} customers ({count/len(customer_metrics)*100:.1f}%)")
        
        # 6. Advanced Clustering
        print_section_header("ADVANCED CLUSTERING")
        print("🎯 Applying K-means clustering for customer grouping...")
        analyzer.advanced_clustering()
        print("✅ K-means clustering completed")
        
        # Create 3D customer segmentation plot
        visualizer.plot_customer_segmentation_3d()
        print("📊 3D customer segmentation visualization generated")
        
        # 7. Cohort Analysis
        print_section_header("COHORT ANALYSIS")
        print("📅 Performing cohort analysis for customer retention...")
        cohort_table = analyzer.cohort_analysis()
        print("✅ Cohort analysis completed")
        
        # Create cohort heatmap
        visualizer.plot_cohort_heatmap()
        print("📊 Cohort retention heatmap generated")
        
        # 8. Market Basket Analysis
        print_section_header("MARKET BASKET ANALYSIS")
        print("🛒 Analyzing product associations and market basket patterns...")
        market_basket_rules = analyzer.market_basket_analysis()
        
        if not market_basket_rules.empty:
            print(f"✅ Found {len(market_basket_rules)} association rules")
            print("\n🔝 Top 5 Association Rules:")
            top_rules = market_basket_rules.head(5)
            for idx, rule in top_rules.iterrows():
                print(f"   • {list(rule['antecedents'])} → {list(rule['consequents'])} "
                      f"(Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f})")
        else:
            print("⚠️  No significant association rules found")
        
        # Create market basket visualization
        visualizer.create_market_basket_network()
        print("📊 Market basket analysis visualization generated")
        
        # 9. Predictive Modeling
        print_section_header("PREDICTIVE MODELING")
        print("🤖 Building predictive models for sales and profit forecasting...")
        model_results = analyzer.predictive_modeling()
        print("✅ Predictive models trained successfully")
        
        print("\n📊 Model Performance:")
        print(f"   Sales Model R²: {model_results['sales_model']['r2']:.3f}")
        print(f"   Profit Model R²: {model_results['profit_model']['r2']:.3f}")
        
        # Create model performance visualization
        visualizer.plot_predictive_model_performance(model_results)
        print("📊 Model performance visualization generated")
        
        # 10. KPI Calculation
        print_section_header("KEY PERFORMANCE INDICATORS")
        print("📊 Calculating comprehensive business KPIs...")
        kpis = analyzer.calculate_kpis()
        print("✅ KPI calculation completed")
        
        print("\n📈 Key Performance Indicators:")
        print(f"   • Monthly Growth Rate: {kpis.get('monthly_growth_rate', 0)*100:.2f}%")
        print(f"   • Average Order Value: ${kpis.get('avg_order_value', 0):.2f}")
        print(f"   • Orders per Customer: {kpis.get('orders_per_customer', 0):.1f}")
        print(f"   • Average Shipping Days: {kpis.get('avg_shipping_days', 0):.1f}")
        
        # Create KPI dashboard
        visualizer.plot_kpi_dashboard()
        print("📊 KPI dashboard generated")
        
        # 11. Executive Dashboard
        print_section_header("EXECUTIVE DASHBOARD")
        print("📈 Creating comprehensive executive dashboard...")
        visualizer.create_executive_dashboard()
        print("✅ Executive dashboard generated")
        
        # 12. Business Recommendations
        print_section_header("BUSINESS RECOMMENDATIONS")
        print("💡 Generating data-driven business recommendations...")
        
        # Generate recommendations based on analysis
        recommendations = []
        
        # Profit margin recommendations
        if kpis.get('profit_margin', 0) < config.KPI_TARGETS['profit_margin']:
            recommendations.append("🎯 Focus on improving profit margins through pricing optimization or cost reduction")
        
        # Customer segmentation recommendations
        if 'At Risk' in customer_metrics['Segment'].values:
            at_risk_count = len(customer_metrics[customer_metrics['Segment'] == 'At Risk'])
            recommendations.append(f"🚨 Implement retention strategies for {at_risk_count} at-risk customers")
        
        # Growth recommendations
        if kpis.get('monthly_growth_rate', 0) < config.KPI_TARGETS['monthly_growth']:
            recommendations.append("📈 Develop growth strategies to achieve target monthly growth rate")
        
        # Market basket recommendations
        if not market_basket_rules.empty:
            recommendations.append("🛒 Implement cross-selling strategies based on market basket analysis findings")
        
        print("\n💼 Strategic Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # 13. Export Summary
        print_section_header("EXPORT SUMMARY")
        print("📁 Analysis artifacts exported to:")
        print(f"   • Visualizations: {config.OUTPUT_DIR}")
        print(f"   • Logs: {config.LOGS_DIR}")
        print(f"   • Formats: {', '.join(config.EXPORT_FORMATS)}")
        
        print_section_header("ANALYSIS COMPLETE")
        print("🎉 Advanced sales analysis completed successfully!")
        print("📊 All visualizations and reports have been generated.")
        print(f"📁 Check the '{config.OUTPUT_DIR}' directory for exported files.")
        
        return True
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)