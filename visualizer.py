"""
Advanced Visualization System for Sales Analysis
Professional-grade charts and dashboards with export functionality
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from operator import attrgetter

import config

class SalesVisualizer:
    """
    Advanced visualization system with professional charts and export capabilities
    """
    
    def __init__(self, analyzer=None):
        """Initialize the visualizer with analyzer instance"""
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)
        self.setup_style()
        
    def setup_style(self):
        """Setup professional plotting style"""
        plt.style.use(config.PLOT_STYLE)
        sns.set_palette("husl")
        
        # Custom color palette
        self.colors = config.COLOR_PALETTE
        self.color_sequence = [self.colors['primary'], self.colors['secondary'], 
                              self.colors['accent'], self.colors['success']]
        
    def save_plot(self, fig, filename: str, formats: List[str] = None):
        """Save plot in multiple formats with high quality"""
        try:
            formats = formats or config.EXPORT_FORMATS
            
            for fmt in formats:
                if fmt == 'png' and hasattr(fig, 'savefig'):
                    # Matplotlib figure
                    fig.savefig(config.OUTPUT_DIR / f"{filename}.png", 
                               dpi=config.DPI, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                elif fmt == 'pdf' and hasattr(fig, 'savefig'):
                    # Matplotlib figure
                    fig.savefig(config.OUTPUT_DIR / f"{filename}.pdf", 
                               bbox_inches='tight', facecolor='white')
                elif fmt == 'html' and hasattr(fig, 'write_html'):
                    # Plotly figure
                    fig.write_html(config.OUTPUT_DIR / f"{filename}.html")
                elif fmt == 'png' and hasattr(fig, 'write_image'):
                    # Plotly figure to PNG
                    fig.write_image(config.OUTPUT_DIR / f"{filename}.png")
                    
            self.logger.info(f"Plot saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving plot {filename}: {str(e)}")
            
    def create_executive_dashboard(self) -> go.Figure:
        """Create comprehensive executive dashboard"""
        try:
            df = self.analyzer.processed_df
            
            # Create subplot structure
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=[
                    'Revenue Trend', 'Category Performance', 'Regional Distribution',
                    'Customer Segments', 'Profit Margin Analysis', 'Monthly Growth',
                    'Top Products', 'Seasonal Patterns', 'Key Metrics'
                ],
                specs=[
                    [{"type": "scatter"}, {"type": "bar"}, {"type": "pie"}],
                    [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "scatter"}, {"type": "table"}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.05
            )
            
            # 1. Revenue Trend
            monthly_revenue = df.groupby('Order_Month')['Sales'].sum()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_labels = [month_names[i-1] for i in monthly_revenue.index]
            fig.add_trace(
                go.Scatter(x=month_labels, y=monthly_revenue.values,
                          mode='lines+markers', name='Revenue Trend',
                          line=dict(color=self.colors['primary'], width=3)),
                row=1, col=1
            )
            
            # 2. Category Performance
            category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=True)
            fig.add_trace(
                go.Bar(y=category_sales.index, x=category_sales.values,
                      orientation='h', name='Category Sales',
                      marker_color=self.colors['secondary']),
                row=1, col=2
            )
            
            # 3. Regional Distribution
            regional_sales = df.groupby('Region')['Sales'].sum()
            fig.add_trace(
                go.Pie(labels=regional_sales.index, values=regional_sales.values,
                      name="Regional Sales", hole=0.4),
                row=1, col=3
            )
            
            # 4. Customer Segments (if available)
            if hasattr(self.analyzer, 'customer_metrics') and self.analyzer.customer_metrics is not None:
                segment_counts = self.analyzer.customer_metrics['Segment'].value_counts()
                fig.add_trace(
                    go.Bar(x=segment_counts.index, y=segment_counts.values,
                          name='Customer Segments',
                          marker_color=self.colors['accent']),
                    row=2, col=1
                )
            
            # 5. Profit Margin Analysis
            fig.add_trace(
                go.Scatter(x=df['Sales'], y=df['Profit'],
                          mode='markers', name='Profit vs Sales',
                          marker=dict(color=self.colors['success'], size=4, opacity=0.6)),
                row=2, col=2
            )
            
            # 6. Monthly Growth
            monthly_growth = monthly_revenue.pct_change().fillna(0) * 100
            growth_labels = [month_names[i-1] for i in monthly_growth.index]
            fig.add_trace(
                go.Bar(x=growth_labels, y=monthly_growth.values,
                      name='Monthly Growth %',
                      marker_color=np.where(monthly_growth.values >= 0, self.colors['primary'], self.colors['success'])),
                row=2, col=3
            )
            
            # 7. Top Products
            top_products = df.groupby('Sub-Category')['Sales'].sum().nlargest(10)
            # Format values in thousands
            formatted_values = [f'${v/1000:.0f}K' for v in top_products.values]
            fig.add_trace(
                go.Bar(y=top_products.index, x=top_products.values,
                      orientation='h', name='Top Products (Sales $)',
                      marker_color=self.colors['neutral'],
                      text=formatted_values, textposition='auto'),
                row=3, col=1
            )
            
            # 8. Seasonal Patterns
            seasonal_data = df.groupby('Month_Name')['Sales'].mean()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            seasonal_ordered = seasonal_data.reindex([m for m in month_order if m in seasonal_data.index])
            
            fig.add_trace(
                go.Scatter(x=list(seasonal_ordered.index), y=seasonal_ordered.values,
                          mode='lines+markers', name='Seasonal Pattern',
                          line=dict(color=self.colors['accent'], width=2)),
                row=3, col=2
            )
            
            # 9. Key Metrics Table
            if hasattr(self.analyzer, 'kpis') and self.analyzer.kpis:
                kpis = self.analyzer.kpis
                metrics_data = [
                    ['Total Revenue', f"${kpis.get('total_revenue', 0):,.0f}"],
                    ['Total Profit', f"${kpis.get('total_profit', 0):,.0f}"],
                    ['Profit Margin', f"{kpis.get('profit_margin', 0)*100:.1f}%"],
                    ['Avg Order Value', f"${kpis.get('avg_order_value', 0):.0f}"],
                    ['Total Customers', f"{kpis.get('total_customers', 0):,}"],
                    ['Total Orders', f"{kpis.get('total_orders', 0):,}"]
                ]
                
                fig.add_trace(
                    go.Table(
                        header=dict(values=['Metric', 'Value'],
                                   fill_color=self.colors['primary'],
                                   font_color='white'),
                        cells=dict(values=list(zip(*metrics_data)),
                                  fill_color='white',
                                  font_color='black')
                    ),
                    row=3, col=3
                )
            
            # Update layout with better axis labels
            fig.update_layout(
                height=1200,
                showlegend=False,
                title_text="Executive Sales Dashboard",
                title_x=0.5,
                title_font_size=24,
                font=dict(size=10)
            )
            
            # Update x-axis labels
            fig.update_xaxes(title_text="Month", row=1, col=1)
            fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
            
            fig.update_xaxes(title_text="Month", row=2, col=3)
            fig.update_yaxes(title_text="Growth Rate (%)", row=2, col=3)
            
            fig.update_xaxes(title_text="Sales ($)", row=3, col=1)
            fig.update_yaxes(title_text="Sub-Category", row=3, col=1)
            
            fig.update_xaxes(title_text="Sales ($)", row=2, col=2)
            fig.update_yaxes(title_text="Profit ($)", row=2, col=2)
            
            # Save dashboard
            self.save_plot(fig, 'executive_dashboard', ['html', 'png'])
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating executive dashboard: {str(e)}")
            raise
            
    def plot_advanced_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """Create advanced correlation heatmap with statistical significance"""
        try:
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            masked_corr = correlation_matrix.mask(mask)
            
            fig = go.Figure(data=go.Heatmap(
                z=masked_corr.values,
                x=masked_corr.columns,
                y=masked_corr.index,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(masked_corr.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation Coefficient")
            ))
            
            fig.update_layout(
                title="Advanced Correlation Analysis",
                title_x=0.5,
                width=800,
                height=600,
                font=dict(size=12)
            )
            
            self.save_plot(fig, 'correlation_heatmap')
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            raise
            
    def plot_customer_segmentation_3d(self) -> go.Figure:
        """Create 3D customer segmentation visualization"""
        try:
            if not hasattr(self.analyzer, 'customer_metrics') or self.analyzer.customer_metrics is None:
                raise ValueError("Customer metrics not available. Run customer_segmentation_rfm first.")
                
            customer_data = self.analyzer.customer_metrics
            
            fig = go.Figure(data=[go.Scatter3d(
                x=customer_data['Total_Sales'],
                y=customer_data['Order_Frequency'],
                z=customer_data['Recency_Days'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=customer_data['KMeans_Cluster'] if 'KMeans_Cluster' in customer_data.columns else customer_data.index,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Cluster")
                ),
                text=customer_data['Segment'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Total Sales: $%{x:,.0f}<br>' +
                             'Order Frequency: %{y}<br>' +
                             'Recency: %{z} days<br>' +
                             '<extra></extra>'
            )])
            
            fig.update_layout(
                title="3D Customer Segmentation Analysis",
                scene=dict(
                    xaxis_title='Total Sales ($)',
                    yaxis_title='Order Frequency',
                    zaxis_title='Recency (Days)'
                ),
                width=900,
                height=700
            )
            
            self.save_plot(fig, 'customer_segmentation_3d')
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating 3D customer segmentation: {str(e)}")
            raise
            
    def plot_cohort_heatmap(self) -> go.Figure:
        """Create cohort retention heatmap"""
        try:
            if not hasattr(self.analyzer, 'cohort_table'):
                self.analyzer.cohort_analysis()
                
            cohort_table = self.analyzer.cohort_table
            
            fig = go.Figure(data=go.Heatmap(
                z=cohort_table.values,
                x=[f'Period {i}' for i in cohort_table.columns],
                y=[str(period) for period in cohort_table.index],
                colorscale='Blues',
                text=np.round(cohort_table.values * 100, 1),
                texttemplate="%{text}%",
                textfont={"size": 10},
                colorbar=dict(title="Retention Rate")
            ))
            
            fig.update_layout(
                title="Customer Cohort Retention Analysis",
                title_x=0.5,
                xaxis_title="Period Number",
                yaxis_title="Cohort Month",
                width=1000,
                height=600
            )
            
            self.save_plot(fig, 'cohort_analysis')
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating cohort heatmap: {str(e)}")
            raise
            
    def plot_predictive_model_performance(self, model_results: Dict) -> go.Figure:
        """Visualize predictive model performance"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Sales Model Performance', 'Profit Model Performance',
                               'Sales Feature Importance', 'Profit Feature Importance'],
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Model performance metrics
            sales_metrics = ['MAE', 'MSE', 'R²']
            sales_values = [
                model_results['sales_model']['mae'],
                model_results['sales_model']['mse'],
                model_results['sales_model']['r2']
            ]
            
            profit_metrics = ['MAE', 'MSE', 'R²']
            profit_values = [
                model_results['profit_model']['mae'],
                model_results['profit_model']['mse'],
                model_results['profit_model']['r2']
            ]
            
            # Add performance bars
            fig.add_trace(
                go.Bar(x=sales_metrics, y=sales_values, name='Sales Model',
                      marker_color=self.colors['primary']),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=profit_metrics, y=profit_values, name='Profit Model',
                      marker_color=self.colors['secondary']),
                row=1, col=2
            )
            
            # Feature importance
            sales_features = list(model_results['sales_model']['feature_importance'].keys())
            sales_importance = list(model_results['sales_model']['feature_importance'].values())
            
            profit_features = list(model_results['profit_model']['feature_importance'].keys())
            profit_importance = list(model_results['profit_model']['feature_importance'].values())
            
            fig.add_trace(
                go.Bar(x=sales_features, y=sales_importance, name='Sales Features',
                      marker_color=self.colors['accent']),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(x=profit_features, y=profit_importance, name='Profit Features',
                      marker_color=self.colors['success']),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text="Predictive Model Performance Analysis",
                title_x=0.5,
                showlegend=False
            )
            
            # Rotate x-axis labels for feature importance
            fig.update_xaxes(tickangle=45, row=2, col=1)
            fig.update_xaxes(tickangle=45, row=2, col=2)
            
            self.save_plot(fig, 'model_performance')
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating model performance plot: {str(e)}")
            raise
            
    def plot_kpi_dashboard(self) -> go.Figure:
        """Create KPI performance dashboard"""
        try:
            if not hasattr(self.analyzer, 'kpis'):
                self.analyzer.calculate_kpis()
                
            kpis = self.analyzer.kpis
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=['Revenue vs Target', 'Profit Margin vs Target', 'Customer Growth',
                               'Order Trends', 'Performance Scorecard', 'Monthly Metrics'],
                specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "table"}, {"type": "scatter"}]]
            )
            
            # Revenue indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=kpis['total_revenue'],
                    title={'text': "Total Revenue"},
                    delta={'reference': kpis['total_revenue'] * 0.9},
                    gauge={'axis': {'range': [None, kpis['total_revenue'] * 1.2]},
                           'bar': {'color': self.colors['primary']},
                           'steps': [{'range': [0, kpis['total_revenue'] * 0.5], 'color': "lightgray"},
                                    {'range': [kpis['total_revenue'] * 0.5, kpis['total_revenue']], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': kpis['total_revenue'] * 0.9}}
                ),
                row=1, col=1
            )
            
            # Profit margin indicator
            target_margin = config.KPI_TARGETS.get('profit_margin', 0.15)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=kpis['profit_margin'],
                    title={'text': "Profit Margin"},
                    number={'suffix': "%", 'valueformat': ".1%"},
                    delta={'reference': target_margin, 'valueformat': ".1%"},
                    gauge={'axis': {'range': [None, 0.3]},
                           'bar': {'color': self.colors['secondary']},
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': target_margin}}
                ),
                row=1, col=2
            )
            
            # Additional visualizations would go here...
            # (Customer growth, order trends, etc.)
            
            fig.update_layout(
                height=800,
                title_text="Key Performance Indicators Dashboard",
                title_x=0.5
            )
            
            self.save_plot(fig, 'kpi_dashboard')
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating KPI dashboard: {str(e)}")
            raise
            
    def create_market_basket_network(self) -> go.Figure:
        """Create network visualization for market basket analysis"""
        try:
            if not hasattr(self.analyzer, 'market_basket_rules'):
                self.analyzer.market_basket_analysis()
                
            rules = self.analyzer.market_basket_rules
            
            if rules.empty:
                self.logger.warning("No market basket rules available for visualization")
                return go.Figure()
                
            # Create network graph (simplified version)
            # This would typically use networkx for more complex network analysis
            
            top_rules = rules.head(20)  # Top 20 rules
            
            fig = go.Figure()
            
            # Add scatter plot of rules
            fig.add_trace(go.Scatter(
                x=top_rules['support'],
                y=top_rules['confidence'],
                mode='markers',
                marker=dict(
                    size=top_rules['lift'] * 10,
                    color=top_rules['lift'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Lift")
                ),
                text=[f"Antecedents: {ant}<br>Consequents: {cons}" 
                      for ant, cons in zip(top_rules['antecedents'], top_rules['consequents'])],
                hovertemplate='%{text}<br>Support: %{x:.3f}<br>Confidence: %{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Market Basket Analysis - Association Rules",
                xaxis_title="Support",
                yaxis_title="Confidence",
                width=800,
                height=600
            )
            
            self.save_plot(fig, 'market_basket_analysis')
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating market basket network: {str(e)}")
            raise