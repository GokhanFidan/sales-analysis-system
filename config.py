"""
Configuration settings for the Sales Analysis project
"""

import os
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'
LOGS_DIR = BASE_DIR / 'logs'

# Data file
DATA_FILE = DATA_DIR / 'superstore.csv'

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8'
COLOR_PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#333366'
}

FIGURE_SIZE = {
    'small': (8, 6),
    'medium': (12, 8),
    'large': (15, 10),
    'extra_large': (20, 12)
}

# Analysis parameters
CUSTOMER_SEGMENTS = {
    'champions': ['555', '554', '544', '545', '454', '455', '445'],
    'loyal_customers': ['543', '444', '435', '355', '354', '345', '344', '335'],
    'potential_loyalists': ['512', '511', '422', '421', '412', '411', '311'],
    'new_customers': ['155', '154', '144', '214', '215', '115', '114'],
    'promising': ['255', '254', '245', '154', '245', '143', '142', '135', '125'],
    'need_attention': ['331', '321', '231', '241', '251'],
    'cannot_lose': ['155', '211', '111', '112', '121', '131', '141', '151'],
    'at_risk': ['155', '132', '123', '122', '212', '213']
}

# Statistical settings
CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_LEVEL = 0.05

# Clustering parameters
KMEANS_CLUSTERS = 4
RANDOM_STATE = 42

# Business KPIs
KPI_TARGETS = {
    'profit_margin': 0.15,  # 15%
    'customer_retention': 0.80,  # 80%
    'avg_order_value': 500,  # $500
    'monthly_growth': 0.05  # 5%
}

# Export settings
EXPORT_FORMATS = ['png', 'pdf', 'html']
DPI = 300

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create directories if they don't exist
for directory in [OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)