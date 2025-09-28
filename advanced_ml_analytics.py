"""
Advanced Machine Learning Analytics Engine
Enterprise-grade ML pipeline with deep learning, ensemble methods,
automated feature engineering, and model explainability
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, mutual_info_regression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import TSNE, UMAP
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor, BayesianRidge, HuberRegressor
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score, silhouette_score
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.calibration import CalibratedClassifierCV

# Advanced ML Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Model Explainability
import shap
import lime
import lime.lime_tabular
from pdpbox import pdp
import eli5
from eli5.sklearn import PermutationImportance

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import prophet
from prophet import Prophet

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import hyperopt
from hyperopt import fmin, tpe, hp, Trials

# Model Monitoring
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of ML models"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"

class FeatureEngineeringMethod(Enum):
    """Feature engineering methods"""
    STATISTICAL = "statistical"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    AUTOMATED = "automated"
    DEEP_LEARNING = "deep_learning"

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    mse: float
    mae: float
    r2_score: float
    cross_val_score: float
    training_time: float
    prediction_time: float
    model_size: int
    feature_importance: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeatureImportance:
    """Feature importance analysis"""
    feature_name: str
    importance_score: float
    rank: int
    method: str
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

class AdvancedMLAnalytics:
    """Enterprise-grade machine learning analytics engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.explanation_models = {}
        self.feature_engineering_pipeline = None
        self.optimization_results = {}
        
        # Initialize MLflow for experiment tracking
        self._initialize_mlflow()
        
        # Initialize feature engineering pipeline
        self._initialize_feature_engineering()
        
        # Initialize model registry
        self._initialize_model_registry()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default ML configuration"""
        return {
            'experiment_name': 'retail_analytics',
            'random_state': 42,
            'test_size': 0.2,
            'validation_size': 0.1,
            'cv_folds': 5,
            'feature_selection': {
                'method': 'mutual_info',
                'k_best': 20,
                'threshold': 0.01
            },
            'hyperparameter_optimization': {
                'method': 'optuna',
                'n_trials': 100,
                'timeout': 3600
            },
            'model_explainability': {
                'shap': True,
                'lime': True,
                'pdp': True,
                'permutation': True
            }
        }
    
    def _initialize_mlflow(self):
        """Initialize MLflow for experiment tracking"""
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment(self.config['experiment_name'])
            logger.info("MLflow initialized successfully")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
    
    def _initialize_feature_engineering(self):
        """Initialize automated feature engineering pipeline"""
        self.feature_engineering_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('feature_selector', SelectKBest(k=self.config['feature_selection']['k_best']))
        ])
    
    def _initialize_model_registry(self):
        """Initialize model registry for versioning"""
        self.model_registry = {}
    
    def load_data(self, db_path: str = "retail_data.db") -> Dict[str, pd.DataFrame]:
        """Load data from database with advanced preprocessing"""
        logger.info("Loading data for ML analytics...")
        
        conn = sqlite3.connect(db_path)
        
        # Load all tables
        tables = ['customers', 'products', 'orders', 'order_items']
        data = {}
        
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            data[table] = df
            logger.info(f"Loaded {len(df)} records from {table}")
        
        conn.close()
        
        # Create comprehensive dataset
        comprehensive_data = self._create_comprehensive_dataset(data)
        
        return comprehensive_data
    
    def _create_comprehensive_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create comprehensive dataset for ML analysis"""
        logger.info("Creating comprehensive ML dataset...")
        
        # Merge all data
        df = data['order_items'].merge(data['products'], on='product_id', how='left')
        df = df.merge(data['orders'], on='order_id', how='left')
        df = df.merge(data['customers'], on='customer_id', how='left')
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        logger.info(f"Comprehensive dataset created: {len(df)} records, {len(df.columns)} features")
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for ML analysis"""
        logger.info("Adding derived features...")
        
        # Customer-level features
        customer_features = df.groupby('customer_id').agg({
            'net_total': ['sum', 'mean', 'count'],
            'quantity': 'sum',
            'order_date': ['min', 'max']
        }).reset_index()
        
        customer_features.columns = ['customer_id', 'total_spent', 'avg_order_value', 'total_orders', 
                                   'total_quantity', 'first_order', 'last_order']
        
        # Calculate additional customer metrics
        customer_features['days_since_first_order'] = (customer_features['last_order'] - customer_features['first_order']).dt.days
        customer_features['order_frequency'] = customer_features['total_orders'] / (customer_features['days_since_first_order'] + 1)
        customer_features['avg_days_between_orders'] = customer_features['days_since_first_order'] / (customer_features['total_orders'] - 1)
        
        # Merge customer features
        df = df.merge(customer_features, on='customer_id', how='left')
        
        # Product-level features
        product_features = df.groupby('product_id').agg({
            'net_total': ['sum', 'mean'],
            'quantity': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        
        product_features.columns = ['product_id', 'product_total_revenue', 'product_avg_revenue', 
                                  'product_total_quantity', 'product_order_count']
        
        # Merge product features
        df = df.merge(product_features, on='product_id', how='left')
        
        # Time-based features
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['order_year'] = df['order_date'].dt.year
        df['order_quarter'] = df['order_date'].dt.quarter
        df['order_month'] = df['order_date'].dt.month
        df['order_day_of_week'] = df['order_date'].dt.dayofweek
        df['order_hour'] = df['order_date'].dt.hour
        df['is_weekend'] = df['order_date'].dt.dayofweek.isin([5, 6])
        df['is_holiday'] = self._is_holiday(df['order_date'])
        
        # Customer-product interaction features
        df['customer_product_affinity'] = df['quantity'] * df['unit_price']
        df['price_sensitivity'] = df['unit_price'] / df['avg_order_value']
        
        return df
    
    def _is_holiday(self, dates: pd.Series) -> pd.Series:
        """Simple holiday detection"""
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25), # Christmas
        ]
        
        return dates.dt.month.isin([1, 7, 12]) & dates.dt.day.isin([1, 4, 25])
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with advanced techniques"""
        logger.info("Handling missing values...")
        
        # Use KNN imputation for numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 1:
            imputer = KNNImputer(n_neighbors=5)
            df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
        
        # Use mode imputation for categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            if df[column].isnull().any():
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column] = df[column].fillna(mode_value[0])
                else:
                    df[column] = df[column].fillna('Unknown')
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features using automated techniques"""
        logger.info("Engineering features...")
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        # Domain knowledge features
        df = self._add_domain_features(df)
        
        # Automated feature engineering
        df = self._add_automated_features(df)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        # Rolling statistics for time series
        if 'order_date' in df.columns:
            df_sorted = df.sort_values('order_date')
            df_sorted['revenue_rolling_mean'] = df_sorted['net_total'].rolling(window=7).mean()
            df_sorted['revenue_rolling_std'] = df_sorted['net_total'].rolling(window=7).std()
            df = df_sorted
        
        # Z-scores for outlier detection
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for column in numerical_columns:
            if column not in ['customer_id', 'product_id', 'order_id']:
                df[f'{column}_zscore'] = (df[column] - df[column].mean()) / df[column].std()
        
        return df
    
    def _add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-specific features"""
        # Customer lifetime value
        if 'total_spent' in df.columns and 'total_orders' in df.columns:
            df['customer_lifetime_value'] = df['total_spent'] * 12  # Annual CLV
        
        # Product profitability
        if 'unit_price' in df.columns and 'cost' in df.columns:
            df['profit_margin'] = (df['unit_price'] - df['cost']) / df['unit_price']
        
        # Order value categories
        if 'net_total' in df.columns:
            df['order_value_category'] = pd.cut(df['net_total'], 
                                              bins=[0, 50, 100, 200, 500, float('inf')], 
                                              labels=['Micro', 'Small', 'Medium', 'Large', 'XL'])
        
        return df
    
    def _add_automated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add automated features using ML techniques"""
        # Polynomial features for numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for column in numerical_columns[:5]:  # Limit to first 5 numerical columns
            if column not in ['customer_id', 'product_id', 'order_id']:
                df[f'{column}_squared'] = df[column] ** 2
                df[f'{column}_log'] = np.log1p(df[column])
        
        # Interaction features
        if 'quantity' in df.columns and 'unit_price' in df.columns:
            df['quantity_price_interaction'] = df['quantity'] * df['unit_price']
        
        return df
    
    def perform_customer_segmentation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced customer segmentation using multiple algorithms"""
        logger.info("Performing customer segmentation...")
        
        # Prepare customer features
        customer_features = df.groupby('customer_id').agg({
            'net_total': ['sum', 'mean', 'count'],
            'quantity': 'sum',
            'order_date': ['min', 'max'],
            'age': 'first',
            'customer_segment': 'first'
        }).reset_index()
        
        customer_features.columns = ['customer_id', 'total_spent', 'avg_order_value', 'total_orders', 
                                   'total_quantity', 'first_order', 'last_order', 'age', 'segment']
        
        # Calculate additional features
        customer_features['days_active'] = (customer_features['last_order'] - customer_features['first_order']).dt.days
        customer_features['order_frequency'] = customer_features['total_orders'] / (customer_features['days_active'] + 1)
        customer_features['avg_days_between_orders'] = customer_features['days_active'] / (customer_features['total_orders'] - 1)
        
        # Select features for clustering
        feature_columns = ['total_spent', 'avg_order_value', 'total_orders', 'total_quantity', 
                          'age', 'order_frequency']
        X = customer_features[feature_columns].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply multiple clustering algorithms
        clustering_results = {}
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        customer_features['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
        clustering_results['kmeans'] = {
            'model': kmeans,
            'silhouette_score': silhouette_score(X_scaled, customer_features['kmeans_cluster'])
        }
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        customer_features['dbscan_cluster'] = dbscan.fit_predict(X_scaled)
        if len(set(customer_features['dbscan_cluster'])) > 1:
            clustering_results['dbscan'] = {
                'model': dbscan,
                'silhouette_score': silhouette_score(X_scaled, customer_features['dbscan_cluster'])
            }
        
        # Gaussian Mixture Model
        gmm = GaussianMixture(n_components=4, random_state=42)
        customer_features['gmm_cluster'] = gmm.fit_predict(X_scaled)
        clustering_results['gmm'] = {
            'model': gmm,
            'silhouette_score': silhouette_score(X_scaled, customer_features['gmm_cluster'])
        }
        
        # Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=4)
        customer_features['hierarchical_cluster'] = hierarchical.fit_predict(X_scaled)
        clustering_results['hierarchical'] = {
            'model': hierarchical,
            'silhouette_score': silhouette_score(X_scaled, customer_features['hierarchical_cluster'])
        }
        
        # Analyze clusters
        cluster_analysis = {}
        for method, result in clustering_results.items():
            cluster_analysis[method] = self._analyze_clusters(customer_features, f'{method}_cluster', feature_columns)
        
        return {
            'customer_features': customer_features,
            'clustering_results': clustering_results,
            'cluster_analysis': cluster_analysis,
            'feature_columns': feature_columns
        }
    
    def _analyze_clusters(self, df: pd.DataFrame, cluster_column: str, feature_columns: List[str]) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        analysis = {}
        
        for cluster_id in df[cluster_column].unique():
            cluster_data = df[df[cluster_column] == cluster_id]
            analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'characteristics': cluster_data[feature_columns].mean().to_dict()
            }
        
        return analysis
    
    def predict_customer_lifetime_value(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict customer lifetime value using advanced ML models"""
        logger.info("Predicting customer lifetime value...")
        
        # Prepare features for CLV prediction
        customer_clv = df.groupby('customer_id').agg({
            'net_total': ['sum', 'mean', 'count'],
            'quantity': 'sum',
            'age': 'first',
            'customer_segment': 'first',
            'order_date': ['min', 'max']
        }).reset_index()
        
        customer_clv.columns = ['customer_id', 'total_spent', 'avg_order_value', 'total_orders',
                              'total_quantity', 'age', 'segment', 'first_order', 'last_order']
        
        # Calculate target variable (future CLV)
        customer_clv['days_active'] = (customer_clv['last_order'] - customer_clv['first_order']).dt.days
        customer_clv['monthly_spend'] = customer_clv['total_spent'] / (customer_clv['days_active'] / 30 + 1)
        customer_clv['predicted_clv'] = customer_clv['monthly_spend'] * 12  # Annual CLV
        
        # Prepare features
        le = LabelEncoder()
        customer_clv['segment_encoded'] = le.fit_transform(customer_clv['segment'])
        
        feature_columns = ['avg_order_value', 'total_orders', 'total_quantity', 'age', 
                          'segment_encoded']
        X = customer_clv[feature_columns]
        y = customer_clv['predicted_clv']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
            'CatBoost': cb.CatBoostRegressor(iterations=100, random_seed=42, verbose=False)
        }
        
        model_results = {}
        
        for name, model in models.items():
            # Train model
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Make predictions
            start_time = datetime.now()
            y_pred = model.predict(X_test)
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_columns, model.feature_importances_))
            else:
                feature_importance = {}
            
            model_results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'prediction_time': prediction_time,
                'feature_importance': feature_importance
            }
        
        # Find best model
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
        best_model = model_results[best_model_name]
        
        return {
            'customer_clv': customer_clv,
            'model_results': model_results,
            'best_model': best_model_name,
            'best_model_performance': best_model,
            'feature_columns': feature_columns
        }
    
    def build_product_recommendation_system(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build advanced product recommendation system"""
        logger.info("Building product recommendation system...")
        
        # Create customer-product matrix
        customer_product_matrix = df.groupby(['customer_id', 'product_id']).agg({
            'quantity': 'sum',
            'net_total': 'sum'
        }).reset_index()
        
        # Calculate affinity scores
        customer_product_matrix['affinity_score'] = (
            customer_product_matrix['quantity'] * 0.3 + 
            (customer_product_matrix['net_total'] / customer_product_matrix['net_total'].max()) * 0.7
        )
        
        # Collaborative filtering
        pivot_matrix = customer_product_matrix.pivot_table(
            index='customer_id', 
            columns='product_id', 
            values='affinity_score', 
            fill_value=0
        )
        
        # Calculate similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(pivot_matrix)
        similarity_df = pd.DataFrame(
            similarity_matrix, 
            index=pivot_matrix.index, 
            columns=pivot_matrix.index
        )
        
        # Generate recommendations
        recommendations = {}
        for customer_id in pivot_matrix.index[:10]:  # Sample of customers
            # Find similar customers
            similar_customers = similarity_df[customer_id].sort_values(ascending=False)[1:6]
            
            # Get products from similar customers
            recommended_products = []
            for similar_customer in similar_customers.index:
                customer_products = pivot_matrix.loc[similar_customer]
                top_products = customer_products.sort_values(ascending=False).head(3)
                recommended_products.extend(top_products.index.tolist())
            
            # Remove products customer already bought
            customer_products = set(pivot_matrix.loc[customer_id][pivot_matrix.loc[customer_id] > 0].index)
            recommended_products = [p for p in recommended_products if p not in customer_products]
            
            # Get top recommendations
            recommendations[customer_id] = list(set(recommended_products))[:5]
        
        return {
            'customer_product_matrix': customer_product_matrix,
            'similarity_matrix': similarity_df,
            'recommendations': recommendations
        }
    
    def perform_anomaly_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform anomaly detection using multiple algorithms"""
        logger.info("Performing anomaly detection...")
        
        # Prepare features for anomaly detection
        feature_columns = ['net_total', 'quantity', 'unit_price']
        X = df[feature_columns].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply multiple anomaly detection algorithms
        anomaly_results = {}
        
        # Isolation Forest
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        df['isolation_forest_anomaly'] = isolation_forest.fit_predict(X_scaled)
        anomaly_results['isolation_forest'] = {
            'model': isolation_forest,
            'anomaly_count': (df['isolation_forest_anomaly'] == -1).sum()
        }
        
        # One-Class SVM
        from sklearn.svm import OneClassSVM
        one_class_svm = OneClassSVM(nu=0.1)
        df['one_class_svm_anomaly'] = one_class_svm.fit_predict(X_scaled)
        anomaly_results['one_class_svm'] = {
            'model': one_class_svm,
            'anomaly_count': (df['one_class_svm_anomaly'] == -1).sum()
        }
        
        # Local Outlier Factor
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        df['lof_anomaly'] = lof.fit_predict(X_scaled)
        anomaly_results['lof'] = {
            'model': lof,
            'anomaly_count': (df['lof_anomaly'] == -1).sum()
        }
        
        # Statistical anomaly detection
        df['zscore_anomaly'] = False
        for column in feature_columns:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df['zscore_anomaly'] |= (z_scores > 3)
        
        anomaly_results['statistical'] = {
            'anomaly_count': df['zscore_anomaly'].sum()
        }
        
        return {
            'anomaly_results': anomaly_results,
            'anomaly_data': df
        }
    
    def generate_model_explanations(self, model, X_test: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Generate model explanations using multiple techniques"""
        logger.info("Generating model explanations...")
        
        explanations = {}
        
        # SHAP explanations
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            explanations['shap'] = {
                'shap_values': shap_values,
                'feature_importance': np.abs(shap_values).mean(0)
            }
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
        
        # LIME explanations
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_test.values, 
                feature_names=feature_names,
                mode='regression'
            )
            explanations['lime'] = {
                'explainer': lime_explainer,
                'sample_explanations': []
            }
            
            # Generate explanations for sample instances
            for i in range(min(5, len(X_test))):
                explanation = lime_explainer.explain_instance(
                    X_test.iloc[i].values, 
                    model.predict, 
                    num_features=len(feature_names)
                )
                explanations['lime']['sample_explanations'].append(explanation)
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
        
        # Permutation importance
        try:
            perm_importance = PermutationImportance(model, random_state=42)
            perm_importance.fit(X_test, model.predict(X_test))
            explanations['permutation'] = {
                'feature_importance': perm_importance.feature_importances_,
                'feature_names': feature_names
            }
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
        
        return explanations
    
    def optimize_hyperparameters(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                                param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        logger.info("Optimizing hyperparameters...")
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config['low'], 
                        param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_config['low'], 
                        param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config['choices']
                    )
            
            # Create model with sampled parameters
            model_instance = model.__class__(**params)
            
            # Cross-validation score
            scores = cross_val_score(model_instance, X_train, y_train, cv=5, scoring='r2')
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['hyperparameter_optimization']['n_trials'])
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def run_complete_ml_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete ML analysis pipeline"""
        logger.info("Running complete ML analysis...")
        
        results = {}
        
        # Customer segmentation
        results['segmentation'] = self.perform_customer_segmentation(df)
        
        # CLV prediction
        results['clv_prediction'] = self.predict_customer_lifetime_value(df)
        
        # Product recommendations
        results['recommendations'] = self.build_product_recommendation_system(df)
        
        # Anomaly detection
        results['anomaly_detection'] = self.perform_anomaly_detection(df)
        
        # Generate comprehensive report
        results['summary'] = self._generate_ml_summary(results)
        
        return results
    
    def _generate_ml_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive ML analysis summary"""
        summary = {
            'total_customers': len(results['segmentation']['customer_features']),
            'total_products': len(results['recommendations']['customer_product_matrix']['product_id'].unique()),
            'segmentation_methods': len(results['segmentation']['clustering_results']),
            'clv_models': len(results['clv_prediction']['model_results']),
            'best_clv_model': results['clv_prediction']['best_model'],
            'best_clv_r2': results['clv_prediction']['best_model_performance']['r2'],
            'anomaly_detection_methods': len(results['anomaly_detection']['anomaly_results']),
            'recommendation_system': 'collaborative_filtering'
        }
        
        return summary

def main():
    """Demonstrate the advanced ML analytics engine"""
    ml_engine = AdvancedMLAnalytics()
    
    # Load data
    data = ml_engine.load_data()
    
    # Run complete ML analysis
    results = ml_engine.run_complete_ml_analysis(data)
    
    print("=== ADVANCED ML ANALYTICS DEMONSTRATION ===")
    print(f"Total customers analyzed: {results['summary']['total_customers']}")
    print(f"Total products: {results['summary']['total_products']}")
    print(f"Segmentation methods: {results['summary']['segmentation_methods']}")
    print(f"CLV models tested: {results['summary']['clv_models']}")
    print(f"Best CLV model: {results['summary']['best_clv_model']}")
    print(f"Best CLV R²: {results['summary']['best_clv_r2']:.3f}")
    print(f"Anomaly detection methods: {results['summary']['anomaly_detection_methods']}")
    
    # Save results
    with open('advanced_ml_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Advanced ML analysis results saved to 'advanced_ml_results.json'")

if __name__ == "__main__":
    main()
