"""
Main Application - Retail Data Modeling Demonstration
Comprehensive showcase of all required skills for the Bosch internship
"""

import sys
import os
from pathlib import Path
import logging

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from etl_pipeline import RetailETLPipeline
from data_analysis import RetailDataAnalyzer
from data_warehouse import DataWarehouse
from dashboard import RetailDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retail_analytics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RetailAnalyticsApplication:
    """Main application class demonstrating all required skills"""
    
    def __init__(self):
        self.etl_pipeline = RetailETLPipeline()
        self.data_analyzer = RetailDataAnalyzer()
        self.data_warehouse = DataWarehouse()
        self.dashboard = RetailDashboard()
        
    def run_complete_demonstration(self):
        """Run the complete demonstration of all skills"""
        print("=" * 80)
        print("🏪 RETAIL DATA MODELING DEMONSTRATION")
        print("=" * 80)
        print("This application demonstrates all skills required for the Bosch internship:")
        print("✅ Semantic data modeling with ontologies (TTL files)")
        print("✅ ETL pipeline development and data processing")
        print("✅ Multiple data modeling schemas (3NF, Star, Snowflake)")
        print("✅ Python data analysis and machine learning")
        print("✅ Data warehousing principles and dimensional modeling")
        print("✅ Interactive dashboards and visualizations")
        print("=" * 80)
        
        try:
            # Step 1: Run ETL Pipeline
            print("\n🔄 STEP 1: ETL PIPELINE")
            print("-" * 40)
            transformed_data = self.etl_pipeline.run_pipeline()
            print("✅ ETL pipeline completed successfully")
            
            # Step 2: Data Analysis and ML
            print("\n📊 STEP 2: DATA ANALYSIS & MACHINE LEARNING")
            print("-" * 40)
            analysis_results = self.data_analyzer.run_complete_analysis()
            print("✅ Data analysis and ML completed successfully")
            
            # Step 3: Data Warehouse Setup
            print("\n🏗️ STEP 3: DATA WAREHOUSE SETUP")
            print("-" * 40)
            # Use the analytical dataset for warehouse setup
            analytical_data = self.data_analyzer.data['analytical_dataset']
            self.data_warehouse.run_data_warehouse_setup(analytical_data)
            print("✅ Data warehouse setup completed successfully")
            
            # Step 4: Dashboard Launch Instructions
            print("\n📈 STEP 4: INTERACTIVE DASHBOARD")
            print("-" * 40)
            print("To launch the interactive dashboard, run:")
            print("streamlit run dashboard.py")
            print("✅ Dashboard ready for launch")
            
            # Summary
            print("\n🎉 DEMONSTRATION COMPLETE!")
            print("=" * 80)
            print("All components have been successfully implemented:")
            print("📁 Files created:")
            print("   • ontology/retail_ontology.ttl - Semantic data model")
            print("   • data_models/ - SQL schemas (3NF, Star, Snowflake)")
            print("   • retail_data.db - Source database")
            print("   • data_warehouse.db - Data warehouse")
            print("   • retail_analysis_dashboard.png - Visualizations")
            print("   • retail_analytics.log - Application logs")
            print("\n🚀 Next steps:")
            print("   1. Run 'streamlit run dashboard.py' for interactive dashboard")
            print("   2. Explore the generated visualizations and analysis")
            print("   3. Review the semantic ontology and data models")
            print("   4. Examine the ETL pipeline and data warehouse implementation")
            
        except Exception as e:
            logger.error(f"Demonstration failed: {e}")
            print(f"\n❌ Error: {e}")
            print("Check the log file for detailed error information.")
            raise
    
    def show_skill_demonstration(self):
        """Show what skills are demonstrated in this application"""
        print("\n" + "=" * 80)
        print("🎯 SKILLS DEMONSTRATED")
        print("=" * 80)
        
        skills = {
            "Semantic Data Modeling": [
                "• TTL ontology file with retail domain concepts",
                "• RDF/OWL semantic relationships",
                "• Business rules and constraints",
                "• Entity-relationship modeling"
            ],
            "ETL Pipeline Development": [
                "• Data extraction from multiple sources",
                "• Data transformation and cleansing",
                "• Data loading into target systems",
                "• Error handling and logging"
            ],
            "Data Modeling Schemas": [
                "• 3rd Normal Form (3NF) relational schema",
                "• Star schema for data warehousing",
                "• Snowflake schema for advanced analytics",
                "• Dimensional modeling principles"
            ],
            "Python Data Processing": [
                "• Pandas for data manipulation",
                "• NumPy for numerical computing",
                "• SQLite for data storage",
                "• Data quality and validation"
            ],
            "Machine Learning": [
                "• Customer segmentation with K-means",
                "• Customer lifetime value prediction",
                "• Product recommendation system",
                "• Feature engineering and selection"
            ],
            "Data Visualization": [
                "• Matplotlib and Seaborn plots",
                "• Interactive Plotly visualizations",
                "• Streamlit dashboard",
                "• Business intelligence reporting"
            ],
            "Data Warehousing": [
                "• Dimension and fact table design",
                "• Slowly changing dimensions",
                "• Data aggregation strategies",
                "• Performance optimization"
            ]
        }
        
        for skill, details in skills.items():
            print(f"\n📋 {skill}:")
            for detail in details:
                print(f"   {detail}")
        
        print("\n" + "=" * 80)

def main():
    """Main function"""
    app = RetailAnalyticsApplication()
    
    # Show skills demonstration
    app.show_skill_demonstration()
    
    # Run complete demonstration
    print("\n" + "=" * 80)
    print("🚀 STARTING COMPLETE DEMONSTRATION")
    print("=" * 80)
    
    try:
        app.run_complete_demonstration()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demonstration failed: {e}")
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
