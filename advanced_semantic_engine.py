"""
Advanced Semantic Data Engine
Enterprise-grade semantic data modeling with RDF/OWL reasoning, SPARQL queries,
and complex business rule inference
"""

import rdflib
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
import owlrl
from owlrl import DeductiveClosure, OWLRL_Semantics
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import uuid
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessRuleType(Enum):
    """Types of business rules for semantic reasoning"""
    CONSTRAINT = "constraint"
    DERIVATION = "derivation"
    AGGREGATION = "aggregation"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"

@dataclass
class SemanticRule:
    """Represents a business rule in the semantic model"""
    rule_id: str
    rule_type: BusinessRuleType
    condition: str
    consequence: str
    confidence: float
    temporal_validity: Optional[Tuple[datetime, datetime]] = None
    context: Optional[Dict[str, Any]] = None

class AdvancedSemanticEngine:
    """Enterprise semantic data engine with advanced reasoning capabilities"""
    
    def __init__(self):
        self.graph = Graph()
        self.namespaces = {}
        self.business_rules = []
        self.reasoning_engine = None
        self.inferred_facts = []
        self.semantic_metrics = {}
        
        # Initialize namespaces
        self._initialize_namespaces()
        
        # Initialize reasoning engine
        self._initialize_reasoning_engine()
    
    def _initialize_namespaces(self):
        """Initialize RDF/OWL namespaces for the retail domain"""
        self.namespaces = {
            'retail': Namespace('http://example.org/retail#'),
            'retail_ontology': Namespace('http://example.org/retail/ontology#'),
            'retail_data': Namespace('http://example.org/retail/data#'),
            'retail_rules': Namespace('http://example.org/retail/rules#'),
            'retail_metrics': Namespace('http://example.org/retail/metrics#'),
            'retail_analytics': Namespace('http://example.org/retail/analytics#'),
            'time': Namespace('http://www.w3.org/2006/time#'),
            'foaf': Namespace('http://xmlns.com/foaf/0.1/'),
            'vcard': Namespace('http://www.w3.org/2006/vcard/ns#'),
            'schema': Namespace('http://schema.org/'),
            'qb': Namespace('http://purl.org/linked-data/cube#'),
            'skos': Namespace('http://www.w3.org/2004/02/skos/core#')
        }
        
        # Bind namespaces to graph
        for prefix, namespace in self.namespaces.items():
            self.graph.bind(prefix, namespace)
    
    def _initialize_reasoning_engine(self):
        """Initialize OWL reasoning engine with custom rules"""
        self.reasoning_engine = DeductiveClosure(OWLRL_Semantics)
        
        # Add custom business rules
        self._add_custom_business_rules()
    
    def _add_custom_business_rules(self):
        """Add complex business rules for retail domain"""
        retail = self.namespaces['retail']
        rules = self.namespaces['retail_rules']
        
        # Rule 1: High-value customer classification
        high_value_rule = """
        PREFIX retail: <http://example.org/retail#>
        PREFIX retail_rules: <http://example.org/retail/rules#>
        
        CONSTRUCT {
            ?customer retail:isHighValueCustomer true .
            ?customer retail:customerTier "Premium" .
        }
        WHERE {
            ?customer retail:totalLifetimeValue ?clv .
            ?customer retail:orderFrequency ?freq .
            FILTER(?clv > 10000 && ?freq > 0.5)
        }
        """
        
        # Rule 2: Product affinity scoring
        affinity_rule = """
        PREFIX retail: <http://example.org/retail#>
        PREFIX retail_rules: <http://example.org/retail/rules#>
        
        CONSTRUCT {
            ?customer retail:hasProductAffinity ?product .
            ?affinity retail:affinityScore ?score .
        }
        WHERE {
            ?customer retail:hasPurchased ?product .
            ?product retail:category ?category .
            ?customer retail:preferredCategory ?category .
            BIND(?score AS ?score)
        }
        """
        
        # Rule 3: Churn prediction
        churn_rule = """
        PREFIX retail: <http://example.org/retail#>
        PREFIX retail_rules: <http://example.org/retail/rules#>
        
        CONSTRUCT {
            ?customer retail:churnRisk "High" .
            ?customer retail:requiresIntervention true .
        }
        WHERE {
            ?customer retail:lastOrderDate ?lastOrder .
            ?customer retail:averageOrderInterval ?interval .
            FILTER(?lastOrder < NOW() - ?interval * 3)
        }
        """
        
        # Add rules to graph
        self.graph.add((rules.HighValueCustomerRule, RDF.type, OWL.Class))
        self.graph.add((rules.ProductAffinityRule, RDF.type, OWL.Class))
        self.graph.add((rules.ChurnPredictionRule, RDF.type, OWL.Class))
    
    def create_advanced_ontology(self):
        """Create a comprehensive retail ontology with complex relationships"""
        retail = self.namespaces['retail']
        retail_ontology = self.namespaces['retail_ontology']
        time = self.namespaces['time']
        
        # Core Entity Classes
        self._define_core_entities(retail, retail_ontology)
        
        # Complex Relationship Classes
        self._define_relationship_classes(retail, retail_ontology)
        
        # Temporal and Spatial Classes
        self._define_temporal_spatial_classes(retail, retail_ontology, time)
        
        # Analytics and Metrics Classes
        self._define_analytics_classes(retail, retail_ontology)
        
        # Business Process Classes
        self._define_business_process_classes(retail, retail_ontology)
        
        # Data Quality and Governance Classes
        self._define_data_governance_classes(retail, retail_ontology)
    
    def _define_core_entities(self, retail, retail_ontology):
        """Define core business entities with complex properties"""
        
        # Customer Entity with Advanced Properties
        self.graph.add((retail.Customer, RDF.type, OWL.Class))
        self.graph.add((retail.Customer, RDFS.subClassOf, retail.BusinessEntity))
        
        # Customer Demographics
        self.graph.add((retail.customerId, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.customerId, RDFS.domain, retail.Customer))
        self.graph.add((retail.customerId, RDFS.range, XSD.string))
        
        self.graph.add((retail.customerSegment, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.customerSegment, RDFS.domain, retail.Customer))
        self.graph.add((retail.customerSegment, RDFS.range, XSD.string))
        
        # Customer Behavioral Properties
        self.graph.add((retail.customerLifetimeValue, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.customerLifetimeValue, RDFS.domain, retail.Customer))
        self.graph.add((retail.customerLifetimeValue, RDFS.range, XSD.decimal))
        
        self.graph.add((retail.orderFrequency, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.orderFrequency, RDFS.domain, retail.Customer))
        self.graph.add((retail.orderFrequency, RDFS.range, XSD.float))
        
        self.graph.add((retail.churnRisk, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.churnRisk, RDFS.domain, retail.Customer))
        self.graph.add((retail.churnRisk, RDFS.range, XSD.string))
        
        # Product Entity with Complex Taxonomy
        self.graph.add((retail.Product, RDF.type, OWL.Class))
        self.graph.add((retail.Product, RDFS.subClassOf, retail.BusinessEntity))
        
        # Product Hierarchy
        self.graph.add((retail.ProductCategory, RDF.type, OWL.Class))
        self.graph.add((retail.ProductSubcategory, RDF.type, OWL.Class))
        self.graph.add((retail.ProductSubcategory, RDFS.subClassOf, retail.ProductCategory))
        
        # Product Properties
        self.graph.add((retail.productId, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.productId, RDFS.domain, retail.Product))
        self.graph.add((retail.productId, RDFS.range, XSD.string))
        
        self.graph.add((retail.productPrice, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.productPrice, RDFS.domain, retail.Product))
        self.graph.add((retail.productPrice, RDFS.range, XSD.decimal))
        
        # Store Entity with Geographic Properties
        self.graph.add((retail.Store, RDF.type, OWL.Class))
        self.graph.add((retail.Store, RDFS.subClassOf, retail.BusinessEntity))
        
        # Geographic Properties
        self.graph.add((retail.storeLocation, RDF.type, OWL.ObjectProperty))
        self.graph.add((retail.storeLocation, RDFS.domain, retail.Store))
        self.graph.add((retail.storeLocation, RDFS.range, retail.GeographicLocation))
        
        # Order Entity with Complex State Machine
        self.graph.add((retail.Order, RDF.type, OWL.Class))
        self.graph.add((retail.Order, RDFS.subClassOf, retail.BusinessEntity))
        
        # Order States
        self.graph.add((retail.OrderState, RDF.type, OWL.Class))
        self.graph.add((retail.PendingOrder, RDF.type, OWL.Class))
        self.graph.add((retail.PendingOrder, RDFS.subClassOf, retail.OrderState))
        self.graph.add((retail.ProcessingOrder, RDF.type, OWL.Class))
        self.graph.add((retail.ProcessingOrder, RDFS.subClassOf, retail.OrderState))
        self.graph.add((retail.CompletedOrder, RDF.type, OWL.Class))
        self.graph.add((retail.CompletedOrder, RDFS.subClassOf, retail.OrderState))
        self.graph.add((retail.CancelledOrder, RDF.type, OWL.Class))
        self.graph.add((retail.CancelledOrder, RDFS.subClassOf, retail.OrderState))
    
    def _define_relationship_classes(self, retail, retail_ontology):
        """Define complex relationship classes"""
        
        # Customer-Product Relationships
        self.graph.add((retail.CustomerProductAffinity, RDF.type, OWL.Class))
        self.graph.add((retail.affinityScore, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.affinityScore, RDFS.domain, retail.CustomerProductAffinity))
        self.graph.add((retail.affinityScore, RDFS.range, XSD.float))
        
        # Customer-Store Relationships
        self.graph.add((retail.CustomerStorePreference, RDF.type, OWL.Class))
        self.graph.add((retail.preferenceScore, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.preferenceScore, RDFS.domain, retail.CustomerStorePreference))
        self.graph.add((retail.preferenceScore, RDFS.range, XSD.float))
        
        # Product-Product Relationships
        self.graph.add((retail.ProductComplement, RDF.type, OWL.Class))
        self.graph.add((retail.ProductSubstitute, RDF.type, OWL.Class))
        self.graph.add((retail.ProductBundle, RDF.type, OWL.Class))
        
        # Temporal Relationships
        self.graph.add((retail.TemporalRelationship, RDF.type, OWL.Class))
        self.graph.add((retail.precedes, RDF.type, OWL.ObjectProperty))
        self.graph.add((retail.precedes, RDFS.domain, retail.TemporalRelationship))
        self.graph.add((retail.precedes, RDFS.range, retail.TemporalRelationship))
        
        self.graph.add((retail.follows, RDF.type, OWL.ObjectProperty))
        self.graph.add((retail.follows, RDFS.domain, retail.TemporalRelationship))
        self.graph.add((retail.follows, RDFS.range, retail.TemporalRelationship))
    
    def _define_temporal_spatial_classes(self, retail, retail_ontology, time):
        """Define temporal and spatial classes"""
        
        # Temporal Classes
        self.graph.add((retail.TemporalEntity, RDF.type, OWL.Class))
        self.graph.add((retail.TimePoint, RDF.type, OWL.Class))
        self.graph.add((retail.TimePoint, RDFS.subClassOf, retail.TemporalEntity))
        self.graph.add((retail.TimeInterval, RDF.type, OWL.Class))
        self.graph.add((retail.TimeInterval, RDFS.subClassOf, retail.TemporalEntity))
        
        # Spatial Classes
        self.graph.add((retail.SpatialEntity, RDF.type, OWL.Class))
        self.graph.add((retail.GeographicLocation, RDF.type, OWL.Class))
        self.graph.add((retail.GeographicLocation, RDFS.subClassOf, retail.SpatialEntity))
        self.graph.add((retail.Region, RDF.type, OWL.Class))
        self.graph.add((retail.Region, RDFS.subClassOf, retail.GeographicLocation))
        self.graph.add((retail.City, RDF.type, OWL.Class))
        self.graph.add((retail.City, RDFS.subClassOf, retail.GeographicLocation))
        
        # Geographic Properties
        self.graph.add((retail.latitude, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.latitude, RDFS.domain, retail.GeographicLocation))
        self.graph.add((retail.latitude, RDFS.range, XSD.decimal))
        
        self.graph.add((retail.longitude, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.longitude, RDFS.domain, retail.GeographicLocation))
        self.graph.add((retail.longitude, RDFS.range, XSD.decimal))
    
    def _define_analytics_classes(self, retail, retail_ontology):
        """Define analytics and metrics classes"""
        
        # Metrics Classes
        self.graph.add((retail.Metric, RDF.type, OWL.Class))
        self.graph.add((retail.BusinessMetric, RDF.type, OWL.Class))
        self.graph.add((retail.BusinessMetric, RDFS.subClassOf, retail.Metric))
        self.graph.add((retail.TechnicalMetric, RDF.type, OWL.Class))
        self.graph.add((retail.TechnicalMetric, RDFS.subClassOf, retail.Metric))
        
        # Specific Metrics
        self.graph.add((retail.RevenueMetric, RDF.type, OWL.Class))
        self.graph.add((retail.RevenueMetric, RDFS.subClassOf, retail.BusinessMetric))
        
        self.graph.add((retail.CustomerLifetimeValueMetric, RDF.type, OWL.Class))
        self.graph.add((retail.CustomerLifetimeValueMetric, RDFS.subClassOf, retail.BusinessMetric))
        
        self.graph.add((retail.ChurnRateMetric, RDF.type, OWL.Class))
        self.graph.add((retail.ChurnRateMetric, RDFS.subClassOf, retail.BusinessMetric))
        
        # Analytics Classes
        self.graph.add((retail.AnalyticsModel, RDF.type, OWL.Class))
        self.graph.add((retail.PredictiveModel, RDF.type, OWL.Class))
        self.graph.add((retail.PredictiveModel, RDFS.subClassOf, retail.AnalyticsModel))
        
        self.graph.add((retail.ClassificationModel, RDF.type, OWL.Class))
        self.graph.add((retail.ClassificationModel, RDFS.subClassOf, retail.PredictiveModel))
        
        self.graph.add((retail.RegressionModel, RDF.type, OWL.Class))
        self.graph.add((retail.RegressionModel, RDFS.subClassOf, retail.PredictiveModel))
        
        self.graph.add((retail.ClusteringModel, RDF.type, OWL.Class))
        self.graph.add((retail.ClusteringModel, RDFS.subClassOf, retail.AnalyticsModel))
    
    def _define_business_process_classes(self, retail, retail_ontology):
        """Define business process classes"""
        
        # Process Classes
        self.graph.add((retail.BusinessProcess, RDF.type, OWL.Class))
        self.graph.add((retail.OrderProcess, RDF.type, OWL.Class))
        self.graph.add((retail.OrderProcess, RDFS.subClassOf, retail.BusinessProcess))
        
        self.graph.add((retail.CustomerOnboardingProcess, RDF.type, OWL.Class))
        self.graph.add((retail.CustomerOnboardingProcess, RDFS.subClassOf, retail.BusinessProcess))
        
        self.graph.add((retail.ProductRecommendationProcess, RDF.type, OWL.Class))
        self.graph.add((retail.ProductRecommendationProcess, RDFS.subClassOf, retail.BusinessProcess))
        
        # Process Properties
        self.graph.add((retail.processStep, RDF.type, OWL.ObjectProperty))
        self.graph.add((retail.processStep, RDFS.domain, retail.BusinessProcess))
        self.graph.add((retail.processStep, RDFS.range, retail.ProcessStep))
        
        self.graph.add((retail.processDuration, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.processDuration, RDFS.domain, retail.BusinessProcess))
        self.graph.add((retail.processDuration, RDFS.range, XSD.duration))
    
    def _define_data_governance_classes(self, retail, retail_ontology):
        """Define data governance and quality classes"""
        
        # Data Quality Classes
        self.graph.add((retail.DataQualityMetric, RDF.type, OWL.Class))
        self.graph.add((retail.CompletenessMetric, RDF.type, OWL.Class))
        self.graph.add((retail.CompletenessMetric, RDFS.subClassOf, retail.DataQualityMetric))
        
        self.graph.add((retail.AccuracyMetric, RDF.type, OWL.Class))
        self.graph.add((retail.AccuracyMetric, RDFS.subClassOf, retail.DataQualityMetric))
        
        self.graph.add((retail.ConsistencyMetric, RDF.type, OWL.Class))
        self.graph.add((retail.ConsistencyMetric, RDFS.subClassOf, retail.DataQualityMetric))
        
        # Data Lineage Classes
        self.graph.add((retail.DataLineage, RDF.type, OWL.Class))
        self.graph.add((retail.DataSource, RDF.type, OWL.Class))
        self.graph.add((retail.DataTransformation, RDF.type, OWL.Class))
        self.graph.add((retail.DataDestination, RDF.type, OWL.Class))
        
        # Data Governance Properties
        self.graph.add((retail.dataOwner, RDF.type, OWL.ObjectProperty))
        self.graph.add((retail.dataOwner, RDFS.domain, retail.DataSource))
        self.graph.add((retail.dataOwner, RDFS.range, retail.DataSteward))
        
        self.graph.add((retail.dataClassification, RDF.type, OWL.DatatypeProperty))
        self.graph.add((retail.dataClassification, RDFS.domain, retail.DataSource))
        self.graph.add((retail.dataClassification, RDFS.range, XSD.string))
    
    def add_business_rule(self, rule: SemanticRule):
        """Add a complex business rule to the semantic model"""
        self.business_rules.append(rule)
        
        # Convert rule to RDF/OWL
        self._convert_rule_to_rdf(rule)
        
        logger.info(f"Added business rule: {rule.rule_id}")
    
    def _convert_rule_to_rdf(self, rule: SemanticRule):
        """Convert business rule to RDF/OWL representation"""
        rules = self.namespaces['retail_rules']
        retail = self.namespaces['retail']
        
        # Create rule URI
        rule_uri = rules[f"rule_{rule.rule_id}"]
        
        # Add rule as OWL class
        self.graph.add((rule_uri, RDF.type, OWL.Class))
        self.graph.add((rule_uri, RDF.type, rules.BusinessRule))
        
        # Add rule properties
        self.graph.add((rule_uri, rules.ruleType, Literal(rule.rule_type.value)))
        self.graph.add((rule_uri, rules.condition, Literal(rule.condition)))
        self.graph.add((rule_uri, rules.consequence, Literal(rule.consequence)))
        self.graph.add((rule_uri, rules.confidence, Literal(rule.confidence, datatype=XSD.float)))
        
        if rule.temporal_validity:
            self.graph.add((rule_uri, rules.validFrom, Literal(rule.temporal_validity[0], datatype=XSD.dateTime)))
            self.graph.add((rule_uri, rules.validTo, Literal(rule.temporal_validity[1], datatype=XSD.dateTime)))
    
    def perform_semantic_reasoning(self):
        """Perform advanced semantic reasoning with business rules"""
        logger.info("Starting semantic reasoning...")
        
        # Apply OWL reasoning
        self.reasoning_engine.expand(self.graph)
        
        # Apply custom business rules
        self._apply_custom_business_rules()
        
        # Generate inferred facts
        self._generate_inferred_facts()
        
        # Calculate semantic metrics
        self._calculate_semantic_metrics()
        
        logger.info("Semantic reasoning completed")
    
    def _apply_custom_business_rules(self):
        """Apply custom business rules to the graph"""
        for rule in self.business_rules:
            if rule.rule_type == BusinessRuleType.DERIVATION:
                self._apply_derivation_rule(rule)
            elif rule.rule_type == BusinessRuleType.CONSTRAINT:
                self._apply_constraint_rule(rule)
            elif rule.rule_type == BusinessRuleType.AGGREGATION:
                self._apply_aggregation_rule(rule)
    
    def _apply_derivation_rule(self, rule: SemanticRule):
        """Apply derivation rule to infer new facts"""
        # This would contain complex SPARQL queries to derive new facts
        # based on existing data and business rules
        pass
    
    def _apply_constraint_rule(self, rule: SemanticRule):
        """Apply constraint rule to validate data consistency"""
        # This would contain SPARQL queries to check data consistency
        # and identify violations
        pass
    
    def _apply_aggregation_rule(self, rule: SemanticRule):
        """Apply aggregation rule to compute derived metrics"""
        # This would contain SPARQL queries to compute aggregated metrics
        # from detailed data
        pass
    
    def _generate_inferred_facts(self):
        """Generate inferred facts from reasoning"""
        # Query for inferred facts
        query = """
        PREFIX retail: <http://example.org/retail#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?subject ?predicate ?object WHERE {
            ?subject ?predicate ?object .
            FILTER(?predicate != rdf:type)
        }
        """
        
        results = self.graph.query(query)
        for row in results:
            self.inferred_facts.append({
                'subject': str(row.subject),
                'predicate': str(row.predicate),
                'object': str(row.object)
            })
    
    def _calculate_semantic_metrics(self):
        """Calculate semantic metrics for the knowledge graph"""
        # Graph complexity metrics
        total_triples = len(self.graph)
        unique_subjects = len(set(str(s) for s, p, o in self.graph))
        unique_predicates = len(set(str(p) for s, p, o in self.graph))
        unique_objects = len(set(str(o) for s, p, o in self.graph))
        
        self.semantic_metrics = {
            'total_triples': total_triples,
            'unique_subjects': unique_subjects,
            'unique_predicates': unique_predicates,
            'unique_objects': unique_objects,
            'graph_density': total_triples / (unique_subjects * unique_predicates) if unique_subjects * unique_predicates > 0 else 0,
            'average_degree': total_triples / unique_subjects if unique_subjects > 0 else 0,
            'inferred_facts_count': len(self.inferred_facts),
            'business_rules_count': len(self.business_rules)
        }
    
    def execute_sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute SPARQL query against the semantic graph"""
        results = []
        query_results = self.graph.query(query)
        
        for row in query_results:
            result_dict = {}
            for var_name, var_value in row.asdict().items():
                result_dict[var_name] = str(var_value)
            results.append(result_dict)
        
        return results
    
    def generate_semantic_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive semantic analytics report"""
        report = {
            'semantic_metrics': self.semantic_metrics,
            'business_rules': [
                {
                    'rule_id': rule.rule_id,
                    'rule_type': rule.rule_type.value,
                    'condition': rule.condition,
                    'consequence': rule.consequence,
                    'confidence': rule.confidence
                }
                for rule in self.business_rules
            ],
            'inferred_facts': self.inferred_facts[:100],  # Limit for readability
            'graph_statistics': self._analyze_graph_structure(),
            'semantic_relationships': self._analyze_semantic_relationships()
        }
        
        return report
    
    def _analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze the structure of the semantic graph"""
        # Convert to NetworkX for analysis
        nx_graph = nx.Graph()
        
        for s, p, o in self.graph:
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                nx_graph.add_edge(str(s), str(o))
        
        return {
            'nodes': nx_graph.number_of_nodes(),
            'edges': nx_graph.number_of_edges(),
            'connected_components': nx.number_connected_components(nx_graph),
            'average_clustering': nx.average_clustering(nx_graph),
            'diameter': nx.diameter(nx_graph) if nx.is_connected(nx_graph) else None
        }
    
    def _analyze_semantic_relationships(self) -> Dict[str, Any]:
        """Analyze semantic relationships in the graph"""
        relationship_counts = {}
        
        for s, p, o in self.graph:
            predicate = str(p)
            if predicate not in relationship_counts:
                relationship_counts[predicate] = 0
            relationship_counts[predicate] += 1
        
        return {
            'total_relationships': len(relationship_counts),
            'most_common_relationships': sorted(
                relationship_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    
    def export_ontology(self, format: str = "turtle") -> str:
        """Export the ontology in specified format"""
        if format.lower() == "turtle":
            return self.graph.serialize(format="turtle")
        elif format.lower() == "rdf":
            return self.graph.serialize(format="xml")
        elif format.lower() == "json-ld":
            return self.graph.serialize(format="json-ld")
        else:
            raise ValueError(f"Unsupported format: {format}")

def main():
    """Demonstrate the advanced semantic engine"""
    engine = AdvancedSemanticEngine()
    
    # Create comprehensive ontology
    engine.create_advanced_ontology()
    
    # Add complex business rules
    high_value_rule = SemanticRule(
        rule_id="high_value_customer_001",
        rule_type=BusinessRuleType.DERIVATION,
        condition="Customer has lifetime value > $10,000 AND order frequency > 0.5",
        consequence="Customer is classified as high-value and gets premium treatment",
        confidence=0.95,
        temporal_validity=(datetime(2024, 1, 1), datetime(2024, 12, 31))
    )
    
    churn_prediction_rule = SemanticRule(
        rule_id="churn_prediction_001",
        rule_type=BusinessRuleType.DERIVATION,
        condition="Customer last order > 3 * average order interval",
        consequence="Customer has high churn risk and requires intervention",
        confidence=0.88
    )
    
    engine.add_business_rule(high_value_rule)
    engine.add_business_rule(churn_prediction_rule)
    
    # Perform semantic reasoning
    engine.perform_semantic_reasoning()
    
    # Generate analytics report
    report = engine.generate_semantic_analytics_report()
    
    print("=== ADVANCED SEMANTIC ENGINE DEMONSTRATION ===")
    print(f"Total triples: {report['semantic_metrics']['total_triples']}")
    print(f"Business rules: {report['semantic_metrics']['business_rules_count']}")
    print(f"Inferred facts: {report['semantic_metrics']['inferred_facts_count']}")
    
    # Export ontology
    ontology_turtle = engine.export_ontology("turtle")
    with open("advanced_retail_ontology.ttl", "w") as f:
        f.write(ontology_turtle)
    
    print("Advanced ontology exported to 'advanced_retail_ontology.ttl'")

if __name__ == "__main__":
    main()
