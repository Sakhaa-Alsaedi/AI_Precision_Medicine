#!/usr/bin/env python3
"""
Generate synthetic datasets for AI-powered precision medicine tutorial
Based on Alsaedi et al. (2024) methodology for genetic risk factor optimization
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_genetic_risk_factors():
    """Generate genetic risk factors dataset with three categories: rare, common, and fuzzy GRFs"""
    
    # Define gene names associated with common diseases
    genes = [
        # Cardiovascular disease genes
        'APOE', 'LDLR', 'PCSK9', 'ABCG5', 'ABCG8', 'CYP7A1', 'HMGCR', 'NPC1L1',
        # Diabetes genes  
        'TCF7L2', 'PPARG', 'KCNJ11', 'WFS1', 'HNF4A', 'GCK', 'HNF1A', 'ABCC8',
        # Cancer genes
        'BRCA1', 'BRCA2', 'TP53', 'APC', 'MLH1', 'MSH2', 'MSH6', 'PMS2',
        # Neurological genes
        'APP', 'PSEN1', 'PSEN2', 'MAPT', 'GRN', 'C9orf72', 'SNCA', 'LRRK2',
        # Immune system genes
        'HLA-DRB1', 'HLA-DQB1', 'PTPN22', 'CTLA4', 'IL2RA', 'STAT4', 'IRF5', 'TNFAIP3',
        # Metabolic genes
        'FTO', 'MC4R', 'POMC', 'LEP', 'LEPR', 'ADIPOQ', 'PPARA', 'SREBF1',
        # Additional genes for diversity
        'CFTR', 'HTT', 'DMD', 'F8', 'F9', 'HEXA', 'PAH', 'SMN1'
    ]
    
    # Generate 150 genetic variants across these genes
    variants = []
    for i in range(150):
        gene = random.choice(genes)
        
        # Generate variant information
        chromosome = random.randint(1, 22)
        position = random.randint(1000000, 250000000)
        ref_allele = random.choice(['A', 'T', 'G', 'C'])
        alt_allele = random.choice([a for a in ['A', 'T', 'G', 'C'] if a != ref_allele])
        
        # Categorize GRFs based on MAF and effect size
        maf = np.random.beta(0.5, 2)  # Skewed towards lower frequencies
        
        if maf < 0.01:
            grf_category = 'rare'
            effect_size = np.random.normal(0.8, 0.3)  # Larger effect sizes for rare variants
        elif maf < 0.05:
            grf_category = 'fuzzy'
            effect_size = np.random.normal(0.4, 0.2)  # Moderate effect sizes
        else:
            grf_category = 'common'
            effect_size = np.random.normal(0.2, 0.1)  # Smaller effect sizes for common variants
        
        # Generate p-value based on effect size (stronger effects = lower p-values)
        p_value = np.random.exponential(0.01) * (1 / max(abs(effect_size), 0.1))
        p_value = min(p_value, 0.5)  # Cap at 0.5
        
        # Generate odds ratio
        odds_ratio = np.exp(effect_size)
        
        # Confidence interval
        ci_lower = odds_ratio * np.exp(-1.96 * 0.1)
        ci_upper = odds_ratio * np.exp(1.96 * 0.1)
        
        variants.append({
            'variant_id': f'rs{1000000 + i}',
            'gene': gene,
            'chromosome': chromosome,
            'position': position,
            'ref_allele': ref_allele,
            'alt_allele': alt_allele,
            'maf': round(maf, 4),
            'grf_category': grf_category,
            'effect_size': round(effect_size, 3),
            'odds_ratio': round(odds_ratio, 3),
            'ci_lower': round(ci_lower, 3),
            'ci_upper': round(ci_upper, 3),
            'p_value': f"{p_value:.2e}",
            'associated_disease': assign_disease(gene),
            'functional_consequence': random.choice(['missense', 'synonymous', 'nonsense', 'splice_site', 'regulatory']),
            'population': random.choice(['EUR', 'EAS', 'AFR', 'AMR', 'SAS'])
        })
    
    return pd.DataFrame(variants)

def assign_disease(gene):
    """Assign disease based on gene"""
    disease_mapping = {
        'APOE': 'Alzheimer_disease', 'LDLR': 'Hypercholesterolemia', 'PCSK9': 'Hypercholesterolemia',
        'TCF7L2': 'Type2_diabetes', 'PPARG': 'Type2_diabetes', 'KCNJ11': 'Type2_diabetes',
        'BRCA1': 'Breast_cancer', 'BRCA2': 'Breast_cancer', 'TP53': 'Cancer_syndrome',
        'APP': 'Alzheimer_disease', 'PSEN1': 'Alzheimer_disease', 'SNCA': 'Parkinson_disease',
        'HLA-DRB1': 'Rheumatoid_arthritis', 'PTPN22': 'Autoimmune_disease',
        'FTO': 'Obesity', 'MC4R': 'Obesity', 'CFTR': 'Cystic_fibrosis'
    }
    return disease_mapping.get(gene, 'Complex_trait')

def generate_patient_data():
    """Generate synthetic patient data for precision medicine analysis"""
    
    n_patients = 500
    patients = []
    
    for i in range(n_patients):
        # Basic demographics
        age = np.random.normal(55, 15)
        age = max(18, min(90, age))  # Constrain age between 18-90
        
        sex = random.choice(['Male', 'Female'])
        ethnicity = random.choice(['European', 'Asian', 'African', 'Hispanic', 'Other'])
        
        # Generate genetic risk scores for different disease categories
        cardiovascular_grs = np.random.normal(0, 1)
        diabetes_grs = np.random.normal(0, 1)
        cancer_grs = np.random.normal(0, 1)
        neurological_grs = np.random.normal(0, 1)
        
        # Generate phenotypic data influenced by genetic risk scores and demographics
        # Cardiovascular risk
        cv_risk_base = 0.1 + 0.02 * (age - 50) + 0.1 * (sex == 'Male') + 0.05 * cardiovascular_grs
        cv_risk = max(0, min(1, cv_risk_base + np.random.normal(0, 0.1)))
        
        # Diabetes risk
        diabetes_risk_base = 0.08 + 0.01 * (age - 50) + 0.03 * (ethnicity in ['Hispanic', 'African']) + 0.06 * diabetes_grs
        diabetes_risk = max(0, min(1, diabetes_risk_base + np.random.normal(0, 0.08)))
        
        # Cancer risk
        cancer_risk_base = 0.12 + 0.015 * (age - 50) + 0.02 * (sex == 'Female') + 0.04 * cancer_grs
        cancer_risk = max(0, min(1, cancer_risk_base + np.random.normal(0, 0.09)))
        
        # Generate biomarkers influenced by genetic factors
        cholesterol = 180 + 30 * cardiovascular_grs + np.random.normal(0, 20)
        glucose = 90 + 15 * diabetes_grs + np.random.normal(0, 10)
        bmi = 25 + 3 * (cardiovascular_grs + diabetes_grs) / 2 + np.random.normal(0, 4)
        
        # Blood pressure
        systolic_bp = 120 + 10 * cardiovascular_grs + 0.5 * age + np.random.normal(0, 15)
        diastolic_bp = 80 + 5 * cardiovascular_grs + 0.2 * age + np.random.normal(0, 10)
        
        # Generate current health status
        has_diabetes = diabetes_risk > 0.15 and random.random() < diabetes_risk
        has_hypertension = systolic_bp > 140 or diastolic_bp > 90
        has_hyperlipidemia = cholesterol > 240
        
        # AI-predicted treatment recommendations
        treatment_recommendations = generate_treatment_recommendations(
            cardiovascular_grs, diabetes_grs, cancer_grs, has_diabetes, has_hypertension, has_hyperlipidemia
        )
        
        patients.append({
            'patient_id': f'P{i+1:04d}',
            'age': round(age, 1),
            'sex': sex,
            'ethnicity': ethnicity,
            'cardiovascular_grs': round(cardiovascular_grs, 3),
            'diabetes_grs': round(diabetes_grs, 3),
            'cancer_grs': round(cancer_grs, 3),
            'neurological_grs': round(neurological_grs, 3),
            'cv_risk_score': round(cv_risk, 3),
            'diabetes_risk_score': round(diabetes_risk, 3),
            'cancer_risk_score': round(cancer_risk, 3),
            'cholesterol_mg_dl': round(cholesterol, 1),
            'glucose_mg_dl': round(glucose, 1),
            'bmi': round(bmi, 1),
            'systolic_bp': round(systolic_bp, 1),
            'diastolic_bp': round(diastolic_bp, 1),
            'has_diabetes': has_diabetes,
            'has_hypertension': has_hypertension,
            'has_hyperlipidemia': has_hyperlipidemia,
            'ai_treatment_recommendation': treatment_recommendations['primary'],
            'ai_prevention_strategy': treatment_recommendations['prevention'],
            'ai_monitoring_frequency': treatment_recommendations['monitoring'],
            'precision_medicine_score': round(np.random.uniform(0.6, 0.95), 3)
        })
    
    return pd.DataFrame(patients)

def generate_treatment_recommendations(cv_grs, diabetes_grs, cancer_grs, has_diabetes, has_hypertension, has_hyperlipidemia):
    """Generate AI-powered treatment recommendations based on genetic risk scores and current health"""
    
    recommendations = {
        'primary': [],
        'prevention': [],
        'monitoring': 'Standard'
    }
    
    # Primary treatment recommendations
    if has_diabetes:
        if diabetes_grs > 0.5:
            recommendations['primary'].append('Metformin_personalized_dosing')
        else:
            recommendations['primary'].append('Metformin_standard')
    
    if has_hypertension:
        if cv_grs > 0.5:
            recommendations['primary'].append('ACE_inhibitor_high_dose')
        else:
            recommendations['primary'].append('ACE_inhibitor_standard')
    
    if has_hyperlipidemia:
        if cv_grs > 0.3:
            recommendations['primary'].append('Statin_high_intensity')
        else:
            recommendations['primary'].append('Statin_moderate_intensity')
    
    # Prevention strategies based on genetic risk
    if cv_grs > 0.7:
        recommendations['prevention'].append('Intensive_lifestyle_modification')
        recommendations['monitoring'] = 'High_frequency'
    elif cv_grs > 0.3:
        recommendations['prevention'].append('Moderate_lifestyle_modification')
        recommendations['monitoring'] = 'Moderate_frequency'
    
    if diabetes_grs > 0.6:
        recommendations['prevention'].append('Diabetes_prevention_program')
    
    if cancer_grs > 0.8:
        recommendations['prevention'].append('Enhanced_cancer_screening')
        recommendations['monitoring'] = 'High_frequency'
    
    # Default recommendations if none specified
    if not recommendations['primary']:
        recommendations['primary'].append('Standard_care')
    if not recommendations['prevention']:
        recommendations['prevention'].append('Standard_prevention')
    
    return {
        'primary': '; '.join(recommendations['primary']),
        'prevention': '; '.join(recommendations['prevention']),
        'monitoring': recommendations['monitoring']
    }

def generate_biomarker_data():
    """Generate biomarker data for precision medicine analysis"""
    
    biomarkers = [
        'CRP', 'IL6', 'TNF_alpha', 'HbA1c', 'LDL_cholesterol', 'HDL_cholesterol',
        'Triglycerides', 'Creatinine', 'eGFR', 'ALT', 'AST', 'PSA', 'CA125',
        'CEA', 'AFP', 'Troponin_I', 'BNP', 'D_dimer', 'Homocysteine', 'Vitamin_D'
    ]
    
    # Generate data for 500 patients
    biomarker_data = []
    for patient_id in range(1, 501):
        for biomarker in biomarkers:
            # Generate values based on biomarker type
            if biomarker == 'CRP':
                value = np.random.lognormal(0, 1)  # Log-normal distribution
                unit = 'mg/L'
                reference_range = '< 3.0'
            elif biomarker == 'HbA1c':
                value = np.random.normal(5.7, 0.8)
                unit = '%'
                reference_range = '< 5.7'
            elif biomarker == 'LDL_cholesterol':
                value = np.random.normal(130, 30)
                unit = 'mg/dL'
                reference_range = '< 100'
            elif biomarker == 'HDL_cholesterol':
                value = np.random.normal(50, 15)
                unit = 'mg/dL'
                reference_range = '> 40 (M), > 50 (F)'
            elif biomarker == 'eGFR':
                value = np.random.normal(90, 20)
                unit = 'mL/min/1.73m²'
                reference_range = '> 60'
            else:
                value = np.random.lognormal(0, 0.5)
                unit = 'ng/mL'
                reference_range = 'Variable'
            
            biomarker_data.append({
                'patient_id': f'P{patient_id:04d}',
                'biomarker': biomarker,
                'value': round(value, 2),
                'unit': unit,
                'reference_range': reference_range,
                'collection_date': datetime.now() - timedelta(days=random.randint(0, 365))
            })
    
    return pd.DataFrame(biomarker_data)

def generate_ai_model_performance():
    """Generate AI model performance metrics for different precision medicine tasks"""
    
    models = [
        'Random_Forest_GRS', 'XGBoost_GRS', 'Neural_Network_GRS', 'SVM_GRS',
        'Logistic_Regression_GRS', 'Ensemble_GRS', 'Deep_Learning_Multiomics',
        'Transformer_Genomics', 'CNN_Genomics', 'LSTM_Temporal'
    ]
    
    tasks = [
        'Disease_Risk_Prediction', 'Treatment_Response_Prediction', 'Biomarker_Discovery',
        'Drug_Dosing_Optimization', 'Adverse_Event_Prediction', 'Prognosis_Prediction'
    ]
    
    performance_data = []
    for model in models:
        for task in tasks:
            # Generate realistic performance metrics
            if 'Deep_Learning' in model or 'Transformer' in model or 'CNN' in model:
                base_performance = 0.85  # Higher for deep learning models
            elif 'Ensemble' in model:
                base_performance = 0.82
            else:
                base_performance = 0.78
            
            accuracy = base_performance + np.random.normal(0, 0.05)
            precision = accuracy + np.random.normal(0, 0.03)
            recall = accuracy + np.random.normal(0, 0.03)
            f1_score = 2 * (precision * recall) / (precision + recall)
            auc_roc = accuracy + np.random.normal(0, 0.04)
            
            # Constrain values between 0 and 1
            accuracy = max(0.5, min(0.98, accuracy))
            precision = max(0.5, min(0.98, precision))
            recall = max(0.5, min(0.98, recall))
            f1_score = max(0.5, min(0.98, f1_score))
            auc_roc = max(0.5, min(0.98, auc_roc))
            
            performance_data.append({
                'model_name': model,
                'task': task,
                'accuracy': round(accuracy, 3),
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'f1_score': round(f1_score, 3),
                'auc_roc': round(auc_roc, 3),
                'training_samples': random.randint(1000, 10000),
                'validation_samples': random.randint(200, 2000),
                'cross_validation_folds': random.choice([5, 10]),
                'hyperparameter_optimization': random.choice(['Grid_Search', 'Random_Search', 'Bayesian_Optimization'])
            })
    
    return pd.DataFrame(performance_data)

def generate_drug_response_data():
    """Generate drug response data for pharmacogenomics analysis"""
    
    drugs = [
        'Warfarin', 'Clopidogrel', 'Simvastatin', 'Metformin', 'Carbamazepine',
        'Abacavir', 'Allopurinol', 'Phenytoin', 'Codeine', 'Tamoxifen'
    ]
    
    pharmacogenes = [
        'CYP2D6', 'CYP2C19', 'CYP3A4', 'VKORC1', 'CYP2C9', 'SLCO1B1',
        'DPYD', 'TPMT', 'UGT1A1', 'HLA-B*5701'
    ]
    
    drug_response_data = []
    for patient_id in range(1, 501):
        for drug in drugs:
            # Assign relevant pharmacogenes to each drug
            relevant_genes = random.sample(pharmacogenes, random.randint(1, 3))
            
            # Generate genetic variants for pharmacogenes
            genetic_variants = []
            for gene in relevant_genes:
                variant = f"{gene}*{random.randint(1, 10)}"
                genetic_variants.append(variant)
            
            # Generate response based on genetic factors
            genetic_score = np.random.normal(0, 1)
            
            # Response categories
            if genetic_score > 1:
                response_category = 'Poor_metabolizer'
                efficacy = np.random.uniform(0.2, 0.5)
                adverse_events = np.random.uniform(0.3, 0.8)
            elif genetic_score > 0:
                response_category = 'Intermediate_metabolizer'
                efficacy = np.random.uniform(0.5, 0.8)
                adverse_events = np.random.uniform(0.1, 0.4)
            elif genetic_score > -1:
                response_category = 'Normal_metabolizer'
                efficacy = np.random.uniform(0.7, 0.9)
                adverse_events = np.random.uniform(0.05, 0.2)
            else:
                response_category = 'Ultra_rapid_metabolizer'
                efficacy = np.random.uniform(0.3, 0.6)
                adverse_events = np.random.uniform(0.1, 0.3)
            
            # AI-recommended dosing
            if response_category == 'Poor_metabolizer':
                ai_dose_recommendation = 'Reduce_dose_50%'
            elif response_category == 'Ultra_rapid_metabolizer':
                ai_dose_recommendation = 'Increase_dose_25%'
            else:
                ai_dose_recommendation = 'Standard_dose'
            
            drug_response_data.append({
                'patient_id': f'P{patient_id:04d}',
                'drug': drug,
                'pharmacogenes': '; '.join(relevant_genes),
                'genetic_variants': '; '.join(genetic_variants),
                'metabolizer_status': response_category,
                'predicted_efficacy': round(efficacy, 3),
                'predicted_adverse_events': round(adverse_events, 3),
                'ai_dose_recommendation': ai_dose_recommendation,
                'confidence_score': round(np.random.uniform(0.7, 0.95), 3)
            })
    
    return pd.DataFrame(drug_response_data)

def main():
    """Generate all datasets for AI-powered precision medicine tutorial"""
    
    print("Generating AI-powered precision medicine datasets...")
    
    # Generate datasets
    print("1. Generating genetic risk factors dataset...")
    genetic_df = generate_genetic_risk_factors()
    genetic_df.to_csv('ai_genetic_risk_factors.csv', index=False)
    print(f"   Generated {len(genetic_df)} genetic variants")
    
    print("2. Generating patient data...")
    patients_df = generate_patient_data()
    patients_df.to_csv('ai_patient_data.csv', index=False)
    print(f"   Generated data for {len(patients_df)} patients")
    
    print("3. Generating biomarker data...")
    biomarkers_df = generate_biomarker_data()
    biomarkers_df.to_csv('ai_biomarker_data.csv', index=False)
    print(f"   Generated {len(biomarkers_df)} biomarker measurements")
    
    print("4. Generating AI model performance data...")
    performance_df = generate_ai_model_performance()
    performance_df.to_csv('ai_model_performance.csv', index=False)
    print(f"   Generated performance metrics for {len(performance_df)} model-task combinations")
    
    print("5. Generating drug response data...")
    drug_response_df = generate_drug_response_data()
    drug_response_df.to_csv('ai_drug_response_data.csv', index=False)
    print(f"   Generated {len(drug_response_df)} drug response predictions")
    
    # Generate summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Genetic Risk Factors: {len(genetic_df)} variants across {genetic_df['gene'].nunique()} genes")
    print(f"  - Rare GRFs: {len(genetic_df[genetic_df['grf_category'] == 'rare'])}")
    print(f"  - Common GRFs: {len(genetic_df[genetic_df['grf_category'] == 'common'])}")
    print(f"  - Fuzzy GRFs: {len(genetic_df[genetic_df['grf_category'] == 'fuzzy'])}")
    
    print(f"\nPatient Data: {len(patients_df)} patients")
    print(f"  - Age range: {patients_df['age'].min():.1f} - {patients_df['age'].max():.1f} years")
    print(f"  - Sex distribution: {patients_df['sex'].value_counts().to_dict()}")
    print(f"  - High CV risk (>0.5): {len(patients_df[patients_df['cv_risk_score'] > 0.5])}")
    print(f"  - High diabetes risk (>0.3): {len(patients_df[patients_df['diabetes_risk_score'] > 0.3])}")
    
    print(f"\nBiomarkers: {biomarkers_df['biomarker'].nunique()} different biomarkers")
    print(f"AI Models: {performance_df['model_name'].nunique()} models tested on {performance_df['task'].nunique()} tasks")
    print(f"Drug Responses: {drug_response_df['drug'].nunique()} drugs analyzed")
    
    print("\nAll datasets generated successfully!")
    print("Files created:")
    print("  - ai_genetic_risk_factors.csv")
    print("  - ai_patient_data.csv") 
    print("  - ai_biomarker_data.csv")
    print("  - ai_model_performance.csv")
    print("  - ai_drug_response_data.csv")

if __name__ == "__main__":
    main()

