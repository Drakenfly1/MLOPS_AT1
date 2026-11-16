from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES_A = ['Mileage','Maintenance_History','Reported_Issues','Vehicle_Age','Fuel_Type','Transmission_Type','Engine_Size','Odometer_Reading','Owner_Type','Insurance_Premium','Service_History','Accident_History','Fuel_Efficiency']
MECHANICAL = ['Battery_Status','Brake_Condition','Tire_Condition']
NUMERIC = ['Mileage','Reported_Issues','Vehicle_Age','Engine_Size','Odometer_Reading','Insurance_Premium','Fuel_Efficiency']
ORDINAL = ['Maintenance_History']
ORDINAL_ORDER = [['Poor','Average','Good']]

def prepare_data(data_path, scenario='A'):
    """Charge le CSV, construit risk_level et sépare train/test (80/20 stratifié)."""
    df = pd.read_csv(data_path)
    risks = {'Battery_Status': {'New':0,'Good':1,'Weak':2}, 'Brake_Condition': {'New':0,'Good':1,'Worn Out':2}, 'Tire_Condition': {'New':0,'Good':1,'Worn Out':2}}
    for col, mapping in risks.items():
        df[col+'_Risk'] = df[col].map(mapping)
    df['Risk_Score'] = df[[c+'_Risk' for c in risks]].sum(axis=1, min_count=3)
    df['risk_level'] = df['Risk_Score'].map(lambda s: 'low' if s <= 1 else ('medium' if s <= 3 else 'high') if pd.notna(s) else None)
    features = {'A': FEATURES_A, 'B': FEATURES_A+MECHANICAL, 'C': MECHANICAL}.get(scenario.upper())
    if features is None:
        raise ValueError('Le scénario doit être A, B ou C')
    dataset = df[features+['risk_level']].dropna()
    return train_test_split(dataset[features], dataset['risk_level'], test_size=0.20, random_state=42, stratify=dataset['risk_level'])

def create_preprocessor(features):
    """Construit un transformateur adapté aux colonnes du scénario."""
    numeric = [c for c in NUMERIC if c in features]
    ordinal = [c for c in ORDINAL if c in features]
    nominal = [c for c in features if c not in numeric+ordinal]
    steps = []
    if numeric: steps.append(('numeric', StandardScaler(), numeric))
    if ordinal: steps.append(('ordinal', OrdinalEncoder(categories=ORDINAL_ORDER, handle_unknown='use_encoded_value', unknown_value=-1), ordinal))
    if nominal: steps.append(('nominal', OneHotEncoder(handle_unknown='ignore'), nominal))
    return ColumnTransformer(steps)

def train_model(X_train, y_train, algorithm='logistic'):
    """Entraîne un pipeline complet : preprocessing + classifieur."""
    algorithms = {
        'logistic': LogisticRegression(max_iter=2000, random_state=42),
        'tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
    }
    if algorithm not in algorithms: raise ValueError(f'Algorithme inconnu : {algorithm}')
    model = Pipeline([('preprocessing', create_preprocessor(list(X_train.columns))), ('model', algorithms[algorithm])])
    return model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    """Affiche accuracy, F1 macro et rapport par classe."""
    predictions = model.predict(X_test)
    metrics = {'accuracy': accuracy_score(y_test, predictions), 'f1_macro': f1_score(y_test, predictions, average='macro')}
    print(f"Accuracy : {metrics['accuracy']:.4f}\nF1 macro : {metrics['f1_macro']:.4f}")
    print(classification_report(y_test, predictions, zero_division=0))
    return metrics

def save_model(model, model_path='model.joblib'):
    """Sauvegarde le pipeline complet, prétraitement compris."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path

def load_model(model_path='model.joblib'):
    """Recharge un pipeline provenant d'une source de confiance."""
    return joblib.load(model_path)
