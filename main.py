import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

def main():
    parser = argparse.ArgumentParser(description='Maintenance prédictive des véhicules')
    parser.add_argument('--data', default='vehicle_maintenance_data.csv')
    parser.add_argument('--scenario', choices=['A','B','C'], default='A')
    parser.add_argument('--algorithm', choices=['logistic','tree','forest'], default='logistic')
    parser.add_argument('--model-path', default='model.joblib')
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare','train','evaluate','load'):
        actions.add_argument('--'+action, action='store_true')
    args = parser.parse_args()
    if args.load:
        model = load_model(args.model_path)
        print('Modèle chargé :', model)
        return
    X_train, X_test, y_train, y_test = prepare_data(args.data, args.scenario)
    if args.prepare:
        print(f'Données préparées : train={X_train.shape}, test={X_test.shape}')
    elif args.train:
        model = train_model(X_train, y_train, args.algorithm)
        save_model(model, args.model_path)
        print(f'Modèle entraîné et sauvegardé : {args.model_path}')
    elif args.evaluate:
        model = load_model(args.model_path)
        expected = list(model.feature_names_in_)
        if list(X_test.columns) != expected:
            raise ValueError('Le scénario sélectionné ne correspond pas au modèle sauvegardé.')
        evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
