import argparse




def build_pipeline(numeric_features, categorical_features):
numeric_transformer = Pipeline(steps=[
('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
('ohe', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(transformers=[
('num', numeric_transformer, numeric_features),
('cat', categorical_transformer, categorical_features)
])


model = RandomForestRegressor(n_estimators=200, random_state=42)


pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
return pipeline




def evaluate_and_save(pipeline, X_test, y_test, out_path):
preds = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)


metrics = {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
with open(out_path + '/metrics.json', 'w') as f:
json.dump(metrics, f, indent=2)


joblib.dump(pipeline, out_path + '/model.joblib')
return metrics


if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--out', type=str, required=True)
args = parser.parse_args()


df = load_data(args.data)
X_train, X_test, y_train, y_test = preprocess(df)


numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()


pipeline = build_pipeline(numeric_features, categorical_features)
pipeline.fit(X_train, y_train)


metrics = evaluate_and_save(pipeline, X_test, y_test, args.out)
print('Done. Metrics:', metrics)
