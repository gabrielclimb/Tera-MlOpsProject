from helpers.data import get_california_house_data, save_dataframe
from helpers.model import get_best_model, save_model, evaluation
from sklearn.model_selection import train_test_split

X, y = get_california_house_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

for key, value in data.items():
    save_dataframe(value, f"pipeline_deploy/artifacts/{key}")

print(f"Train Dataframe Size: {X_train.shape}\nTest Dataframe Size: {X_test.shape}")

model = get_best_model(X_train, y_train)
save_model(model, "pipeline_deploy/artifacts/model.gzip")
r2 = evaluation(model, X_test, y_test)
print(f"Rˆ2 -> {r2}")
