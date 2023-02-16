from utils import get_dataset, get_best_model, save_model, evaluation
from utils.data import save_dataframe
from sklearn.model_selection import train_test_split


X_all, y_all = get_dataset()

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3)


dfs = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

for key, value in dfs.items():
    save_dataframe(value, f"src/artifacts/datasets/{key}.csv")


# save_dataframe(X_train)
# save_dataframe(X_test)
# save_dataframe(y_train)
# save_dataframe(y_test)

print("Getting best model")
model = get_best_model(X_train, y_train)

save_model(model, "src/artifacts/models/model.gzip")

r2_score = evaluation(model, X_test, y_test)
print(f"Rˆ2 Score: {r2_score}")
