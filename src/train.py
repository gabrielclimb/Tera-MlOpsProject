from utils.data import get_california_housing, save_dataframe
from utils.model import get_best_model, save_model, evaluation
from sklearn.model_selection import train_test_split

X_all, y_all = get_california_housing()
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3)

dataframes = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
}

for key, value in dataframes.items():
    save_dataframe(value, f"src/documents/data/{key}.csv")

# save_dataframe(X_train, "src/documents/train.csv")
# save_dataframe(X_test, "src/documents/X_test.csv")
# save_dataframe(y_train, "src/documents/y_train.csv")
# save_dataframe(y_test, "src/documents/y_test.csv")

print(f"Train dataframe size: {X_train.shape}")
print(f"Test dataframe size: {X_test.shape}")

model = get_best_model(X_train, y_train)

save_model(model, "src/documents/model/model.gzip")
print("Save model")

r2_score = evaluation(model, X_test, y_test)
print(f"Rˆ2 score: {r2_score}")
