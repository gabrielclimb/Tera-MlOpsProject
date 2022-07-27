from handlers.data import get_california_house_data, save_dataframe
from handlers.model import get_best_model, save_model, evaluation
from sklearn.model_selection import train_test_split


X_all, y_all = get_california_house_data()
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3)

data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

for key, value in data.items():
    save_dataframe(value, f"src/artifacts/data/{key}.csv")
    # save_dataframe(X_test, "src/artifacts/data/X_test.csv")
    # save_dataframe(y_train, "src/artifacts/data/y_train.csv")
    # save_dataframe(y_test, "src/artifacts/data/y_test.csv")

print(f"Train dataframe size: {X_train.shape}\nTest dataframe size:{X_test.shape}")
# Modelagem
model = get_best_model(X_train, y_train)
save_model(model, "src/artifacts/model/model.gzip")
print("model saved")

# Avaliação
r2_score = evaluation(model, X_test, y_test)
print(f"R^2 score: {r2_score}")
