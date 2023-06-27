from datetime import datetime
from sklearn.model_selection import train_test_split

from utils import get_california_house_data, save_dataframe
from utils.model import evaluation, get_best_model, save_model

X, y = get_california_house_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


data ={
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}

now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
for key, value in data.items():
    save_dataframe(value, f"src/artifacts/data/{key}_{now}.csv")

print(f"Train dataframe size: {X_train.shape}\nTest dataframe size: {X_test.shape}")

model = get_best_model(X_train, y_train)
save_model(model, f"src/artifacts/model/model_{now}.gzip")
print("model saved")

r2_score = evaluation(model, X_test, y_test)
print(f"Rˆ2: {r2_score}")