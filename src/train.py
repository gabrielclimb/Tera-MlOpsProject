from datetime import datetime
from helpers.data import save_dataframe, get_california_house_data
from helpers.model import get_best_model,save_model, evaluation
from sklearn.model_selection import train_test_split

X, y = get_california_house_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

now = datetime.now().strftime("%Y-%m-%d-%H-%M")

save_dataframe(X_train, f"artifacts/X_train_{now}.csv")
save_dataframe(y_train, f"artifacts/y_train_{now}.csv")

print(f"Train dataframe size: {X_train.shape}")
print(f"Test dataframe size: {X_test.shape}")

model = get_best_model(X_train, y_train)
save_model(model, f"artifacts/model_{now}.gzip")

r2_score = evaluation(model, X_test, y_test)
print(f"Rˆ2 score: {r2_score}")

