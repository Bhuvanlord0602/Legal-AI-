from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

def train_model(X, y):
    model.fit(X, y)

def predict(X):
    return model.predict(X)