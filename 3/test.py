from os import X_OK


def predict():
    if not hasattr(predict, 'x'):
        predict.x = 0
    print(predict.x)
    predict.x += 1

for i in range(3):
    predict()