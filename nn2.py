from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback
import numpy as np
from  matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


model = Sequential()
model.add(Dense(1, input_dim=784, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# generate dummy data
data = np.random.random((100, 784))
labels = np.random.randint(2, size=(100, 1))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.8, random_state=0)
print(data.shape)

# train the model, iterating on the data in batches
# of 32 samples
score = model.test_on_batch(X_test, y_test)
print('metrics names: ', model.metrics_names, 'metrics: ', score)
accu = score[1]
model.fit(X_train, y_train, nb_epoch=78, batch_size=32)
print(X_test[0], y_test[0], model.predict(np.array([X_test[0]])))

# while(accu < 0.9):
#     model.train_on_batch(X_train, y_train)
        # y_pred = []
        # for i in range (0,49):
        #     prediction = model.predict(np.array([data[i]]))
        #     y_pred.append(prediction[0])
        # print(y_pred)
    # score = model.test_on_batch(X_test, y_test)
    # print('metrics names: ', model.metrics_names, 'metrics: ', score)
    # accu = score[1]
    # model.fit(X_train, y_train, nb_epoch=78, batch_size=32)
    # print(model.predict(np.array([X_test[0]])))
