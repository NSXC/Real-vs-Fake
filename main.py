import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from joblib import dump

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


human_codes = np.array([111000,900000,800000,123123,919191,101010,999000,808080,191919,313131,212121,200000,445445,999000,888000,889889,333222, 111111, 222222, 333333, 444444, 555555, 666666, 777777, 888888, 999999, 234567, 345678, 456789, 567890, 123123, 321321, 234234, 567567, 789789, 987987, 987654, 101010, 101010, 555999, 121212, 111000, 90909, 909090, 800800, 700700, 989878,600600, 999000, 888000, 777000, 111333, 222444, 55566, 555000, 111000, 222000, 777000, 707070, 999000,888000,000000])
bot_codes = np.array([235343,192502,704635,14005,816452,816452,105344,797448,798087,731182,631172,494549,273390,853091,813125,671057,782023,433233,918539,684632,338161,404179,331814,611084, 542177,374011,506855,496480,316764,486289,392537, 535034, 268967, 343771,580671,970550,631884, 481473, 679869, 396041, 712450, 417019, 275620, 659923, 363847, 861488, 223169, 217662, 363847, 256607, 662996, 427125, 580070, 708138, 95115, 840207, 698701, 714862, 140145, 108922, 850495, 965128, 694641, 461200, 983897, 128984, 670766, 840147, 369914, 952753, 465332, 10264, 585808, 9253, 432534, 713667, 626678, 787100, 464086, 402342, 475342, 547230, 419466, 473065, 888043, 194364, 954462, 851779, 710895, 726159, 307330, 604046])

human_labels = np.ones(human_codes.shape[0])
bot_labels = np.zeros(bot_codes.shape[0])

X = np.concatenate((human_codes, bot_codes))
y = np.concatenate((human_labels, bot_labels))

X_onehot = np.zeros((X.shape[0], 6, 10))
for i in range(X.shape[0]):
    code = str(X[i])
    if len(code) < 6:
        code = '0' * (6 - len(code)) + code
    for j in range(6):
        digit = int(code[j])
        X_onehot[i, j, digit] = 1

y_onehot = np.zeros((y.shape[0], 2))
y_onehot[np.arange(y.shape[0]), y.astype(int)] = 1

X_train, X_test, y_train, y_test = train_test_split(X_onehot, y_onehot, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(6, 10)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", test_acc)
model.save("newmod.h5")
