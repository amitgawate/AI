from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from PlotLearning import PlotLearning

# Generate training data
# np.random.seed(12)
# Define the length of your binary string and number of samples
binary_length = 8
num_samples = 1000
num_samples_test = 200

# Generate x_train data
x_train = np.random.randint(0, 2, size=(num_samples, binary_length))


# Generate y_train data
y_train = []

for i in range(num_samples):
    # Split the list into two equal halves
    split_lists = np.split(x_train[i], 2)

    # Convert binary to decimal
    num1 = int(''.join([str(int(x)) for x in split_lists[0]]), 2)
    num2 = int(''.join([str(int(x)) for x in split_lists[1]]), 2)

    # Calculate the sum and take modulo 2
    y_train.append((num1 * num2) % 2)

y_train = np.array(y_train)

# Generate testing data 
# Generate x_test data
x_test = np.random.randint(0, 2, size=(num_samples_test, 8))


# Generate y_test data
y_test = []

for i in range(num_samples_test):
    # Split the list into two equal halves
    split_lists = np.split(x_test[i], 2)

    # Convert binary to decimal
    num1 = int(''.join([str(int(x)) for x in split_lists[0]]), 2)
    num2 = int(''.join([str(int(x)) for x in split_lists[1]]), 2)

    # Calculate the sum and take modulo 2
    y_test.append((num1 * num2) % 2)

y_test = np.array(y_test)

print('x_train:', x_test)
print('y_test:', y_test)

# Build the model
model = Sequential([
    Dense(32, activation='relu', input_shape=(binary_length,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid'),
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20)

plot = PlotLearning()

# Train the model
history_1 = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[early_stopping, plot])

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
          
# # Make a prediction 
num_samples_pred = 5
x_pred = np.random.randint(0, 2, size=(num_samples_pred, binary_length))

# Generate y_test data
y_pred_actual = []
y = []

numbers = []

for i in range(num_samples_pred):
    pair = []
    # Split the list into two equal halves
    split_lists = np.split(x_pred[i], 2)

    # Convert binary to decimal
    num1 = int(''.join([str(int(x)) for x in split_lists[0]]), 2)
    num2 = int(''.join([str(int(x)) for x in split_lists[1]]), 2)

    # Calculate the sum and take modulo 2
    y_pred_actual.append((num1 * num2) % 2)
    y.append(num1*num2)
    pair.append(num1)
    pair.append(num2)
    numbers.append(pair)

y_pred_actual = np.array(y_pred_actual)

y_pred = model.predict(x_pred)
thresholded_data = (y_pred > 0.5).astype(int)

print (numbers)
print (y)
print (y_pred.T)
print (thresholded_data.T)
print (y_pred_actual)
