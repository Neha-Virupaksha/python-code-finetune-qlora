# QLoRA Fine-tune Evaluation Report

Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`  
Adapter: `qwen2.5-coder-pandas-lora-v2`  
Eval set: `eval_holdout.jsonl` (332 examples)

## Quantitative: held-out completion loss / perplexity

| Model | Mean loss | Perplexity |
|---|---|---|
| Base | 0.7727 | 2.166 |
| Fine-tuned | 0.5489 | 1.731 |

**Loss reduction:** 0.2238  
**Perplexity improvement:** 20.0% relative to base

The fine-tuned model assigns higher probability to the reference completions than the base model does on held-out data -- i.e. it has measurably adapted toward this dataset's style/content.

## Qualitative: sample generations

### Sample 1

**Instruction:** Create a data visualization in Python using Matplotlib that displays the total number of cases for coronavirus in different countries.

**Input:** [[United States, 212000], [Russia, 180000], [Italy, 105000], [Spain, 90000], [Germany, 70000]]

**Reference output:**
```
import matplotlib.pyplot as plt

country_data = [['United States', 212000], ['Russia', 180000], ['Italy', 105000], ['Spain', 90000], ['Germany', 70000]]
country = [data[0] for data in country_data]
cases = [data[1] for data in country_data]

plt.barh(country, cases)
plt.title("Total number of Coronavirus cases by country")
plt.xlabel('Number of cases')
plt.ylabel('Country')
plt.show()
```

**Base model output:**
```
```python
import matplotlib.pyplot as plt

# Data
countries = ["United States", "Russia", "Italy", "Spain", "Germany"]
cases = [212000, 180000, 105000, 90000, 70000]

# Create bar chart
plt.bar(countries, cases)

# Add title and labels
plt.title("Total Cases for Coronavirus by Country")
plt.xlabel("Country")
plt.ylabel("Number of Cases")

# Show the plot
plt.show()
```

This code snippet will generate a bar chart showing the total number of confirmed cases for coronavirus in different countries. The `matplotlib` library is used to create the visual representation. The `bar()` function is used to draw the bars, and the `title()`, `xlabel()`, and `ylabel()` functions are used to add a title and labels to the chart. Finally, `plt.show()` is
```

**Fine-tuned model output:**
```
import matplotlib.pyplot as plt

data = [['United States', 212000], ['Russia', 180000], ['Italy', 105000], ['Spain', 90000], ['Germany', 70000]]
labels = [country[0] for country in data]
values = [country[1] for country in data]

plt.bar(labels, values)
plt.title('Total Number of Cases for Coronavirus')
plt.xlabel('Country')
plt.ylabel('Number of Cases')

plt.show()
```

---

### Sample 2

**Instruction:** Design a machine learning classifier using the scikit-learn library in Python that predicts whether someone is a male or female using features like height and age.

**Reference output:**
```
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# define features (height, age) and labels (gender)
X = [[180, 25], [170, 27], [150, 40], [145, 37], [177, 23]]
y = ['male', 'female', 'female', 'male', 'female']

# create model 
svm_model = svm.SVC(gamma='auto', decision_function_shape='ovo')
knn_model = KNeighborsClassifier(n_neighbors=5)

# train and predict
svm_model.fit(X, y)
pred = svm_model.predict([[160, 32]])
knn_model.fit(X, y)
pred_knn = knn_model.predict([[160, 32]])

print(pred)
print(pred_knn)
```

**Base model output:**
```
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv('gender_data.csv')

# Preprocess the data
X = data[['height', 'age']]
y = data['gender']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Evaluate the model
accuracy = model.score(X_test_scaled, y_test)
print(f
```

**Fine-tuned model output:**
```
import numpy as np
from sklearn.svm import SVC

# Define the features and labels
X = np.array([[1.75, 20], [1.85, 30], [1.95, 40]])
y = np.array([0, 1, 0])

# Create an SVM model
model = SVC(kernel='linear')

# Train the model
model.fit(X, y)

# Make predictions
predictions = model.predict([[1.8, 25]])

print(predictions)
```

---

### Sample 3

**Instruction:** Implement a Tensorflow 2.0 model in Python to classify MNIST handwritten digits.

**Input:** Not applicable

**Reference output:**
```
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,  Flatten

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess input data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Create model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Compile and fit the model
model.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy'])

model.fit(x_train, y_train,
 batch_size=128,
 epochs=5,
 verbose=1,
 validation_data=(x_test, y_test))
```

**Base model output:**
```
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model with appropriate loss function and optimizer
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Train the model on the training data
model.fit(x_train, y_train, epochs=10)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
```

This solution provides a complete implementation of a Tensorflow 2
```

**Fine-tuned model output:**
```
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load the data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images / 255.0
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images / 255.0

# Build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Max
```

---

### Sample 4

**Instruction:** Create a program in Python to solve this equation: X^2 - 6X + 9.

**Reference output:**
```
import numpy as np

# Define the function
def function(x):
    return x**2 - 6*x + 9

# Calculate the roots
roots = np.roots([1, -6, 9])

print("The solutions for the equation are: x={} and x={}".format(roots[0], roots[1]))
```

**Base model output:**
```
Here is a simple Python solution for the given equation:

```python
import math

# Define the coefficients of the quadratic equation
a = 1
b = -6
c = 9

# Calculate the discriminant
discriminant = b**2 - 4*a*c

# Check if the discriminant is positive, negative or zero
if discriminant > 0:
    # Two distinct real roots
    root1 = (-b + math.sqrt(discriminant)) / (2*a)
    root2 = (-b - math.sqrt(discriminant)) / (2*a)
    print("The roots are:", root1, "and", root2)
elif discriminant == 0:
    # One real root (repeated root)
    root = -b / (2*a)
    print("The root is:", root)
else:
    # Complex roots
    real_part = -b / (2*a)
    imaginary_part = math.sqrt(-
```

**Fine-tuned model output:**
```
import numpy as np

# Define the coefficients
a = 1
b = -6
c = 9

# Calculate the discriminant
discriminant = b**2 - 4*a*c

# Calculate the two solutions
x1 = (-b + np.sqrt(discriminant)) / (2*a)
x2 = (-b - np.sqrt(discriminant)) / (2*a)

print("The solutions are:", x1, "and", x2)
```

---

### Sample 5

**Instruction:** Generate a python code to sort an array of numbers in descending order.

**Reference output:**
```
def sort_array(arr): 
    arr.sort(reverse = True) 
    return arr
```

**Base model output:**
```
```python
def sort_descending(arr):
    arr.sort(reverse=True)
    return arr

# Example usage
numbers = [5, 3, 8, 1, 9]
sorted_numbers = sort_descending(numbers)
print(sorted_numbers)  # Output: [9, 8, 5, 3, 1]
```


**Explanation**:
The function `sort_descending` takes an array as input and sorts it in descending order using the built-in `sort()` method with the `reverse=True` parameter. The sorted array is then returned. In the example usage, we have an array `[5, 3, 8, 1, 9]`, which is sorted into `[9, 8, 5, 3, 1]`. This demonstrates how the function works by sorting the array in descending order.
```

**Fine-tuned model output:**
```
def sort_descending(arr):
    arr.sort(reverse=True)
    return arr
```

---
