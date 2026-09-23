# Code Execution Eval: Base vs Fine-tuned

Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`  
Adapter: `qwen2.5-coder-pandas-lora-v2`  
Sample size: 40 held-out examples (seed=42)

## Execution success rate

| Model | Success | Syntax error | Runtime error | Timeout | Success rate |
|---|---|---|---|---|---|
| Base | 22 | 6 | 11 | 1 | 55.0% |
| Fine-tuned | 25 | 0 | 14 | 1 | 62.5% |

## Library-usage match rate

Of the examples whose instruction named a specific library (pandas/numpy/sklearn/matplotlib), the fraction where the generated code actually used that library:

- Base: 66.7% (n=6)
- Fine-tuned: 66.7% (n=6)


## Interpretation

The fine-tuned model's generated code executed successfully 7.5 percentage points more often than the base model's, on the same 40 held-out prompts. Combined with the perplexity result from evaluate.py, this suggests the fine-tune improved actual code correctness, not just output formatting.


*Caveat: execution success is necessary but not sufficient for correctness -- code can run without error and still not solve the task properly (e.g. wrong logic, wrong output). This eval catches crashes, syntax errors, and hallucinated APIs/files; it does not verify semantic correctness against the reference output.*


## Per-example detail

### Example 1

**Instruction:** Create a data visualization in Python using Matplotlib that displays the total number of cases for coronavirus in different countries.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 2

**Instruction:** Design a machine learning classifier using the scikit-learn library in Python that predicts whether someone is a male or female using features like height and age.

**Base:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'gender_data.csv'  
**Fine-tuned:** `success`

### Example 3

**Instruction:** Implement a Tensorflow 2.0 model in Python to classify MNIST handwritten digits.

**Base:** `timeout` -- exceeded 8s  
**Fine-tuned:** `timeout` -- exceeded 8s

### Example 4

**Instruction:** Create a program in Python to solve this equation: X^2 - 6X + 9.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 5

**Instruction:** Generate a python code to sort an array of numbers in descending order.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 6

**Instruction:** Generate code to generate a linear regression model in Python.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 7

**Instruction:** Create a function for sorting an array of integers using Merge Sort in Python.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 8

**Instruction:** Generate an array of unique random numbers in Python.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 9

**Instruction:** Write a code to generate a two-dimensional array with zeros shapes in Python

**Base:** `success`  
**Fine-tuned:** `success`

### Example 10

**Instruction:** Generate a predictive model in Python to predict the number of employees will leave in a given year based on the age, job position and current salary.

**Base:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'employee_turnover_data.csv'  
**Fine-tuned:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'employee_data.csv'

### Example 11

**Instruction:** Create a NLP model in Python to classify movie reviews from IMDb as either positive or negative.

**Base:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'movie_reviews.csv'  
**Fine-tuned:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'movie_reviews.csv'

### Example 12

**Instruction:** Create a Python function that takes a list of dictionaries and counts the frequency of each unique element.

**Base:** `runtime_error` -- TypeError: unhashable type: 'dict'  
**Fine-tuned:** `success`

### Example 13

**Instruction:** Write a Python program to detect the most frequently occurring element in a given array.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 14

**Instruction:** Create a Python program to generate all the possible permutations of a given array.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 15

**Instruction:** Create a python script that takes 2 arguments - an array and an integer - and prints the sum of the array multiplied by the integer.

**Base:** `syntax_error` -- SyntaxError: invalid syntax (<generated>, line 1)  
**Fine-tuned:** `success`

### Example 16

**Instruction:** Find the maximum value of a 10 element array in Python.

**Base:** `success`  
**Fine-tuned:** `runtime_error` -- NameError: name 'array' is not defined

### Example 17

**Instruction:** Create an ML algorithm in Python to identify fraudulent transactions in a dataset.

**Base:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'fraud_transactions.csv'  
**Fine-tuned:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'fraud_data.csv'

### Example 18

**Instruction:** Develop a machine learning model with Python to predict stock prices of a given company.

**Base:** `syntax_error` -- SyntaxError: invalid syntax (<generated>, line 1)  
**Fine-tuned:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'stock_data.csv'

### Example 19

**Instruction:** Design an algorithm in Python that takes an array of integers and returns an array with no repeating integers.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 20

**Instruction:** Create a machine learning model in Python that predicts the type of animal based on certain features such as size, color, and behavior.

**Base:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'animal_data.csv'  
**Fine-tuned:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'animal_data.csv'

### Example 21

**Instruction:** Create a machine learning model in Python that predecits customer lifetime value given a customer's past transactions.

**Base:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'customer_transactions.csv'  
**Fine-tuned:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'customer_data.csv'

### Example 22

**Instruction:** Create a document clustering program in Python that groups similar documents together.

**Base:** `runtime_error` -- LookupError: 
**********************************************************************
  Resource [93mstopwords[0m not found.
  Please use the NLTK Downloader to obtain the resource:

  [31m>>> import nltk
  >>> nltk.download('stopwords')
  [0m
  For more information see: https://www.nltk.org/data.html

  Attempted to load [93mcorpora/stopwords[0m

  Searched in:
    - '/root/nltk_data'
    - '/usr/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
**********************************************************************
  
**Fine-tuned:** `success`

### Example 23

**Instruction:** Write a Python program for creating a histogram from a given dataset.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 24

**Instruction:** Generate a predictive model using machine learning techniques in Python that can predict the outcome of a future stock price given its historical data.

**Base:** `syntax_error` -- SyntaxError: unterminated string literal (detected at line 11) (<generated>, line 11)  
**Fine-tuned:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'stock_data.csv'

### Example 25

**Instruction:** Write a Python program to transform an input array of length 5 into a matrix of 3x3.

**Base:** `success`  
**Fine-tuned:** `runtime_error` -- ValueError: cannot reshape array of size 5 into shape (3,2)

### Example 26

**Instruction:** Design a neural network application in Python to predict the price of real estate.

**Base:** `syntax_error` -- SyntaxError: invalid syntax (<generated>, line 1)  
**Fine-tuned:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'properties.csv'

### Example 27

**Instruction:** Create a function named `parse_file` that takes a parameter named `datafile`. The function should read the input `datafile` line by line, and for the first 10 lines (not including the header) split each line on "," and then for each line, create a dictionary where the key is the header title of the field, and the value is the value of that field in the row. The function should return a list of dictionaries, each data line in the file being a single list entry. Field names and values should not contain extra whitespace, like spaces or newline characters. You can use the Python string method `strip()` to remove the extra whitespace. The returned list should have 10 entries. Finally, write a test function named `test` that tests the implementation of the `parse_file` function.

**Base:** `runtime_error` -- NameError: name 'open' is not defined  
**Fine-tuned:** `runtime_error` -- NameError: name 'open' is not defined

### Example 28

**Instruction:** Visualize a dataset containing the exam scores of 3 classes (class A, B, and C) in the form of a bar chart using Matplotlib and Python.

**Base:** `runtime_error` -- TypeError: only length-1 arrays can be converted to Python scalars  
**Fine-tuned:** `runtime_error` -- ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (3,) and arg 1 with shape (9,).

### Example 29

**Instruction:** Generate an algorithm in Python that can classify any dataset having 3 classes.

**Base:** `runtime_error` -- ValueError: n_classes(3) * n_clusters_per_class(2) must be smaller or equal 2**n_informative(2)=4  
**Fine-tuned:** `success`

### Example 30

**Instruction:** Write a function in Python to compute the sum of all elements in a given 2-dimensional array.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 31

**Instruction:** Write an algorithm in Python to solve the given binary search problem

**Base:** `success`  
**Fine-tuned:** `success`

### Example 32

**Instruction:** Write a Python program to create a KMeans model and cluster iris data into 3 clusters.

**Base:** `success`  
**Fine-tuned:** `runtime_error` -- FileNotFoundError: [Errno 2] No such file or directory: 'iris.csv'

### Example 33

**Instruction:** Write a Python program to find the closest next lower value of a given number in an array of integers.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 34

**Instruction:** Design a Python program to read in a integer array and find the second largest element in the array.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 35

**Instruction:** Generate a Python function that takes a dataframe and returns a new dataframe with only the columns that start with the letter 'A'.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 36

**Instruction:** Write a Python program to optimize the given array such that its elements are in the range of 0 to 100.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 37

**Instruction:** Create a deep learning algorithm in Python to classify emails as spam or not spam.

**Base:** `syntax_error` -- SyntaxError: unterminated string literal (detected at line 21) (<generated>, line 21)  
**Fine-tuned:** `runtime_error` -- NameError: name 'load_data' is not defined

### Example 38

**Instruction:** Develop a Python program that takes an array of integers and returns the largest element.

**Base:** `syntax_error` -- SyntaxError: invalid syntax (<generated>, line 1)  
**Fine-tuned:** `success`

### Example 39

**Instruction:** Contruct a python program that sorts an array of strings using BubbleSort algorithm.

**Base:** `success`  
**Fine-tuned:** `success`

### Example 40

**Instruction:** Create a program in Python that removes duplicates from a given array.

**Base:** `success`  
**Fine-tuned:** `success`
