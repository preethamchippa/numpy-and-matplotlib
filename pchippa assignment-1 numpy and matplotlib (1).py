#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import numpy as np


# # Numpy

# In[14]:


# 1. Create a NumPy array of shape (5, 5) filled with zeros.
A = np.zeros((5,5))
A


# In[15]:


# 2. Create a NumPy array of shape (3, 4) filled with random numbers.
B = np.random.random((3, 4))
B


# In[16]:


# 3. Find the sum of all elements in the previous NumPy array.
B.sum()


# In[17]:


# 4. Calculate the mean value of of the previous NumPy array
C = np.mean(B)
print("C:", C)


# In[18]:


# 5. Find the maximum and minimum value of the previous NumPy array.
D = B.max()
E = B.min()

print("Maximum value: ",D)
print("Minimum value: ",E)


# In[19]:


# 6. reshape the array into a 2 by 6 array
F= B.reshape(2,6)
F


# In[20]:


# 7. Write the code to perform the following tasks:

# Import the NumPy library
# Create a 2D NumPy array of shape (4, 3)
# Calculate the sum along each row
# Find the index of the row with the maximum sum
# Extract that row from the 2D array
# Print the row sums and the row with the maximum sum
import numpy as np
G = np.random.random((4,3))
H = G.sum(axis=1)
I = H.argmax()
J = G[I]
print(G)
print("Row Sums:", H)
print("Row with Maximum Sum (Index {}):".format(I), J)


# In[21]:


# 8.
# Write a Python function that takes two 2D NumPy arrays A and B of the same shape and performs the specified operations:
# adding, multiplying, dot-product

# Test your function with two 3x3 arrays of random integers
import numpy as np

def perform_operations(A, B):
    if A.shape != B.shape:
        raise ValueError("Arrays A and B must have the same shape for these operations.")
    
    addition_result = A + B

    multiplication_result = A * B

    dot_product_result = np.dot(A, B)
    
    return addition_result, multiplication_result, dot_product_result

random_array_1 = np.random.randint(1, 10, size=(3, 3))
random_array_2 = np.random.randint(1, 10, size=(3, 3))

addition, multiplication, dot_product = perform_operations(random_array_1, random_array_2)

print("Array A:")
print(random_array_1)
print("Array B:")
print(random_array_2)
print("Addition Result:")
print(addition)
print("Multiplication Result:")
print(multiplication)
print("Dot Product Result:")
print(dot_product)


# # Matplotlib

# In[22]:


# 9: Line Plot
# Generate a line plot for the equation y = x ** 2,
# where x ranges from -10 to 10
# Create an array of x values from -10 to 10

# Hint: 
# Create an array x of 400 points ranging from -10 to 10.
# Compute the corresponding y values
# Plot the curve using plt.plot(x, y).
# Add a title, and labels for the x-axis and y-axis.
# Add a grid for easier visualization.
# Finally, display the plot using plt.show().

K = np.linspace(-10, 10, 400)
L = K ** 2
plt.plot(K, L)
plt.title('Line Plot of L = K^2')
plt.xlabel('K')
plt.ylabel('L')
plt.grid(True)
plt.show()


# In[23]:


# 10: Bar Chart
# Generate a bar chart that represents the grades of 5 students, with the following data:


students = ['Alice', 'Bob', 'Charlie', 'Dave', 'Eva']
grades = [85, 90, 77, 92, 88]

# Hint:
# Create the bar chart using plt.bar().
# Add a title and labels for the x-axis and y-axis using plt.title(), plt.xlabel(), and plt.ylabel().
# Finally, display the plot using plt.show()

plt.bar(students, grades)
plt.title('Student Grades')
plt.xlabel('Students')
plt.ylabel('Grades')
plt.show()


# In[24]:


# Q11: Scatter plot
# Generate a scatter plot for the height and weight of 10 individuals. given by the following data:

height = [160, 165, 170, 175, 180, 185, 190, 195, 200, 205]
weight = [55, 60, 65, 70, 80, 85, 90, 95, 100, 105]

# Create the scatter plot
plt.scatter(height, weight)

# Add title and labels
plt.title('Height vs. Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')

# Annotate each point with the corresponding (height, weight)
for i in range(len(height)):
    plt.annotate(f'({height[i]}, {weight[i]})', (height[i], weight[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Show the plot
plt.show()


# In[25]:


# 12: Subplots
# Generate a single figure containing four subplots as follows:
# sin(x), cos(x), tan(x), 1.0/(1.0  + exp(-x)

x = np.linspace(-2 * np.pi, 2 * np.pi, 400)

sin_x = np.sin(x)
cos_x = np.cos(x)
tan_x = np.tan(x)
sigmoid_x = 1.0 / (1.0 + np.exp(-x))

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(x, sin_x)
axs[0, 0].set_title('sin(x)')

axs[0, 1].plot(x, cos_x)
axs[0, 1].set_title('cos(x)')

axs[1, 0].plot(x, tan_x)
axs[1, 0].set_title('tan(x)')

axs[1, 1].plot(x, sigmoid_x)
axs[1, 1].set_title('1.0 / (1.0 + exp(-x))')

plt.tight_layout()

plt.show()


# In[26]:


# 13: Histogram
# Generate a histogram for the following dataset representing the scores of 50 students in an exam.

scores = [50, 60, 65, 70, 75, 80, 85, 90, 95, 100,
          55, 58, 61, 67, 72, 78, 82, 88, 92, 97,
          52, 59, 63, 71, 76, 79, 84, 89, 93, 98,
          54, 57, 62, 68, 73, 77, 81, 87, 91, 96,
          51, 56, 64, 69, 74, 83, 86, 94, 99, 53]

plt.hist(scores, bins=10, edgecolor='black')

plt.title('Exam Scores Histogram')
plt.xlabel('Scores')
plt.ylabel('Frequency')

plt.show()


# In[27]:


# 14: Pie Chart
# You are given data on the number of cars sold by a dealership in a month for four different brands: Toyota, Ford, Honda, and Tesla. The numbers are as follows:
#
# Toyota: 45 cars
# Ford: 30 cars
# Honda: 15 cars
# Tesla: 10 cars
# Create a pie chart using Matplotlib to visualize the share of monthly car sales for each brand. Make sure to include:

# Labels for each brand
# Percentages on the pie chart
# A title
# Different colors for each section

brands = ['Toyota', 'Ford', 'Honda', 'Tesla']
sales = [45, 30, 15, 10]

colors = ['#ff9999', '#66b3ff', '#99ff99', '#c2c2f0']

plt.pie(sales, labels=brands, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title('Monthly Car Sales by Brand')

plt.show()


# In[ ]:




