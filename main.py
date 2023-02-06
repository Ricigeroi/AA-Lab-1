import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_iterative(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

def fibonacci_dynamic(n):
    if n <= 1:
        return n
    fib = [0] * (n + 1)
    fib[0] = 0
    fib[1] = 1
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib[n]

def fibonacci_matrix(n):
    n+=2;
    if n <= 1:
        return n
    def matrix_mult(A, B):
        C = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    C[i][j] += A[i][k] * B[k][j]
        return C
    def matrix_pow(A, n):
        if n <= 1:
            return A
        B = matrix_pow(A, n // 2)
        B = matrix_mult(B, B)
        if n % 2 != 0:
            B = matrix_mult(B, A)
        return B
    A = [[0, 1], [1, 1]]
    A = matrix_pow(A, n - 1)
    return A[0][0]


def fibonacci_binet(n):
    phi = (1 + math.sqrt(5)) / 2
    psi = (1 - math.sqrt(5)) / 2
    return int((phi**n - psi**n) / math.sqrt(5))

def fibonacci_iterative_space_optimized(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for i in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr

def fibonacci_fast_doubling(n):
    def fib(n):
        if n == 0:
            return 0, 1
        else:
            a, b = fib(n // 2)
            c = a * ((2 * b) - a)
            d = a * a + b * b
            if n % 2 == 0:
                return c, d
            else:
                return d, c + d

    if n >= 0:
        return fib(n)[0]
    else:
        if n % 2 == 0:
            return fib(-n)[0]
        else:
            return -fib(-n)[0]

###############################################################


def time_fibonacci(n):
    start_time = time.time()

    # Here call desired algorithm (functions above)
    fibonacci_fast_doubling(n)

    return time.time() - start_time

# here should be input
inputs = []

# !! Here set input !!
for i in range(0, 10001, 500):
    inputs.append(i)

times = [time_fibonacci(n) for n in inputs]

# Prints results table in terminal
print("n\tTimes")
for i, time in enumerate(times):
    print(f"{inputs[i]}\t{time:.4f}"+'s')



# Plot results graph
plt.grid()
plt.plot(inputs, times, linewidth=1)
plt.title("Fast Doubling Algorithm")
plt.scatter(inputs, times, label='Results')
plt.xlabel("n-th Fibonacci Term")
plt.ylabel("Time (s)")

# Linear regression stuff and plot
inputs = np.array(inputs).reshape(-1, 1)
times = np.array(times)
reg = LinearRegression().fit(inputs, times)

# UNCOMMENT IF YOU ARE INTERESTED IN THIS LINE
# plt.plot(inputs, reg.predict(inputs), color='red', linewidth=1, linestyle='-', label='Linear regression')

plt.legend()
plt.show()


