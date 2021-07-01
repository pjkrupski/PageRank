import numpy as np
import matplotlib.pyplot as plt
import csv


def create_transition_matrix(m, b, d):
    transition_array = np.full((m, m), 0.0)
    for i in range(m):
        if i == 0:
            transition_array[0, 0] = 1 - b
            transition_array.__setitem__((0, 1), b)  # [0, 1] = b
            continue
        if i == m - 1:
            transition_array[m - 1, m - 1] = 1 - d
            transition_array[m - 1, m - 2] = d
            continue
        else:
            transition_array[i, i] = 1 - b - d
            transition_array[i, i + 1] = b
            transition_array[i, i - 1] = d
    return transition_array


def compute_pi(num, T, pi):
    epsilon_vals = []
    pi_vals = []
    i_vals = []
    answers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    temp_pi = pi
    for i in range(num):
        temp_pi_before = temp_pi
        temp_pi = np.matmul(temp_pi, T)
        epsilon_vals.append(np.sum(np.absolute(np.subtract(temp_pi_before, temp_pi))))
        pi_vals.append(temp_pi)
        i_vals.append(i)
    plot_vals(epsilon_vals, i_vals, " Epsilon ", " epsilon(t) vs time")
    for r in pi_vals:
        e = 0
        for c in r:
            answers[e] += c
            e += 1
    return np.divide(answers, 100)


def compute_pi_google(num, T, pi):
    # num is num of iterations (100) T is the G matrix
    # pi is the unifor ditribution, 1X9664 mat with 1/9664
    # for each element
    epsilon_vals = []
    pi_vals = []
    i_vals = []
    answers = np.full((1, len(pi)), 0.0)
    temp_pi = pi
    for i in range(num):
        temp_pi_before = temp_pi
        temp_pi = np.matmul(temp_pi, T)
        epsilon_vals.append(np.sum(np.absolute(np.subtract(temp_pi_before, temp_pi))))
        pi_vals.append(temp_pi)
        i_vals.append(i)
    plot_vals(epsilon_vals, i_vals, " Epsilon ", " epsilon(t) vs time")
    return temp_pi


def plot_vals(y_vals, x_vals, y_lab, title):
    _ = plt.scatter(x_vals, y_vals)
    _ = plt.xlabel("States")
    _ = plt.ylabel(y_lab)
    _ = plt.title(title)
    plt.show()


# Calculates state after 100 moves
def monte_carlo_sample(b, d, pi):
    # b=.2 d = .5
    for i in range(100):
        # print(pi, " is pi on iteration ", i)
        x = rand_num(1)
        if pi == 0:
            if x <= b:
                pi += 1
                continue
            else:
                pi = pi
                continue
        if pi == 9:
            if x >= d:
                pi -= 1
                continue
            else:
                pi = pi
                continue
        else:
            if x <= b:
                pi += 1
                continue
            if x >= d:
                pi -= 1
                continue
            if b < x < d:
                pi = pi
                continue
    return pi


def rand_num(num):
    x = np.random.rand(num)
    return x


def calculate_pi_n_i():
    pi_n_i = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(1000):
        # passed pi_0
        # will increment one of the 10 pi values
        pi_n_i[monte_carlo_sample(.2, .5, 0)] += 1
    return pi_n_i


def calculate_delta(a, b):
    # returns difference between two vectors divided by num occurances in a
    # a is original vector, b is monte carlo vector
    answer = []
    # print(a.__len__(), " is a leng", b.__len__(), " is b leng")
    # print(a)
    # print(b)
    temp = np.divide(np.absolute(np.subtract(a, b)), a)
    y_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plot_vals(temp, y_vals, "Delta(i)", "percent error for each state")
    return temp


def calculate_equl_dist(T, s, num):
    # print(np.linalg.matrix_power(T, num))
    for i in range(100):
        s = np.matmul(s, T)
        # print(s)


# names is 1 X 9683
# L is 9664 X 9664
def build_t2(t2):
    num_ones = num_ones_in_rows()
    vfile_vocab = open("L.csv")
    read_csv_vocab = csv.reader(vfile_vocab)
    i = 0
    for row in read_csv_vocab:
        j = 0
        k = 0
        for col in row:
            if col > '0':
                t2[(i, j)] = 1 / num_ones[i]
                k += 1
            j += 1
        if k == 0:
            t2[(i, i)] = 1
        i += 1
    return t2


def num_ones_in_rows():
    num_ones = np.full(9664, 0)
    vfile_vocab = open("L.csv")
    read_csv_vocab = csv.reader(vfile_vocab)
    i = 0
    for row in read_csv_vocab:
        for col in row:
            if col > '0':
                num_ones[i] += 1
        i += 1
    return num_ones


# T2 = np.array([[0, 1 / 3, 1 / 3, 1 / 3], [0, 0, 1 / 2, 1 / 2], [1, 0, 0, 0], [1 / 2, 0, 1 / 2, 0]])

T = create_transition_matrix(10, .2, .5)
pi_one_tenth = np.array([1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10])
pi_zero = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# print(np.matmul(pi_zero, create_transition_matrix(10, .2, .5)))
a = [1, 2, 3, 4, 5]
b = [3, 7, 8, 3, 5]
c = np.sum(np.absolute(np.subtract(a, b)))
s = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
B = np.full((9664, 9664), 1 / 9664)
pi_Google = np.full(9664, 1 / 9664)
T2 = np.full((9664, 9664), 0.0)
T2 = build_t2(T2)
alpha = .15
G = (1-alpha)*np.transpose(T2) + alpha*B
pi = compute_pi_google(100, G, pi_Google)
sorted_pi = np.flip(np.sort(pi))
top_25 = []
top_25_elements = []
for i in range(25):
    top_25.append(sorted_pi[i])

p = 1
for n in pi:
    if top_25.__contains__(n):
        print(n, " for website ", p)
        top_25_elements.append(p)
    p += 1
print(top_25_elements)




# a is original b is monte carlo
# print(calculate_delta(compute_pi(100, create_transition_matrix(10, .2, .5), pi_zero), np.divide(calculate_pi_n_i(), 1000)))
# calculate_equl_dist(G, s, 4)
# print(compute_pi(100, create_transition_matrix(10, .9, .1), pi_one_tenth))
