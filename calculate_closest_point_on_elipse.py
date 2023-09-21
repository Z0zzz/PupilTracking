from scipy.optimize import minimize
import pandas as pd
from math import sin, cos, isnan
import matplotlib.pyplot as plt
import warnings
import os
from pupil_tracking_optimization import rotate, homogenous_proj
import torch
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

[9.5153251369193, 5.648705590436374]
original_points=[[8,8],[8,7],[8,6]]

def plot_3d_circle(x,y,z, extra_points=None):
    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the circle
    ax.plot(x, y, z)
    if extra_points is not None:
        ax.plot(extra_points[0], extra_points[1], extra_points[2])

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Circle in 3D')

    plt.show()

def generate_ellipse():
    center = (0, 0)
    width = 4
    height = 2
    angle = 45  # Angle in degrees

    # Generate angles
    theta = np.linspace(0, 2*np.pi, 50)

    # Parametric equations of the ellipse
    x = center[0] + width/2 * np.cos(theta) * np.cos(np.radians(angle)) - height/2 * np.sin(theta) * np.sin(np.radians(angle))
    y = center[1] + width/2 * np.cos(theta) * np.sin(np.radians(angle)) + height/2 * np.sin(theta) * np.cos(np.radians(angle))

    return list(zip(x,y))

def get_data():
    # Specify the path to the CSV file
    csv_file_path = '~/Downloads/Untitled spreadsheet - Sheet1 (1).csv'

    # Read the CSV file using pandas
    df = pd.read_csv(csv_file_path)

    # Convert DataFrame to a NumPy array
    array_data = df.values

    cam1Data = df.values[1:, 5:7]
    cam2Data = df.values[1:, 1:3]
    cam1Data = cam1Data.tolist()
    cam2Data = cam2Data.tolist()
    return cam1Data.T, cam2Data.T


def A(theta, a, b):
    return cos(theta)**2 / a**2 + sin(theta)**2 / b**2

def B(theta, a, b):
    return 2*cos(theta)*sin(theta)*(1/a**2-1/b**2)

def C(theta, a, b):
    return sin(theta)**2 / a**2 + cos(theta)**2/b**2

def a_quadr(theta, a, b):
    return C(theta, a, b)

def b_quadr(theta, a, b, h, k, x):
    return -1*2*(C(theta,a,b) * k + B(theta, a, b) * h) + B(theta, a, b) * x

def c_quadr(theta, a, b, h, k, x):
    return A(theta, a, b) * x**2 - (2 * A(theta, a, b) * h + k * B(theta, a, b)) * x + A(theta, a, b) * h**2 + C(theta, a, b) * k**2 - 1

def y_plus(theta, a, b, h, k, x):
    aprime = a_quadr(theta, a, b)
    bprime = b_quadr(theta, a, b, h, k, x)
    cprime = c_quadr(theta, a, b, h, k, x)
    return (-1 * bprime + (bprime**2 - 4 * aprime * cprime)**0.5) / (2 * aprime)

def y_minus(theta, a, b, h, k, x):
    aprime = a_quadr(theta, a, b)
    bprime = b_quadr(theta, a, b, h, k, x)
    cprime = c_quadr(theta, a, b, h, k, x)
    return (-1 * bprime - (bprime**2 - 4 * aprime * cprime)**0.5) / (2 * aprime)

def objective_function_plus(x):
    global theta, a, b, h, k, original_point
    point = ((y_plus(theta, a, b, h, k ,x) - original_point[1])**2 + (x[0] - original_point[0])**2)**0.5
    return point

def objective_function_minus(x):
    global theta, a, b, h, k, original_point
    point = ((y_minus(theta, a, b, h, k ,x) - original_point[1])**2 + (x[0] - original_point[0])**2)**0.5
    return point

def L2_distance(x,y):
    return ((y[1]-x[1])**2 + (y[0]-x[0])**2)**0.5


def closest_point(original_point):
    # iteratively optimize
    x0 = original_point[0]
    flag = True
    while flag:
        result_plus = minimize(objective_function_plus, x0)
        result_minus = minimize(objective_function_minus, x0)

        xy_plus = [result_plus.x[0], y_plus(theta,a,b,h,k,result_plus.x[0])]
        xy_minus = [result_minus.x[0], y_minus(theta,a,b,h,k,result_minus.x[0])]

        if not isnan(xy_plus[1]) and not isnan(xy_minus[1]):
            if L2_distance(xy_plus, original_point) > L2_distance(xy_minus, original_point):
                solution = xy_minus
            else:
                solution = xy_plus
            flag = False
        
        if not isnan(xy_plus[1]):
            solution = xy_plus
            flag = False
        if not isnan(xy_minus[1]):
            solution = xy_minus
            flag = False
        x0 += 1
    return solution

'''
original_points = generate_ellipse()

solution = []
for point in original_points:
    original_point = point
    solution.append(closest_point(point))
print(solution)
# plotting ellipse on graph
xs = np.linspace(-4.5,10,1000)
ys = y_plus(theta, a, b, h, k, xs)
ys2 = y_minus(theta, a, b, h, k, xs)
plt.plot(xs, ys)
plt.plot(xs, ys2)
# point found on ellipse
for s in solution:
    plt.plot([s[0]], [s[1]], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
# starting point
for s in original_points:
    plt.plot([s[0]], [s[1]], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
plt.show()
'''



# exhaustively optimize
'''
xs = np.linspace(-4.8,4.8,1000)
ys = y_plus(theta, a, b, h, k, xs)
print("calculating points on elipse")
gt_points = list(zip(xs, ys))

test_point = [2,4]
min_loss = float("inf")
for y in gt_points:
    loss = L2_distance(y,test_point)
    if loss < min_loss:
        min_loss = loss
        min_point = y

print("min loss: ", min_loss)
print("min point: ", min_point)
'''


'''

thetax = torch.tensor(-20.)
thetay = torch.tensor(-40.)
thetaz = torch.tensor(10.)
r = torch.tensor(3.)
center = torch.tensor([-211.,-107.,2407.])

rotated_points1 = rotate(thetax, thetay, thetaz, r, center)
r1 = rotated_points1.detach().numpy()

rotated_points2 = rotate(thetax, thetay, thetaz, r, center-0.1)
r2 = rotated_points2.detach().numpy()
# plot_3d_circle(r1[0], r1[1], r1[2], r2)

camMatrix1 = torch.tensor([
        [14563.59278,	0,	1747.722805,	0],
        [0,	14418.55592,	896.3267034,	0],
        [0,	0,	1,	0],
    ])

camMatrix2 = torch.tensor([
        [14911.09787,	-224.3242906,	2180.916566,	-1150110.844],
        [106.4257579,	14620.19877,	795.2746281,	220494.7087],
        [-0.12459133,	-0.04392728697,	0.9912352869,	-50.84174996]
    ])

prediction1a = homogenous_proj(rotated_points1, camMatrix1).detach().numpy()
prediction1b = homogenous_proj(rotated_points1, camMatrix2).detach().numpy()

prediction2a = homogenous_proj(rotated_points2, camMatrix1).detach().numpy()
prediction2b = homogenous_proj(rotated_points2, camMatrix2).detach().numpy()


# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.plot(prediction1a[0], prediction1a[1])
# ax1.plot(prediction2a[0], prediction2a[1])
# ax2.plot(prediction1b[0], prediction1b[1])
# ax2.plot(prediction2b[0], prediction2b[1])
# plt.show()


theta = 1
a = 1
b = 1
h = 475
k = 263
'''
'''
solution = []
for point in original_points:
    original_point = point
    solution.append(closest_point(point))
print(solution)
'''
'''
# plotting ellipse on graph
xs = np.linspace(465,480,1000)
ys = y_plus(theta, a, b, h, k, xs)
ys2 = y_minus(theta, a, b, h, k, xs)
# plt.plot(xs, ys)
# plt.plot(xs, ys2)
# point found on ellipse
'''


'''
for s in solution:
    plt.plot([s[0]], [s[1]], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
# starting point
for s in original_points:
    plt.plot([s[0]], [s[1]], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
'''
'''
# plt.show()

# Ellipse parameters
center = (470, 255)  # Center coordinates
semi_major = 18   # Semi-major axis length
semi_minor = 2.8   # Semi-minor axis length
angle = -50       # Rotation angle in degrees

# Generate points for the ellipse
theta = np.linspace(0, 2*np.pi, 100)
x = center[0] + semi_major * np.cos(theta) * np.cos(np.radians(angle)) - semi_minor * np.sin(theta) * np.sin(np.radians(angle))
y = center[1] + semi_major * np.cos(theta) * np.sin(np.radians(angle)) + semi_minor * np.sin(theta) * np.cos(np.radians(angle))
ax1.plot(x, y)

center = (471, 254)  # Center coordinates
semi_major = 18   # Semi-major axis length
semi_minor = 2.8   # Semi-minor axis length
angle = -50       # Rotation angle in degrees
theta = np.linspace(0, 2*np.pi, 100)
x = center[0] + semi_major * np.cos(theta) * np.cos(np.radians(angle)) - semi_minor * np.sin(theta) * np.sin(np.radians(angle))
y = center[1] + semi_major * np.cos(theta) * np.sin(np.radians(angle)) + semi_minor * np.sin(theta) * np.cos(np.radians(angle))
# ax1.plot(x, y)


a = 18
b = 2.8
h = 470
k = 250
theta = np.radians(angle)

xs = np.linspace(460,482,100)
ys = y_plus(theta, a, b, h, k, xs)
ys2 = y_minus(theta, a, b, h, k, xs)
print("ground truth points: ", list(zip(xs,ys)))
print("#"*20)
# ax1.plot(xs, ys)
# ax1.plot(xs, ys2)

original_points = list(zip(x,y))
print(original_points)
s = [473.85533331932817, 246.6347649858634]
ax1.plot([s[0]], [s[1]], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")

solution = []
original_points = [s]
for point in original_points:
    original_point = point
    solution.append(closest_point(point))
print(solution)

recovered_xs = [item[0] for item in solution]
recovered_ys = [item[1] for item in solution]
'''
