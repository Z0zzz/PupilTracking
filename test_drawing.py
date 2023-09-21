import matplotlib.pyplot as plt
import numpy as np
from test import closest_point_on_elipse

def generate_ellipse(center, width, height, angle, is_radian=False):

    # Generate angles
    theta = np.linspace(0, 2*np.pi, 50)

    if is_radian:
        x = center[0] + width/2 * np.cos(theta) * np.cos(angle) - height/2 * np.sin(theta) * np.sin(angle)
        y = center[1] + width/2 * np.cos(theta) * np.sin(angle) + height/2 * np.sin(theta) * np.cos(angle)
    else:
        # Parametric equations of the ellipse
        x = center[0] + width/2 * np.cos(theta) * np.cos(np.radians(angle)) - height/2 * np.sin(theta) * np.sin(np.radians(angle))
        y = center[1] + width/2 * np.cos(theta) * np.sin(np.radians(angle)) + height/2 * np.sin(theta) * np.cos(np.radians(angle))

    return (x,y)

def euclidean_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

def find_major_minor_axis(points,center):
    min_p = list()
    min_dist = float("inf")
    max_p = list()
    max_dist = float("-inf")

    x = list(points[0].clone().detach().numpy())
    y = list(points[1].clone().detach().numpy())

    iterations = points.shape[1]
    for i in range(iterations):
        p = [x[i], y[i]]
        d = euclidean_distance(p, center)
        if d > max_dist:
            max_dist = i
            max_p = p
        elif d < min_dist:
            min_dist = i
            min_p = p

    return min_dist, min_p, max_dist, max_p

center = [475.0385,252.2438]
width = 40.9194
height = 7.7458
angle = -0.8781

center2 = [417.0743,228.4001]
width2 = 42.6008
height2 = 7.2318
angle2 = -0.8537

center3 = [474.9855, 252.2553]
width3 = 35
height3 = 6.3
angle3 = -0.8781

center4 = [417.1267, 228.3897]
width4 = 48
height4 = 4
angle4 = -0.8537


x1, y1 = generate_ellipse(center=center, width=width, height=height, angle=angle, is_radian=True)
x2, y2 = generate_ellipse(center=center2, width=width2, height=height2, angle=angle2, is_radian=True)
x3, y3 = generate_ellipse(center=center3, width=width3, height=height3, angle=angle3, is_radian=True)
x4, y4 = generate_ellipse(center=center4, width=width4, height=height4, angle=angle4, is_radian=True)


min_p = list()
min_dist = float("inf")
max_p = list()
max_dist = float("-inf")

for p in zip(x3,y3):
    d = euclidean_distance(p, center3)
    if d > max_dist:
        max_dist = d
        max_p = p
    elif d < min_dist:
        min_dist = d
        min_p = p

print(max_dist)
print(min_dist)    


# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))
# ax1.plot(x1,y1,color="red")
# ax2.plot(x2,y2,color="red")
# ax1.plot(x3,y3,color="blue")
# ax2.plot(x4,y4,color="blue")

# x3 = x3[:25]
# y3 = y3[:25]
# closest_points = closest_point_on_elipse(x3,y3, initial_cond=1)
# xs = [l[0] for l in closest_points]
# ys = [l[1] for l in closest_points]
# # ax1.plot(x3,y3, ".", color="red")
# # ax1.plot(xs,ys,".")

# print(x3)
# print(xs)
# for p in range(len(xs)):
#     ax1.plot([x3[p], xs[p].item()], [y3[p], ys[p].item()], linestyle='-', color='b', label='Line')
#     ax1.plot([x3[p]], [y3[p]], ".", color='r', label='Points')
#     ax1.plot([xs[p].item()], [ys[p].item()], ".", color='b', label='Points')

# plt.tight_layout()
# plt.show()
