import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# The given 21x3 array for joints
# joints = np.load('/home/genli/todi_scratch/datasets_aligned/rhand/arctic/s08_phone_use_03_02.npy')[77] 
# joints[:, 0] *= -1
joints = np.array([[-0.04942092,  0.16622986,  1.2349437 ],
       [-0.08351804,  0.08792662,  1.2214261 ],
       [-0.09222279,  0.05743717,  1.2197784 ],
       [-0.09137583,  0.03637879,  1.2164305 ],
       [-0.07570665,  0.07977588,  1.2424043 ],
       [-0.07955125,  0.04945514,  1.2413877 ],
       [-0.07697172,  0.02723947,  1.2411503 ],
       [-0.04086601,  0.09524529,  1.2672898 ],
       [-0.04470586,  0.07842743,  1.2777311 ],
       [-0.04333913,  0.06210365,  1.2853875 ],
       [-0.05776018,  0.08746368,  1.2583191 ],
       [-0.06503601,  0.06088323,  1.260313  ],
       [-0.06733544,  0.03753383,  1.2640668 ],
       [-0.05947516,  0.14345415,  1.2081375 ],
       [-0.0756098 ,  0.12326813,  1.1941079 ],
       [-0.08705054,  0.10067065,  1.1878362 ],
       [-0.09915067,  0.07210089,  1.1733191 ],
       [-0.09576894,  0.01245505,  1.2183691 ],
       [-0.07761945,  0.00207037,  1.2405381 ],
       [-0.06817874,  0.01297362,  1.266314  ],
       [-0.04088224,  0.04271751,  1.290645  ]])
print(np.linalg.norm(joints[18]-joints[0]))
# Define the skeleton (bones) as connections between joints
skeleton = np.array([
    [0, 13, 14, 15, 16], 
    [0, 1, 2, 3, 17], 
    [0, 4, 5, 6, 18], 
    [0, 10, 11, 12, 19], 
    [0, 7, 8, 9, 20]
])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each joint
for i, (x, y, z) in enumerate(joints):
    ax.scatter(x, y, z, color='b')
    ax.text(x, y, z, f'{i}', color='red')  # Label each joint with its index

# Draw bones
for bone in skeleton:
    for start, end in zip(bone[:-1], bone[1:]):
        xs, ys, zs = zip(joints[start], joints[end])
        ax.plot(xs, ys, zs, color='k')  # Draw line segments

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the title
ax.set_title('3D Visualization of Hand Joints with Bones')

# Show the plot
plt.show()
