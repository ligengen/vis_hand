import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load the joint data from the npy file
# joints_data = np.load('/home/genli/todi_scratch/datasets_aligned/rhand/m--20230317--1130--QZX685--pilot--ProjectGoliath--Hands--two-hands-DOU_two_hands_knuckle_cracking_25_02.npy')[:60]  # Replace with your actual file path
# joints_data2 = np.load('/home/genli/Desktop/recon-1.npy')
# joints_data = np.concatenate((joints_data, joints_data2))
# Define the skeleton structure
joints_data = np.load('/home/genli/clariden_scratch/eval_result/hot3d_hand/P0001_f71fc9b1_3-0.npz_tok_lhand.npy')
skeleton = np.array([
    [0, 13, 14, 15, 16], 
    [0, 1, 2, 3, 17], 
    [0, 4, 5, 6, 18], 
    [0, 10, 11, 12, 19], 
    [0, 7, 8, 9, 20]
])

# Calculate axis limits
x_min, x_max = np.min(joints_data[:, :, 0]), np.max(joints_data[:, :, 0])
y_min, y_max = np.min(joints_data[:, :, 1]), np.max(joints_data[:, :, 1])
z_min, z_max = np.min(joints_data[:, :, 2]), np.max(joints_data[:, :, 2])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the plot elements
scatter = ax.scatter([], [], [], color='b')
lines = [ax.plot([], [], [], color='k')[0] for _ in range(skeleton.shape[0])]

def init():
    """Initialize the plot."""
    scatter._offsets3d = ([], [], [])
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return [scatter] + lines

def update(frame):
    """Update the plot for each frame."""
    joints = joints_data[frame]
    xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]
    
    # Update scatter plot
    scatter._offsets3d = (xs, ys, zs)
    
    # Update lines for bones
    for i, bone in enumerate(skeleton):
        start_end_points = joints[bone]
        xs_line, ys_line, zs_line = start_end_points[:, 0], start_end_points[:, 1], start_end_points[:, 2]
        lines[i].set_data(xs_line, ys_line)
        lines[i].set_3d_properties(zs_line)
    
    return [scatter] + lines

# Set the axis limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Set labels and title, with Z as the up direction
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of Hand Joints with Bones over Time (Z-Up)')
ax.view_init(elev=90, azim=0)
# Create an animation
ani = FuncAnimation(fig, update, frames=range(joints_data.shape[0]), init_func=init, blit=True, interval=33)

# Show the plot
plt.show()
