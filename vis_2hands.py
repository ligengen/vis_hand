import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def canonicalize(seq):
    first_jts = seq[0]
    wrist = first_jts[0]
    middle_mcp = first_jts[4]
    y_axis = middle_mcp - wrist
    y_axis[2] = 0  # Flatten
    if np.linalg.norm(y_axis) < 1e-6:
        pdb.set_trace()
    y_axis = y_axis / np.linalg.norm(y_axis)

    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    rot_canonical_2_world = np.stack([x_axis, y_axis, z_axis], axis=1)

    translated = seq - wrist
    canonical_seq = translated @ rot_canonical_2_world

    return canonical_seq

# Load the joint data for both hands from the npy files
left_hand_data = np.load('/home/genli/Desktop/arctic/hot3d_hand/P0001_a68492d5_21-1.npz_tok_lhand.npy') # Replace with your actual file path
right_hand_data = canonicalize(np.load('/home/genli/Desktop/arctic/P0001_a68492d5_21.npz')['lhand'][60:]) # Replace with your actual file path


# Define the skeleton structure for both hands
skeleton = np.array([
    [0, 13, 14, 15, 16], 
    [0, 1, 2, 3, 17], 
    [0, 4, 5, 6, 18], 
    [0, 10, 11, 12, 19], 
    [0, 7, 8, 9, 20]
])

# Calculate axis limits for both hands
all_joints_data = np.concatenate((left_hand_data, right_hand_data), axis=1)
x_min, x_max = np.min(all_joints_data[:, :, 0]), np.max(all_joints_data[:, :, 0])
y_min, y_max = np.min(all_joints_data[:, :, 1]), np.max(all_joints_data[:, :, 1])
z_min, z_max = np.min(all_joints_data[:, :, 2]), np.max(all_joints_data[:, :, 2])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the plot elements for both hands
scatter_left = ax.scatter([], [], [], color='b', label='pred')
scatter_right = ax.scatter([], [], [], color='r', label='gt')
lines_left = [ax.plot([], [], [], color='b')[0] for _ in range(skeleton.shape[0])]
lines_right = [ax.plot([], [], [], color='r')[0] for _ in range(skeleton.shape[0])]

def init():
    """Initialize the plot."""
    scatter_left._offsets3d = ([], [], [])
    scatter_right._offsets3d = ([], [], [])
    for line in lines_left + lines_right:
        line.set_data([], [])
        line.set_3d_properties([])
    return [scatter_left, scatter_right] + lines_left + lines_right

def update(frame):
    """Update the plot for each frame."""
    joints_left = left_hand_data[frame]
    joints_right = right_hand_data[frame]

    # Update scatter plots
    scatter_left._offsets3d = (joints_left[:, 0], joints_left[:, 1], joints_left[:, 2])
    scatter_right._offsets3d = (joints_right[:, 0], joints_right[:, 1], joints_right[:, 2])

    # Update lines for bones for the left hand
    for i, bone in enumerate(skeleton):
        start_end_points_left = joints_left[bone]
        xs_line_left, ys_line_left, zs_line_left = start_end_points_left[:, 0], start_end_points_left[:, 1], start_end_points_left[:, 2]
        lines_left[i].set_data(xs_line_left, ys_line_left)
        lines_left[i].set_3d_properties(zs_line_left)

    # Update lines for bones for the right hand
    for i, bone in enumerate(skeleton):
        start_end_points_right = joints_right[bone]
        xs_line_right, ys_line_right, zs_line_right = start_end_points_right[:, 0], start_end_points_right[:, 1], start_end_points_right[:, 2]
        lines_right[i].set_data(xs_line_right, ys_line_right)
        lines_right[i].set_3d_properties(zs_line_right)

    return [scatter_left, scatter_right] + lines_left + lines_right

# Set the axis limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Set labels and title, with Z as the up direction
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of Both Hands Joints with Bones over Time (Z-Up)')
ax.view_init(elev=90, azim=0)
ax.legend()

# Create an animation
ani = FuncAnimation(fig, update, frames=range(left_hand_data.shape[0]), init_func=init, blit=True, interval=33)

# Show the plot
plt.show()
