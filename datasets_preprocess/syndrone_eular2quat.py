import numpy as np

# Given Euler angles (in degrees)
pitch = -30.0  # rotation around x-axis
yaw = 3.4      # rotation around y-axis
roll = 0       # rotation around z-axis

# Convert degrees to radians
pitch = np.radians(pitch)
yaw = np.radians(yaw)
roll = np.radians(roll)

# Calculate the quaternion components
qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)

# Position in the camera coordinate system (no change needed)
x = 223.14
y = 199.44
z = 20.0

# Print the result in the (x, y, z, qx, qy, qz, qw) format
print(f"Position: ({x}, {y}, {z})")
print(f"Quaternion: ({qx}, {qy}, {qz}, {qw})")
