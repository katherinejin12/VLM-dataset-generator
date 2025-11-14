import numpy as np

# Path to one file
path = "/home/deepak/WarehouseResultrobotnewmult/_World_Robots_iw_hub_01_camera_mount_transporter_camera_third_person/motion_vectors/motion_vectors_00000.npy"

# Load numpy array
arr = np.load(path)

print("Shape:", arr.shape)
print("Dtype:", arr.dtype)
print("First few elements:\n", arr[:5])

