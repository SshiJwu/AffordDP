import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import open3d as o3d
import fpsample
from third_party.Point_SAM.demo.utils import preprocess_point_cloud, execute_global_registration, execute_fast_global_registration, refine_registration, decompose_transformation

def ICP_register(part1, part2, voxel_size=0.05, init="ransac"):
    """
    Use open3d multi-scale ICP
    """
    # # reformate
    # part1 = np.array(part1.detach().cpu())
    # part2 = np.array(part2.detach().cpu())
    
    # source
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(part1)
    pcd1.estimate_normals()
    # target 
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(part2)
    pcd2.estimate_normals()
    
    
    # global registration for init 
    source_down, source_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd2, voxel_size)
    if init.lower() == "fast": 
        # use fast global registration: 
        # -reference: https://link.springer.com/content/pdf/10.1007/978-3-319-46475-6_47.pdf
        result_init = execute_fast_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    elif init.lower() == "ransac": 
        # use RANSAC
        result_init = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    elif init.lower() == "fgr":
        
        pass
    else:
        result_init = np.eye(4)
    if isinstance(result_init, np.ndarray) is False:
        result_init = result_init.transformation
    # local registration for refinement (ICP)
    result_icp = refine_registration(pcd1, pcd2, source_fpfh, target_fpfh,
                                 voxel_size, result_init)
    
    # calculate R and t
    T = result_icp.transformation
    R, t = decompose_transformation(T)
    
    return T, R, t


def get_image_pixel_from_3d(point, view_matrix, projection_matrix, camera_width, camera_height):
    
    ones_column = np.ones((point.shape[0], 1))
    world_point = np.hstack((point, ones_column))

    point_camera = world_point@view_matrix
    point_camera = point_camera/point_camera[:,3].reshape(-1,1)
    x = point_camera[:,0]
    y = point_camera[:,1]
    z = point_camera[:,2]

    centerU = camera_width/2
    centerV = camera_height/2

    fu = 2/projection_matrix[0, 0]
    fv = 2/projection_matrix[1, 1]

    u = -x*camera_width/(z*fu) + centerU
    v = y*camera_height/(z*fv) + centerV

    u = u.reshape(-1,1)
    v = v.reshape(-1,1)
    image_pixel= np.hstack((u, v))

    return image_pixel


def get_3d_from_image_pixel(image_pixel, pixel_depth, view_matrix, projection_matrix, camera_width, camera_height):

    view_inv = np.linalg.inv(view_matrix)
    u = image_pixel[:,0].reshape(-1,1)
    v = image_pixel[:,1].reshape(-1,1)

    centerU = camera_width/2
    centerV = camera_height/2

    fu = 2/projection_matrix[0, 0]
    fv = 2/projection_matrix[1, 1]

    Z = pixel_depth.reshape(-1,1)
    X = -(u-centerU)/camera_width * Z * fu
    Y = (v-centerV)/camera_height * Z * fv

    position = np.hstack((X, Y, Z, np.ones_like(X)))

    position[:,0:4] = position[:,0:4]@view_inv

    return position[:,:3]

def find_nearest_object_pixel_in_box(mask_array, query_point, box_size=50):

    if query_point.ndim != 1:
        query_point = query_point.reshape(-1)
    mask = mask_array
    
    top = max(query_point[1] - box_size // 2, 0)
    bottom = min(query_point[1] + box_size // 2, mask.shape[0])
    left = max(query_point[0] - box_size // 2, 0)
    right = min(query_point[0] + box_size // 2, mask.shape[1])

    box_mask = mask[top:bottom, left:right]

    # object_pixels = box_mask == 1

    object_coords = np.argwhere(box_mask)

    object_coords[:, 0] += top
    object_coords[:, 1] += left
    
    nearest_pixel = None
    min_distance = float('inf')
    
    for y, x in object_coords:
        distance = np.sqrt((x - query_point[0]) ** 2 + (y - query_point[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_pixel = (x, y)

    if nearest_pixel == None:
        nearest_pixel = query_point
    
    return np.array(nearest_pixel).reshape(-1,2)

def sample_point_cloud(point, num_point):

    samples_idx = fpsample.bucket_fps_kdline_sampling(point[:,:3], num_point,h=7)
    point = point[samples_idx]

    return point

def update_afford(new_position, afford, Rotation):

    if new_position.ndim != 1:
        new_position = new_position.reshape(-1)

    delta = new_position - afford[0]
    afford = afford@Rotation.T + delta

    return afford

def calculate_variance_and_mean(lst):
    n = len(lst)
    mean = sum(lst) / n
    variance = sum((x - mean) ** 2 for x in lst) / n
    return variance, mean