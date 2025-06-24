import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import open3d as o3d
import fpsample
from scipy.linalg import polar
import h5py
from plyfile import PlyData, PlyElement


def preprocess_point_cloud(pcd, voxel_size=0.05):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_init):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_init,
        # source, target, distance_threshold, result_init.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def decompose_transformation(T):
    np.set_printoptions(precision=3, suppress=True)
    T = np.array(T)
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    return R, t


def load_ply(filename):
    with open(filename, "r") as rf:
        while True:
            try:
                line = rf.readline()
            except:
                raise NotImplementedError
            if "end_header" in line:
                break
            if "element vertex" in line:
                arr = line.split()
                num_of_points = int(arr[2])

        # print("%d points in ply file" %num_of_points)
        points = np.zeros([num_of_points, 6])
        for i in range(points.shape[0]):
            point = rf.readline().split()
            assert len(point) == 6
            points[i][0] = float(point[0])
            points[i][1] = float(point[1])
            points[i][2] = float(point[2])
            points[i][3] = float(point[3])
            points[i][4] = float(point[4])
            points[i][5] = float(point[5])
    rf.close()
    del rf
    return points

def sample_point_cloud(point, num_points):

    samples_idx = fpsample.bucket_fps_kdline_sampling(point[:,:3], num_points, h=7)
    point = point[samples_idx]

    return point


def draw(pcd, name, values=None, contact_point=None):
    # Define each point as a tuple: necessary for the PlyElement.
    if values is None:
        vertex_list = [
                (pcd[i][0], pcd[i][1], pcd[i][2], 255, 255, 255)
                for i in range(pcd.shape[0])
            ]
        if contact_point is not None:
            vertex_list.append((*contact_point, 255, 0, 255))
        vertex = np.array(vertex_list, dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ])
    elif len(values.shape) >= 2:
        vertex = np.array(
            [
                (pcd[i][0], pcd[i][1], pcd[i][2], int(values[i][0]), int(values[i][1]), int(values[i][2]))
                for i in range(pcd.shape[0])
            ],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
    else:
        colors = np.zeros((values.shape[0], 3))
        
        if max(values) != 0:
            values = values / max(values)
        
        for i in range(colors.shape[0]):
            v = values[i]
            if v == 0:
                colors[i] = [255, 255, 255]
            else:
                colors[i] = [int(v * 255), 0, int((1 - v) * 255)]
                
        vertex_list = [
                (pcd[i][0], pcd[i][1], pcd[i][2], *(colors[i]))
                for i in range(pcd.shape[0])
            ]
        if contact_point is not None:
            vertex_list.append((*contact_point, 255, 0, 255))
        vertex = np.array(vertex_list, dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
    
    breakpoint()
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=True).write(f"vis/{name}.ply")

def rgbd_to_point_cloud(rgb_image_np, depth_image_np, width, height, fx, fy, cx, cy):
    # Load RGB and Depth images
    # For this example, let's assume you have RGB and Depth images as numpy arrays
    # rgb_image_np = <your RGB image as numpy array>
    # depth_image_np = <your Depth image as numpy array>

    # Convert numpy arrays to Open3D images
    rgb_image_o3d = o3d.geometry.Image(rgb_image_np)
    depth_image_o3d = o3d.geometry.Image(depth_image_np)

    # Create an RGBD image from the RGB and Depth images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image_o3d, depth_image_o3d
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy
    )

    # Convert the RGBD image to a point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Optionally visualize the point cloud
    # o3d.visualization.draw_geometries([point_cloud])

    return np.array(point_cloud.points), np.array(point_cloud.colors)


def rotateMatrixToEulerAnglesInRadian(RM):
    theta_z = np.arctan2(RM[1, 0], RM[0, 0])
    theta_y = np.arctan2(-1 * RM[2, 0], np.sqrt(RM[2, 1] * RM[2, 1] + RM[2, 2] * RM[2, 2]))
    theta_x = np.arctan2(RM[2, 1], RM[2, 2])
    print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}\ntheta_z: {theta_z}")
    return theta_x, theta_y, theta_z

def rotateMatrixToEulerAnglesInDegree(RM):
    theta_z = np.arctan2(RM[1, 0], RM[0, 0]) / np.pi * 180
    theta_y = np.arctan2(-1 * RM[2, 0], np.sqrt(RM[2, 1] * RM[2, 1] + RM[2, 2] * RM[2, 2])) / np.pi * 180
    theta_x = np.arctan2(RM[2, 1], RM[2, 2]) / np.pi * 180
    print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}\ntheta_z: {theta_z}")
    return theta_x, theta_y, theta_z

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
