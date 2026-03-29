import numpy as np
import open3d as o3d
import yaml
import os

# ==== 1. 读 YAML 里的 OBB ====
# room_str = "room1right1plant12-17"
# yaml_file_path = room_str + "_bbx_config.yaml"
# pcd = o3d.io.read_point_cloud("D:/VS_project/NBV_Simulation_NRICP_Greenhouse/Greenhouse_data/20250915/room1right1plant12-17/pcd_nerfacto/cropped_final_aligned.ply")

room_str = "room4right1plant5-11"
yaml_file_path = room_str + "_bbx_config.yaml"
pcd = o3d.io.read_point_cloud("D:/VS_project/NBV_Simulation_NRICP_Greenhouse/Greenhouse_data/20250915/room4right1plant5-11/pcd_nerfacto/cropped_final_aligned.ply")

with open(yaml_file_path, "r") as f:
    first_line = f.readline()
    if first_line.startswith("%YAML"):
        # 跳过 directive，从第二行开始给 PyYAML
        cfg = yaml.safe_load(f)
    else:
        f.seek(0)
        cfg = yaml.safe_load(f)

def load_box(node):
    c = np.array(node["center"], dtype=float)
    e = np.array(node["extent"], dtype=float)
    R_flat = np.array(node["R_flat"], dtype=float)
    R = R_flat.reshape(3, 3)  # row-major
    return c, R, e

c_p, R, e_p = load_box(cfg["plant_box"])
c_v, _,  e_v = load_box(cfg["view_box"])

u_axis = R[:, 0]
v_axis = R[:, 1]
w_axis = R[:, 2]

if yaml_file_path.startswith("room4"):
    # room4 的相机朝向和 room1 不同，需要把 v 轴反过来
    v_axis = -v_axis

# ========= 2. 三个相机高度（竖直间隔 0.664 m） =========
spacing = 0.664  # 三行间距 664 mm

# 以 plant_obb 的中心高度作为“中间那台相机”的 look-at 高度
z_mid_local = np.dot(w_axis, c_p - c_v)  # plant 中心在 view_box 局部 w 坐标

z_offset = -0.1  # 如果想让中间相机看得更高/更低，可以调这个值

z_list_local = [
    z_mid_local - spacing + z_offset,   # 下
    z_mid_local + z_offset,             # 中
    z_mid_local + spacing + z_offset    # 上
]

# 相机在轨道一侧面：v_pos 放在 view_box 外侧边
v_pos = -e_v[1] / 2.0        # 你之前用的 -0.3 就是这个

def cam_pos(u_local, v_local, w_local):
    return c_v + u_axis*u_local + v_axis*v_local + w_axis*w_local

num_u = 5  # 取中间一条 u 位置，在 u 上再离散多个点，比如 4、5、6 都行

L_u = e_p[0]            # plant_obb 在 u 方向的总长度
half_L = L_u / 2.0

u_move_in_offset = 0.5  # 相机采样位置稍微往里挪一点

# 在 [-half_L + offset, half_L - offset] 范围内均匀采样 num_u 个位置
u_list = np.linspace(-half_L + u_move_in_offset, half_L - u_move_in_offset, num_u)

cam_positions = []
look_at_targets = [] # 每个相机看向“同高度的植物中心”
for u_local in u_list:
    for z_local in z_list_local:   # 3 个高度
        cam_positions.append(cam_pos(u_local, v_pos, z_local))
        look_at_targets.append(cam_positions[-1] - v_pos * v_axis)  # 看向植物中心

# ========= 2.5 把视点和 look-at dump 到 txt =========
# 输出文件名：和 yaml 同名前缀，加个后缀
base_name = os.path.splitext(os.path.basename(yaml_file_path))[0]
out_txt = room_str + "_initial_views.txt"

# ========= 3. 相机内参（color 相机） =========
W, H = 1280, 720
fx = 9.1560668945312500e+02
fy = 9.1332666015625000e+02
cx = 6.4714532470703125e+02
cy = 3.7251531982421875e+02

corner_pixels = [
    (0, 0),
    (W-1, 0),
    (0, H-1),
    (W-1, H-1),
]

def pixel_to_dir_cam(u, v):
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0
    v = np.array([x, y, z], dtype=float)
    return v / np.linalg.norm(v)

# ========= 4. look-at + 90° roll（竖起来） =========
def look_at(cam, target, up_world=np.array([0, 0, 1.0])):
    """
    返回不带 roll 的 R_wc：camera -> world
    相机坐标系：x 右, y 下, z 前
    """
    f = (target - cam)
    f = f / np.linalg.norm(f)
    r = np.cross(f, up_world)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    R_wc = np.stack([r, u, f], axis=1)  # 列为 x_cam, y_cam, z_cam 在 world 中的方向
    return R_wc

def add_roll(R_wc, roll_deg):
    """
    在相机自身坐标系下，绕 z_cam 轴转 roll_deg（单位：度），
    然后返回新的 R_wc。
    """
    theta = np.deg2rad(roll_deg)
    c, s = np.cos(theta), np.sin(theta)
    # 相机坐标系下绕 z 轴的旋转矩阵
    R_roll_cam = np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1],
    ])
    # camera -> world 的总旋转：先在相机系里 roll，再映射到 world
    return R_wc @ R_roll_cam

# ========= 5. 构造 FOV 线集（3 个相机） =========
def make_fov_lines(cam_pos, R_wc, color):
    t_far = 1.7  # 可视化射线长度（米）
    points = [cam_pos]
    for (u, v) in corner_pixels:
        d_cam = pixel_to_dir_cam(u, v)
        d_world = R_wc @ d_cam
        points.append(cam_pos + d_world * t_far)

    points = o3d.utility.Vector3dVector(points)
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 4], [4, 3], [3, 1],
    ]
    line_set = o3d.geometry.LineSet(
        points=points,
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(
        [color for _ in lines]
    )
    return line_set

colors = [[0, 1, 0] for _ in cam_positions] # 绿色 FOV 线
fov_lines = []
for cam, tgt, col in zip(cam_positions, look_at_targets, colors):
    R_wc0 = look_at(cam, tgt)
    # D435 竖着装：根据方向选 90 或 -90，如果看起来反了就把 90 改成 -90 试一下
    R_wc = add_roll(R_wc0, roll_deg=90.0)
    fov_lines.append(make_fov_lines(cam, R_wc, col))


# ========= 6. 可视化 =========

plant_obb_vis = o3d.geometry.OrientedBoundingBox(c_p, R, e_p)
plant_obb_vis.color = (0, 0, 1)

view_obb_vis = o3d.geometry.OrientedBoundingBox(c_v, R, e_v)
view_obb_vis.color = (1, 0, 0)

world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

geoms = [pcd, plant_obb_vis, view_obb_vis,
         # world_frame
         ] + fov_lines
#o3d.visualization.draw_geometries(geoms)

# ========= 7. 第二阶段：在 view_box 中做 3D 蓝噪声采样 =========

def sample_poisson_in_obb_fixed(c_center, R_obb, extent, 
                                min_dist, 
                                max_attempts=100000,
                                rng=None):
    """
    固定最小间距 min_dist 的 3D Poisson 采样。
    返回: (N,3) world 坐标的 candidate xyz
    """
    if rng is None:
        rng = np.random.default_rng(0)

    half_ext = extent / 2.0
    pts_local = []

    for _ in range(max_attempts):
        cand = (rng.random(3) * 2.0 - 1.0) * half_ext  # [-half_ext, half_ext]

        ok = True
        for p in pts_local:
            if np.linalg.norm(cand - p) < min_dist:
                ok = False
                break
        if not ok:
            continue

        pts_local.append(cand)

    pts_local = np.array(pts_local)
    pts_world = c_center + (R_obb @ pts_local.T).T
    return pts_world


def sample_poisson_in_obb_auto(c_center, R_obb, extent,
                               num_samples=80,
                               r_init=None,
                               r_min=0.05,
                               shrink=0.05,
                               max_attempts=100000,
                               rng=None):
    """
    在 OBB 内做 3D Poisson 采样，自适应调整 min_dist：
    - 从 r_init 开始，如果采样数 < num_samples，就把 r -= shrink 继续尝试
    - 直到采样数 >= num_samples 或 r < r_min
    返回:
        pts_world: (M,3)，M >= num_samples 时会随机下采样到 num_samples
        final_r: 实际使用的 min_dist
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # 如果没给初始半径，用体积 + 目标数估一个
    if r_init is None:
        V = float(extent[0] * extent[1] * extent[2])
        packing = 0.3  # 粗略打包率
        r_init = ((packing * V) / (num_samples * 4.0/3.0 * np.pi)) ** (1.0/3.0)
        r_init *= 1.3  # 稍微放大一点，从“偏保守”的大半径开始

    r = r_init
    best_pts = None

    while r >= r_min:
        pts = sample_poisson_in_obb_fixed(
            c_center, R_obb, extent,
            min_dist=r,
            max_attempts=max_attempts,
            rng=rng
        )

        print (f"Trying r = {r:.3f}, got {len(pts)} points")

        # 如果已经足够多，就可以停了
        if len(pts) >= num_samples:
            best_pts = pts
            break

        # 不够多，缩小半径再试
        best_pts = pts  # 记一下目前为止“最多”的那次
        r -= shrink

    if best_pts is None or len(best_pts) == 0:
        return np.zeros((0, 3)), r

    # 如果最终点数还是比目标少，就全用；多的话随机下采样到 num_samples
    if len(best_pts) > num_samples:
        idx = rng.choice(len(best_pts), size=num_samples, replace=False)
        best_pts = best_pts[idx]

    return best_pts, r  

# 设定 candidate 参数
num_samples = 100         # 想要多少个候选 xyz，可以自己调
min_dist_init = 0.40      # 任意两点最小间距（这是一个初始值，应该要设置为采样不到nbv_num_samples的数量，大值更合适，但是我已经测试了接近最小的最小间距;0.26 205;0.24 238;0.22 301）

# 把 view_box 沿 v 方向切成前半部分
rng = np.random.default_rng(42)
points, used_r = sample_poisson_in_obb_auto(
    c_center=c_p,
    R_obb=R,
    extent=e_p,
    num_samples=num_samples,
    r_init=min_dist_init,          # None的话就会自动估一个初值
    r_min=0.05,           # 最小允许半径
    shrink=0.02,          # 每次缩 0.02 米
    max_attempts=100000,
    rng=rng
)

# 变成 Open3D 点云，用紫色显示
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([1.0, 0.0, 1.0])  # 紫色

# 可视化：基础场景 + 第二阶段 candidate
geoms_nbv = [pcd, plant_obb_vis, view_obb_vis, pcd,
             # world_frame
             ]
o3d.visualization.draw_geometries(geoms_nbv)

# ========= 把观察点 dump 到 txt =========
# 输出文件名：和 yaml 同名前缀，加个后缀
base_name = os.path.splitext(os.path.basename(yaml_file_path))[0]
out_txt = room_str + "_rand_look_at_pts.txt"

with open(out_txt, "w") as f_out:
    for pts in pcd.points:
        # pts 是 numpy 数组，拼成一行 3 个数
        line_vals = list(pts)
        f_out.write("{:.6f} {:.6f} {:.6f}\n".format(*line_vals))

print(f"Dumped {len(pcd.points)} _rand_look_at_pts to {out_txt}")
