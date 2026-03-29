import numpy as np
import open3d as o3d

# ========= 不同Room是不同的番茄 =========
room_str = ["room1right1plant12-17", 
            "room4right1plant5-11",
           ]
for room in room_str:

# ========= 1. 读取点云（所有时序） =========
    time_str = ["20250915",
                "20250918",
                "20250922",
                "20250925",
                "20250929",
                ]
    pcds_path = []
    for time in time_str:
        pcds_path.append(f"D:/VS_project/NBV_Simulation_NRICP_Greenhouse/Greenhouse_data/{time}/{room}/pcd_nerfacto/cropped_final_aligned.ply")
    pcds = [ o3d.io.read_point_cloud(pcd_path) for pcd_path in pcds_path ]
    pcd = o3d.geometry.PointCloud()
    for p in pcds:
        pcd += p
    print(pcd)

    points = np.asarray(pcd.points)

    # ========= 2. 只在 x-y 平面做 PCA，得到“行向”的主轴 =========
    xy = points[:, :2]
    xy_mean = xy.mean(axis=0)
    xy_centered = xy - xy_mean

    cov = np.cov(xy_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # 2x2

    # 最大特征值对应的特征向量 -> x-y 平面里的主方向
    idx = np.argsort(eigvals)[::-1]
    e1_xy = eigvecs[:, idx[0]]
    e1_xy = e1_xy / np.linalg.norm(e1_xy)

    # 与 e1 垂直的第二个方向
    e2_xy = np.array([-e1_xy[1], e1_xy[0]])
    e2_xy = e2_xy / np.linalg.norm(e2_xy)

    # ========= 3. 构造 3D 中的三个正交轴：u,v 在 x-y 平面，w 为世界 z 轴 =========
    u = np.array([e1_xy[0], e1_xy[1], 0.0])  # 长轴（水平方向）
    u /= np.linalg.norm(u)

    v = np.array([e2_xy[0], e2_xy[1], 0.0])  # 横向
    v /= np.linalg.norm(v)

    w = np.array([0.0, 0.0, 1.0])            # 竖直方向（世界 z）

    # R 的三列是包围盒局部坐标系在世界坐标中的三个轴向
    R = np.stack([u, v, w], axis=1)          # 3x3

    # ========= 4. 把点投影到这个新坐标系，计算 min/max -> extent & center =========
    Q = points @ R          # N x 3，每一列是沿 u,v,w 方向的坐标

    min_q = Q.min(axis=0)
    max_q = Q.max(axis=0)

    extent = max_q - min_q                 # 每个方向上的长度（不是半长）
    center_local = 0.5 * (min_q + max_q)   # 在 box 局部坐标系里的中心
    center_world = center_local @ R.T      # 转回世界坐标系

    # （可选）给 bbox 留一点 margin，例如 5%：
    margin_ratio = 0.05
    extent = extent * (1.0 + margin_ratio)

    # ========= 5. 构造 Oriented Bounding Box =========
    obb = o3d.geometry.OrientedBoundingBox(center=center_world, R=R, extent=extent)
    obb.color = (0.0, 0.0, 1.0)   # 蓝色 BBX

    # ========= 6. 世界坐标系可视化 =========
    # 尺寸可以按 bbox 对角线的 10% 来设
    diag = np.linalg.norm(extent)
    axis_size = diag * 0.1

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size,
        origin=[0.0, 0.0, 0.0]
    )

    # ========= 7. 显示点云 + 植物 BBX + 世界坐标系 =========
    #o3d.visualization.draw_geometries([pcd, obb, world_frame])

    # 已有对象：点云和作物 OBB
    plant_obb = obb          # 蓝色包围盒
    R = plant_obb.R          # 3x3, [u_axis, v_axis, w_axis]
    center = plant_obb.center
    extent = plant_obb.extent  # [Lu, Lv, Lw]

    u_axis = R[:, 0]  # 行向
    v_axis = R[:, 1]  # 横向（从作物指向走廊中心）
    w_axis = R[:, 2]  # 竖直

    # ==== 1) 真实几何参数（你可以换成测量值） ====
    # 作物中心 -> 走廊中线 的水平距离
    d_crop_to_walk = 0.70     # m，比如 0.m，根据温室结构修改

    # 横向总偏移（从作物中心沿 +v 方向到“轨道 BBX 中线”）
    if room.startswith("room1"):
        side_offset = -1.0 * d_crop_to_walk
    else:
        side_offset = d_crop_to_walk

    cam_centerline = center + side_offset * v_axis

    # ==== 3) 轨道 BBX 尺寸 ====
    scale_u = 1.1    # 行向长度 = 植物长度的 1.1 倍
    scale_w = 1.1    # 竖直高度 = 植物高度的 1.1 倍

    # 横向厚度由“相机在走廊方向的活动区间”决定：几乎一条线就设得很窄 相机左右最多±cm
    rail_half_width = 0.30

    view_extent = np.empty_like(extent)
    view_extent[0] = extent[0] * scale_u       # 行向长度
    view_extent[1] = 2.0 * rail_half_width     # 横向宽度
    view_extent[2] = extent[2] * scale_w       # 竖直高度

    # 如果你希望轨道整体略高/略低于植物中心，可以加一个竖直偏移：
    up_offset = 0.0
    view_center = cam_centerline + up_offset * w_axis

    # ==== 4) 构造红色 OBB ====
    track_obb = o3d.geometry.OrientedBoundingBox(
        center=view_center,
        R=R,
        extent=view_extent
    )
    track_obb.color = (1.0, 0.0, 0.0)

    # ==== 5) 可视化 ====
    # o3d.visualization.draw_geometries([pcd, plant_obb, track_obb, world_frame])
    o3d.visualization.draw_geometries([pcd, plant_obb, track_obb])

    # ==== 6) 输出轨道 OBB 参数 ====
    print(f"Room: {room}")
    print(f"Track OBB Center: {track_obb.center}")
    print(f"Track OBB R (u,v,w axes):\n{track_obb.R}")
    print(f"Track OBB Extent (Lu, Lv, Lw): {track_obb.extent}")
    print("--------------------------------------------------")

    def obb_to_opencv_dict(obb):
        R = np.asarray(obb.R)          # 3x3
        R_flat = R.reshape(-1).tolist()  # 9 elements, row-major

        return {
            "center": obb.center.tolist(),   # [cx, cy, cz]
            "R_flat": R_flat,               # [r00, r01, r02, ..., r22]
            "extent": obb.extent.tolist(),  # [sx, sy, sz]
        }
    
    data = {
        "room_str": room,
        "plant_box": obb_to_opencv_dict(plant_obb),
        "view_box":  obb_to_opencv_dict(track_obb),
    }

    with open( room + "_bbx_config.yaml", "w") as f:
        import yaml
        # 写一个 OpenCV 风格的头，不写也行，只是更“像”OpenCV
        f.write("%YAML:1.0\n")
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)