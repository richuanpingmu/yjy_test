import torch
import numpy as np

pc_path = 'human_pc/human_pc_filtered_lzy_1.npy'
# 1. Load and sample point cloud
human_points = torch.from_numpy(np.load(pc_path)).float()
human_points = human_points / 1000
point_num = 1024

rot_A_to_B = np.array([
                    [0, 0, -1],   # x_B = z_A（原转换为x_B = -z_A，此处修正符号）
                    [1, 0, 0],   # y轴保持不变
                    [0, -1, 0]   # z_B = -x_A（原转换为z_B = x_A，此处修正符号）
                ], dtype=np.float32)
human_points = human_points @ rot_A_to_B.T

    # 可视化原始点云
try:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(human_points[:,0], human_points[:,1], human_points[:,2], s=1)
    plt.title("Original Point Cloud")
    plt.show()
    plt.savefig("original_pc.png")
    #plt.close()
except Exception as e:
    print(f"[可视化原始点云失败]{e}")

# 打乱点云顺序
perm = torch.randperm(human_points.shape[0])
human_points = human_points[perm]

'''
now_pt_num = int(human_points.shape[0])
    # 使用最远点采样（FPS）
try:
    from torch_cluster import fps
    ratio = min(1.0, point_num / now_pt_num)
    idx = fps(human_points, torch.arange(now_pt_num), ratio=ratio, random_start=False)
    if idx.shape[0] > point_num:
        idx = idx[:point_num]
    human_points_sampled = human_points[idx]
    if human_points_sampled.shape[0] < point_num:
        extra_idx = np.random.choice(human_points_sampled.shape[0], point_num - human_points_sampled.shape[0], replace=True)
        human_points_sampled = torch.cat([human_points_sampled, human_points_sampled[extra_idx]], dim=0)
except ImportError:
    raise ImportError("请先安装 torch-cluster 以支持最远点采样 (FPS)：pip install torch-cluster")

    # 可视化采样后点云
try:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(human_points_sampled[:,0], human_points_sampled[:,1], human_points_sampled[:,2], s=5, c='r')
    plt.title("FPS Sampled Point Cloud")
    plt.show()
    plt.savefig("fps_sampled_pc.png")
    plt.close()
except Exception as e:
    print(f"[可视化采样点云失败]{e}")

'''
# 优先采样边缘点：计算每个点到其k近邻的平均距离，距离大的点更可能在边缘
now_pt_num = int(human_points.shape[0])
k = min(16, now_pt_num-1)  # 近邻数
# 计算距离矩阵
with torch.no_grad():
    dist_mat = torch.cdist(human_points, human_points, p=2)  # (N, N)
    knn_dist, _ = torch.topk(dist_mat, k=k+1, largest=False)  # (N, k+1)，包含自身
    mean_knn_dist = knn_dist[:, 1:].mean(dim=1)  # 排除自身，取均值
    # 按平均距离排序，取前point_num个
    edge_idx = torch.topk(mean_knn_dist, k=point_num, largest=True).indices
    human_points_sampled = human_points[edge_idx]
    if human_points_sampled.shape[0] < point_num:
        extra_idx = torch.randint(0, human_points_sampled.shape[0], (point_num - human_points_sampled.shape[0],))
        human_points_sampled = torch.cat([human_points_sampled, human_points_sampled[extra_idx]], dim=0)

    # 可视化采样后点云
try:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(human_points_sampled[:,0], human_points_sampled[:,1], human_points_sampled[:,2], s=5, c='r')
    plt.title("FPS Sampled Point Cloud")
    plt.show()
    plt.savefig("fps_sampled_pc.png")
    #plt.close()
except Exception as e:
    print(f"[可视化采样点云失败]{e}")
