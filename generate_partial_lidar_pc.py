import argparse
import os
import sys

import numpy as np
import torch

from parallel_lidar_simulator import LidarSimulator, torch_calc_laser_direction


def to_tensor(array, device, dtype=torch.float32):
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)
    return torch.tensor(array, device=device, dtype=dtype)


def load_array(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        return {key: data[key] for key in data.files}
    if data.shape == ():
        item = data.item()
        if isinstance(item, dict):
            return item
    return data


def normalize_vertices_shape(vertices):
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.shape == (6890, 3):
        vertices = vertices[None]
    if vertices.ndim != 3 or vertices.shape[1:] != (6890, 3):
        raise ValueError(f"vertices should have shape (6890, 3) or (B, 6890, 3), got {vertices.shape}")
    return vertices


def vertices_from_smpl_params(param_file, device, gender="neutral"):
    from model.smpl.pytorch.smpl import SMPL

    data = load_array(param_file)
    if not isinstance(data, dict):
        raise ValueError("SMPL parameter file should be a dict/npz containing pose, shape/betas, and optional trans.")

    pose = data.get("pose", data.get("pose_param", data.get("theta")))
    betas = data.get("betas", data.get("shape", data.get("shape_param")))
    trans = data.get("trans", data.get("trans_param", data.get("translation")))

    if pose is None or betas is None:
        raise KeyError("Cannot find pose and betas/shape in parameter file.")

    pose = np.asarray(pose, dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32)
    if pose.ndim == 1:
        pose = pose[None]
    if betas.ndim == 1:
        betas = betas[None]

    batch_size = pose.shape[0]
    if trans is None:
        trans = np.zeros((batch_size, 3), dtype=np.float32)
    else:
        trans = np.asarray(trans, dtype=np.float32)
        if trans.ndim == 1:
            trans = trans[None]

    smpl = SMPL(gender=gender).to(device)
    with torch.no_grad():
        vertices, _ = smpl(
            to_tensor(pose, device),
            to_tensor(betas, device),
            to_tensor(trans, device),
        )
    return vertices


def farthest_point_sample(points, target_num):
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]
    if n == 0:
        raise ValueError("No valid lidar points were generated.")
    if n >= target_num:
        selected = np.zeros(target_num, dtype=np.int64)
        distances = np.full(n, np.inf, dtype=np.float32)
        farthest = 0
        for i in range(target_num):
            selected[i] = farthest
            centroid = points[farthest:farthest + 1]
            dist = np.sum((points - centroid) ** 2, axis=1)
            distances = np.minimum(distances, dist)
            farthest = int(np.argmax(distances))
        return points[selected]

    extra_idx = np.random.choice(n, target_num - n, replace=True)
    return np.concatenate([points, points[extra_idx]], axis=0)


def build_laser_directions(simulator, lidar_pos, vertices, point_num):
    rel_mesh = vertices - lidar_pos[:, None, :]
    laser_phi, laser_theta, pc_num = simulator.get_selected_laser_direction(rel_mesh)
    directions = torch_calc_laser_direction(None, laser_theta, laser_phi).float()

    split_dirs = directions.split(pc_num.tolist())
    padded = []
    for dirs in split_dirs:
        if dirs.shape[0] >= point_num:
            padded.append(dirs[:point_num])
        else:
            repeat_idx = torch.randint(
                low=0,
                high=dirs.shape[0],
                size=(point_num - dirs.shape[0],),
                device=dirs.device,
            )
            padded.append(torch.cat([dirs, dirs[repeat_idx]], dim=0))

    return torch.stack(padded, dim=0)


def generate_partial_lidar_pc(vertices, lidar_pos, point_num=1024, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("parallel_lidar_simulator.py uses .cuda(), but CUDA is not available.")

    vertices = np.asarray(vertices, dtype=np.float32)
    # 标准化顶点形状为 (B, 6890, 3)
    if vertices.shape == (6890, 3):
        vertices = vertices[None]  # 变成 (1, 6890, 3)
    elif vertices.ndim == 2:
        raise ValueError(f"vertices should have shape (6890, 3) or (B, 6890, 3), got {vertices.shape}")

    vertices = to_tensor(vertices, device)
    lidar_pos = to_tensor(lidar_pos, device)
    if lidar_pos.ndim == 1:
        lidar_pos = lidar_pos[None]
    if lidar_pos.shape[0] == 1 and vertices.shape[0] > 1:
        lidar_pos = lidar_pos.repeat(vertices.shape[0], 1)

    simulator = LidarSimulator().to(device)
    laser_direction = build_laser_directions(simulator, lidar_pos, vertices, point_num)
    pc_num = torch.full((vertices.shape[0],), point_num, dtype=torch.int32, device=device)

    with torch.no_grad():
        partial_pc, _ = simulator(lidar_pos, vertices, pc_num, laser_direction)

    partial_pc = partial_pc.reshape(vertices.shape[0], point_num, 3)
    partial_pc = partial_pc.detach().cpu().numpy()

    fixed = []
    for pc in partial_pc:
        valid_mask = np.linalg.norm(pc, axis=1) > 1e-8
        fixed.append(farthest_point_sample(pc[valid_mask], point_num))
    return np.stack(fixed, axis=0)


def parse_lidar_pos(text):
    values = [float(item) for item in text.split(",")]
    if len(values) != 3:
        raise ValueError("--lidar_pos should look like x,y,z, for example 1,-8,0")
    return np.asarray(values, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices", type=str, default=None, help="Path to .npy vertices with shape (6890,3) or (B,6890,3).")
    parser.add_argument("--smpl_params", type=str, default=None, help="Path to .npz/.npy dict containing SMPL pose, shape/betas, and optional trans.")
    parser.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--lidar_pos", type=str, default="1,-8,0")
    parser.add_argument("--point_num", type=int, default=1024)
    parser.add_argument("--output", type=str, default="partial_lidar_pc_1024.npy")
    args = parser.parse_args()

    if (args.vertices is None) == (args.smpl_params is None):
        raise ValueError("Use exactly one of --vertices or --smpl_params.")

    device = "cuda"
    lidar_pos = parse_lidar_pos(args.lidar_pos)

    if args.vertices is not None:
        vertices = normalize_vertices_shape(load_array(args.vertices))
        vertices = to_tensor(vertices, device)
    else:
        vertices = vertices_from_smpl_params(args.smpl_params, device, gender=args.gender)

    partial_pc = generate_partial_lidar_pc(
        vertices=vertices,
        lidar_pos=lidar_pos,
        point_num=args.point_num,
        device=device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, partial_pc)
    print(f"saved: {args.output}")
    print(f"partial point cloud shape: {partial_pc.shape}")


if __name__ == "__main__":
    main()
