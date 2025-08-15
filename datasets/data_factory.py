from collections import defaultdict
from gensim.models import Word2Vec
import numpy as np
import math
# ===== 数据读取 =====
def read_checkins(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            ID = parts[0]
            timestamp = parts[1]
            lat = float(parts[2])
            lon = float(parts[3])
            data.append((ID, timestamp, lat, lon))
    return data

# ===== 分组 =====
def group_by_id_and_date(data):
    grouped = defaultdict(list)
    for ID, timestamp, lat, lon in data:
        date = timestamp.split("T")[0]
        grouped[(ID, date)].append((lat, lon))
    result = []
    for (ID, date), coords in grouped.items():
        result.append([ID] + coords)
    return result

# ===== 经纬度 → 网格编号（乘100取整） =====
def to_grid(lat, lon):
    gx = int(lat * 100)
    gy = int(lon * 100)
    return (gx, gy)

# ===== 构建网格和边关系 =====
def build_grid_edges(data):
    grid_map = {}      # (gx, gy) -> grid_id
    grid_coords = {}   # grid_id -> (lat, lon)
    next_grid_id = 0
    grid_sequence = []

    for _, _, lat, lon in data:
        g = to_grid(lat, lon)
        if g not in grid_map:
            grid_map[g] = next_grid_id
            grid_coords[next_grid_id] = (g[0] / 100.0, g[1] / 100.0)
            next_grid_id += 1
        grid_sequence.append(grid_map[g])

    edges = set()
    # 顺序连接
    for i in range(len(grid_sequence) - 1):
        a, b = grid_sequence[i], grid_sequence[i + 1]
        if a != b:
            edges.add(tuple(sorted((a, b))))

    # 8邻接
    neighbor_dirs = [
        (-1,-1), (-1,0), (-1,1),
        (0,-1),          (0,1),
        (1,-1),  (1,0),  (1,1)
    ]
    for (gx, gy), gid in grid_map.items():
        for dx, dy in neighbor_dirs:
            ng = (gx + dx, gy + dy)
            if ng in grid_map:
                edges.add(tuple(sorted((gid, grid_map[ng]))))

    return edges, grid_coords, grid_map

# ===== 新增：生成轨迹文件 =====
def build_trajectories(result, grid_map):
    trajectories = []
    for row in result:
        ID = row[0]
        coords = row[1:]
        traj_ids = []
        for lat, lon in coords:
            g = to_grid(lat, lon)
            traj_ids.append(grid_map[g])
        trajectories.append([ID] + traj_ids)
    return trajectories

def read_trajectories(file_path):
    """
    从 trajectories.txt 读取轨迹，忽略第一个ID，只保留经纬度编号
    返回格式：[['1','2','3'], ['5','6',...], ...]
    """
    trajectories = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                trajectories.append(parts[1:])  # 忽略 ID
    return trajectories


def split_train_test(trajectories, group_size=10, train_ratio=0.7):
    """
    按每 group_size 条轨迹分组，再按比例划分训练和测试集
    """
    train_set, test_set = [], []
    for i in range(0, len(trajectories), group_size):
        group = trajectories[i:i + group_size]
        split_idx = int(len(group) * train_ratio)
        train_set.extend(group[:split_idx])
        test_set.extend(group[split_idx:])
    return train_set, test_set


# ===== 主程序 =====
if __name__ == "__main__":
    file_path = "gowalla_big.txt"
    outputemb_file = 'gowembbignew128.dat'
    outputedges_file = 'gowallabigedges.txt'
    outputrajectories_file = "gowallabig.txt"
    datasets_name = 'gowbig' # 247users

    data = read_checkins(file_path)

    # 分组
    result = group_by_id_and_date(data)
    print(f"共分组 {len(result)} 条（ID+日期）记录")
    print("示例：", result[0])

    # 网格化 & 边关系
    edges, grid_coords, grid_map = build_grid_edges(data)

    # 保存边
    with open(outputedges_file, "w") as f:
        for a, b in sorted(edges):
            f.write(f"{a}-{b}\n")

    # 保存网格坐标
    with open(datasets_name + "grid_coords.txt", "w") as f:
        for gid, (latc, lonc) in sorted(grid_coords.items()):
            f.write(f"{gid} {latc} {lonc}\n")

    print("边关系写入 edges.txt，网格坐标写入 grid_coords.txt")

    # 生成轨迹文件
    trajectories = build_trajectories(result, grid_map)
    with open(outputrajectories_file, "w") as f:
        for traj in trajectories:
            f.write(" ".join(str(x) for x in traj) + "\n")

    print("轨迹写入轨迹txt")




    # 1. 读取轨迹
    trajectories = read_trajectories(outputrajectories_file)
    print(f"共读取 {len(trajectories)} 条轨迹")

    # 2. 按 10 条为一组，7:3 划分
    train_set, test_set = split_train_test(trajectories, group_size=10, train_ratio=0.7)
    print(f"训练集: {len(train_set)} 条, 测试集: {len(test_set)} 条")

    model = Word2Vec(
        sentences=train_set,
        vector_size=128,
        min_count=0,
        window=9,
        epochs=100,
        hs=1,
        sg=1
    )

    # 获取已有词向量
    word_vectors = {word: model.wv[word] for word in model.wv.index_to_key}

    # 特殊符号 </s> 的嵌入（随机初始化）
    vector_dim = model.vector_size
    all_vectors = np.array(list(word_vectors.values()))
    mean = all_vectors.mean()
    std = all_vectors.std()
    word_vectors['</s>'] = np.random.normal(loc=mean, scale=std, size=vector_dim)

    # 排序 Word2Vec 学到的编号向量
    sorted_word_vectors = sorted(
        [(word, vec) for word, vec in word_vectors.items() if word != '</s>'],
        key=lambda x: int(x[0])
    )

    # 拼接 </s> 放在第一个
    sorted_word_vectors = [('</s>', word_vectors['</s>'])] + sorted_word_vectors

    # 保存为文本文件，第一行写入 词数 维度

    with open(outputemb_file, 'w', encoding='utf-8') as f:
        f.write(f"{len(sorted_word_vectors)} {vector_dim}\n")  # 第一行
        for word, vector in sorted_word_vectors:
            # 每个值保留6位小数
            vector_str = ' '.join(f"{val:.6f}" for val in vector)
            f.write(f"{word} {vector_str}\n")

    print(f"词向量保存完成，文件：{outputemb_file}")
