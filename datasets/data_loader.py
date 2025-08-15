import torch
from collections import defaultdict
from torch_geometric.data import Data
import numpy as np

class DataCenter(object):
    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def load_dataSet(self, embedding_path, trajectory_path, edge_path):
        """
        加载轨迹数据、节点特征向量，并构建 PyG Data 对象。
        使用 edges.txt 中的边关系替代原本轨迹中的边构建逻辑。
        """
        # =========================
        # 1 读取节点特征向量 data.dat
        # =========================
        node_features = []
        node_index_map = {}
        with open(embedding_path, 'r') as f:
            for idx, line in enumerate(f):
                info = line.strip().split()
                if len(info) > 3:  # 忽略空行或异常行
                    node_features.append([float(x) for x in info[1:]])  # 节点特征
                    node_index_map[info[0]] = idx  # 节点ID -> 特征索引
        print(f"总节点数: {len(node_features)}")

        # =========================
        # 2 读取轨迹文件，建立用户ID映射
        # =========================
        user_id_map = defaultdict(int)
        with open(trajectory_path, 'r') as f:
            for line in f:
                info = line.strip().split()
                user_id = info[0]
                if user_id not in user_id_map:
                    user_id_map[user_id] = len(user_id_map)
        print(f"总用户数: {len(user_id_map)}")
        num_users = len(user_id_map)

        # =========================
        # 3 读取边关系 edges.txt -> 构建邻居字典
        # =========================
        edge_dict = defaultdict(list)
        with open(edge_path, 'r') as f:
            for line in f:
                u, v = line.strip().split('-')
                u, v = int(u), int(v)
                edge_dict[u].append(v)
                edge_dict[v].append(u)  # 无向图

        # =========================
        # 4 处理轨迹，生成不同粒度的 Data 对象
        # =========================
        dataset_full = []
        dataset_level2 = []
        dataset_level3 = []

        for i, line in enumerate(open(trajectory_path, 'r')):
            parts = line.strip().split()
            user_id = parts[0]
            label = user_id_map[user_id]
            traj_nodes = [int(x) for x in parts[1:]]

            # --- 三种粒度轨迹特征 ---
            def build_feature_and_edge(traj_nodes, stride=1):
                """
                构建粒度轨迹的节点特征和 edge_index。
                使用轨迹内节点的本地索引映射全局边。
                """
                # 采样后的节点
                sampled_nodes = traj_nodes[::stride]

                # 节点特征矩阵 & 本地编号映射
                feature_list = []
                local_node_map = {}  # 全局编号 -> 本地编号
                for idx, node in enumerate(sampled_nodes):
                    local_node_map[node] = idx
                    try:
                        feature_list.append(node_features[node])
                    except IndexError:
                        feature_list.append(node_features[0])

                # 构建 edge_index：只保留轨迹内节点的邻居
                edge_set = set()
                for node in sampled_nodes:
                    neighbors = edge_dict.get(node, [])
                    for neighbor in neighbors:
                        if neighbor in local_node_map:  # 只保留轨迹内的节点
                            edge_set.add((local_node_map[node], local_node_map[neighbor]))
                            edge_set.add((local_node_map[neighbor], local_node_map[node]))

                edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
                x = torch.tensor(feature_list, dtype=torch.float)
                y = torch.tensor([label], dtype=torch.long)
                return Data(x=x, edge_index=edge_index, y=y)

            # 构建三种粒度
            data_1 = build_feature_and_edge(traj_nodes, stride=1)
            data_2 = build_feature_and_edge(traj_nodes, stride=2)
            data_3 = build_feature_and_edge(traj_nodes, stride=3)

            dataset_full.append(data_1)
            dataset_level2.append(data_2)
            dataset_level3.append(data_3)

        # =========================
        # 5 按每10条轨迹划分训练/测试集
        # =========================
        train_full, test_full = [], []
        train_lvl2, test_lvl2 = [], []
        train_lvl3, test_lvl3 = [], []

        for i in range(0, len(dataset_full), 10):
            train_full += dataset_full[i:i + 7]
            test_full += dataset_full[i + 7:i + 10]
            train_lvl2 += dataset_level2[i:i + 7]
            test_lvl2 += dataset_level2[i + 7:i + 10]
            train_lvl3 += dataset_level3[i:i + 7]
            test_lvl3 += dataset_level3[i + 7:i + 10]

        print(f"总轨迹数: {len(dataset_full)}, 训练集: {len(train_full)}, 测试集: {len(test_full)}")

        # =========================
        # 6️⃣ 保存到对象属性
        # =========================
        setattr(self, '_all_data', dataset_full)
        setattr(self, '_train', train_full)
        setattr(self, '_train2', train_lvl2)
        setattr(self, '_train3', train_lvl3)
        setattr(self, '_test', test_full)
        setattr(self, '_test2', test_lvl2)
        setattr(self, '_test3', test_lvl3)
        setattr(self, '_number', num_users)
        setattr(self, '_feats', np.asarray(node_features))

        print("数据加载完成。")
