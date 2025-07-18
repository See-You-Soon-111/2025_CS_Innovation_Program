import hashlib
import math
from typing import List, Tuple, Optional, Union
import bisect
# SM3哈希函数
def sm3(data: bytes) -> bytes:
    h = hashlib.new('sm3')
    h.update(data)
    return h.digest()

# RFC6962中定义的哈希方式
def rfc6962_hash_leaf(leaf: bytes) -> bytes:
    # 叶子节点哈希前缀: 0x00
    return sm3(b'\x00' + leaf)

def rfc6962_hash_node(left: bytes, right: bytes) -> bytes:
    # 内部节点哈希前缀: 0x01
    return sm3(b'\x01' + left + right)


class MerkleTree:
    def __init__(self, leaves: List[bytes]):
        self.leaves = leaves
        self.tree = self.build_tree(leaves)
        self.root = self.tree[-1][0] if self.tree else None

    def build_tree(self, leaves: List[bytes]) -> List[List[bytes]]:
        """构建Merkle树"""
        if not leaves:
            return []
        # 哈希所有叶子节点
        tree = [[rfc6962_hash_leaf(leaf) for leaf in leaves]]
        # 逐层构建树
        current_level = tree[0]
        while len(current_level) > 1:
            next_level = []
            # 处理成对的节点
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(rfc6962_hash_node(left, right))
            tree.append(next_level)
            current_level = next_level
        return tree

    def get_root(self) -> bytes:
        """获取Merkle根"""
        return self.root

    def get_proof(self, index: int) -> List[bytes]:
        """获取存在性证明的路径"""
        if index < 0 or index >= len(self.leaves):
            raise IndexError("Leaf index out of range")
        proof = []
        for level in self.tree[:-1]:
            # 添加兄弟节点到证明路径
            if index % 2 == 1:
                proof.append(level[index - 1])
            else:
                if index + 1 < len(level):
                    proof.append(level[index + 1])
                else:
                    # 如果是奇数个节点且是最后一个，不需要添加
                    pass
            # 向上移动到父节点
            index = index // 2
        return proof

    def verify_proof(self, leaf: bytes, proof: List[bytes], index: int) -> bool:
        """验证存在性证明"""
        current_hash = rfc6962_hash_leaf(leaf)
        for i, sibling in enumerate(proof):
            if index % 2 == 0:
                # 当前节点是左节点
                current_hash = rfc6962_hash_node(current_hash, sibling)
            else:
                # 当前节点是右节点
                current_hash = rfc6962_hash_node(sibling, current_hash)
            index = index // 2
        return current_hash == self.root


    def get_non_inclusion_proof(self, leaf: bytes):
        """获取不存在性证明"""
        # 检查叶子是否已存在
        if leaf in self.leaves:
            return self.leaves.index(leaf),None, None
        # 获取排序后的（叶子值, 原始索引）列表
        sorted_leaves = sorted((leaf_val, idx) for idx, leaf_val in enumerate(self.leaves))
        leaf_values = [item[0] for item in sorted_leaves]
        original_indices = [item[1] for item in sorted_leaves]
        # 使用二分查找定位插入位置
        insert_pos = bisect.bisect_left(leaf_values, leaf)
        # 处理三种情况
        if insert_pos == 0:
            # 情况1：比所有叶子都小 -> 只需证明最小叶子存在
            neighbor = leaf_values[0]
            proof = self.get_proof(original_indices[0])
            return insert_pos,None, proof

        elif insert_pos == len(leaf_values):
            # 情况2：比所有叶子都大 -> 只需证明最大叶子存在
            neighbor = leaf_values[-1]
            proof = self.get_proof(original_indices[-1])
            return insert_pos,None, proof

        else:
            # 情况3：在两个叶子之间 -> 需证明左右叶子连续
            left_neighbor = leaf_values[insert_pos - 1]
            right_neighbor = leaf_values[insert_pos]
            left_proof = self.get_proof(original_indices[insert_pos - 1])
            right_proof = self.get_proof(original_indices[insert_pos])
            return (self.leaves.index(left_neighbor),self.leaves.index(right_neighbor),),(left_neighbor, right_neighbor), (left_proof, right_proof)


def generate_leaves(n: int = 100000) -> List[bytes]:
    """生成10万叶子节点"""
    return [str(i).encode('utf-8') for i in range(n)]


def test(leaf,leaves,merkle_tree):
    """测试节点是否存在"""
    idx,neighbors,non_inclusion_proof=merkle_tree.get_non_inclusion_proof(leaf)
    # 不存在证明为空，说明存在
    if non_inclusion_proof is None:
        print(f"该节点存在。")
    else:
        if neighbors is None:
            print(f"该节点只有一个邻居：{leaves[idx]}")
            is_valid = merkle_tree.verify_proof(leaves[idx], merkle_tree.get_proof(idx), idx)
            if is_valid:
                print(f"该节点邻居存在，说明该节点不存在。")
            else:
                print(f"该节点不存在证明生成失败。")

        else:
            left_idx,right_idx=idx
            left_neighbor, right_neighbor = neighbors
            left_proof, right_proof = non_inclusion_proof
            print(f"该节点的左右邻居为：{left_neighbor}，{right_neighbor}")
            left_is_valid = merkle_tree.verify_proof(left_neighbor, left_proof, left_idx)
            right_is_valid = merkle_tree.verify_proof(right_neighbor, right_proof, right_idx)
            if left_is_valid and right_is_valid:
                print(f"该节点左右邻居均存在，说明该节点不存在。")
            else:
                print(f"该节点不存在证明生成失败。")


if __name__ == "__main__":
    # 生成10万叶子节点
    leaves = generate_leaves(100000)

    # 构建Merkle树
    merkle_tree = MerkleTree(leaves)
    print("构建 Merkle tree ")
    print(f"根节点哈希值: {merkle_tree.get_root().hex()}")

    # 测试已存在节点证明
    existing_leaf=leaves[100]
    print(f"测试已存在节点：{existing_leaf}")
    test(existing_leaf,leaves,merkle_tree)
    print(f"生成认证路径：")
    proof=merkle_tree.get_proof(100)
    print(f"认证路径：")
    for p in proof:
        print(p.hex())
    is_valid=merkle_tree.verify_proof(existing_leaf,proof,100)
    print(f"验证结果：{is_valid}")

    # 测试不存在节点证明
    non_existing_leaf=b'100866'
    print(f"测试不存在节点：{non_existing_leaf}")
    test(non_existing_leaf,leaves,merkle_tree)