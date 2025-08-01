import random
from typing import List, Tuple, Dict, Set
import hashlib
from phe import paillier  # 使用phe库实现Paillier加密


# 模拟群G的操作（实际实现应使用椭圆曲线或大整数群）
class Group:
    def __init__(self, p, g):
        self.p = p  # 素数阶
        self.g = g  # 生成元

    def hash_to_group(self, x: str) -> int:
        """将输入哈希到群元素"""
        h = hashlib.sha256(x.encode()).hexdigest()
        return pow(self.g, int(h, 16) % self.p, self.p)

    def exponentiate(self, x: int, exponent: int) -> int:
        """群指数运算"""
        return pow(x, exponent, self.p)


# 使用Paillier加法同态加密
class AdditiveHomomorphicEncryption:
    def __init__(self, key_length=2048):
        # 生成Paillier公私钥对
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)

    def encrypt(self, plaintext: int) -> paillier.EncryptedNumber:
        """加密整数"""
        return self.public_key.encrypt(plaintext)

    def decrypt(self, ciphertext: paillier.EncryptedNumber) -> int:
        """解密到整数"""
        return self.private_key.decrypt(ciphertext)

    def add(self, ciphertext1: paillier.EncryptedNumber,
            ciphertext2: paillier.EncryptedNumber) -> paillier.EncryptedNumber:
        """同态加法"""
        return ciphertext1 + ciphertext2

    def add_plain(self, ciphertext: paillier.EncryptedNumber, plaintext: int) -> paillier.EncryptedNumber:
        """加密数加明文"""
        return ciphertext + plaintext

    def multiply_plain(self, ciphertext: paillier.EncryptedNumber, plaintext: int) -> paillier.EncryptedNumber:
        """加密数乘明文"""
        return ciphertext * plaintext


# 协议实现
class DDHPrivateIntersectionSum:
    def __init__(self):
        # 初始化群参数（实际应使用安全参数）
        self.group = Group(
            p=3231700607131100730071487668866995196044410266971548403213034542752465513886789089319720141152291346368871796092189801949411955915049093069501525039194165,
            g=2)

    def party1_round1(self, V: List[str]) -> Tuple[List[int], int]:
        """P1第1轮: 哈希并指数化自己的元素"""
        self.k1 = random.randint(1, self.group.p - 1)
        hashed_exponents = []
        for v in V:
            h = self.group.hash_to_group(v)
            hashed_exponents.append(self.group.exponentiate(h, self.k1))

        # 打乱顺序
        random.shuffle(hashed_exponents)
        return hashed_exponents, self.k1

    def party2_round2(self, received_from_p1: List[int], W: List[Tuple[str, int]]) -> Tuple[
        List[int], List[Tuple[int, paillier.EncryptedNumber]], paillier.PaillierPublicKey]:
        """P2第2轮: 双重指数化并加密关联值"""
        self.k2 = random.randint(1, self.group.p - 1)
        self.ahe = AdditiveHomomorphicEncryption()

        # 处理P1的元素
        double_exponents = []
        for elem in received_from_p1:
            double_exponents.append(self.group.exponentiate(elem, self.k2))
        random.shuffle(double_exponents)

        # 处理自己的元素
        hashed_encrypted = []
        for w, t in W:
            h = self.group.hash_to_group(w)
            hashed = self.group.exponentiate(h, self.k2)
            encrypted = self.ahe.encrypt(t)
            hashed_encrypted.append((hashed, encrypted))
        random.shuffle(hashed_encrypted)

        return double_exponents, hashed_encrypted, self.ahe.public_key

    def party1_round3(self, received_double_exponents: List[int],
                      received_hashed_encrypted: List[Tuple[int, paillier.EncryptedNumber]],
                      k1: int, V: List[str]) -> paillier.EncryptedNumber:
        """P1第3轮: 计算交集和"""
        # 计算交集
        intersection_indices = []
        encrypted_sum = None

        # 将P1的双重指数化结果存入集合便于查找
        p1_elements = set(received_double_exponents)

        for idx, (hashed, encrypted) in enumerate(received_hashed_encrypted):
            # 完成双重指数化
            double_hashed = self.group.exponentiate(hashed, k1)

            # 检查是否在交集中
            if double_hashed in p1_elements:
                intersection_indices.append(idx)

                # 同态累加
                if encrypted_sum is None:
                    encrypted_sum = encrypted
                else:
                    encrypted_sum = self.ahe.add(encrypted_sum, encrypted)

        # 随机化最终的和（Paillier加密本身已经具有随机性）
        return encrypted_sum

    def party2_output(self, encrypted_sum: paillier.EncryptedNumber) -> int:
        """P2输出: 解密得到交集和"""
        return self.ahe.decrypt(encrypted_sum)


# 测试协议
def test_protocol():
    # 模拟数据
    V = ["user1", "user2", "user3", "user4"]
    W = [("user2", 10), ("user3", 20), ("user4", 30), ("user6", 40)]

    protocol = DDHPrivateIntersectionSum()

    # P1第1轮
    p1_round1_result, k1 = protocol.party1_round1(V)

    # P2第2轮
    p2_round2_result1, p2_round2_result2, p2_pubkey = protocol.party2_round2(p1_round1_result, W)

    # P1第3轮
    p1_round3_result = protocol.party1_round3(p2_round2_result1, p2_round2_result2, k1, V)

    # P2输出
    intersection_sum = protocol.party2_output(p1_round3_result)

    print(f"交集和为: {intersection_sum}")  # 应输出60 (user2 + user3 + user4)


if __name__ == "__main__":
    test_protocol()