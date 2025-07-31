import random
from math import gcd, ceil, log
from gmssl import sm3
import time
import hashlib
from tqdm import trange

# 数据类型装换
# 整数到字节串的转换。接收非负整数x和字节串的目标长度k，k满足2^8k > x。返回值是长为k的字节串。k是给定的参数。
def int_to_bytes(x, k):
    if pow(256, k) <= x:
        raise Exception("无法实现整数到字节串的转换，目标字节串长度过短！")
    # s是k*2位十六进制串
    s = hex(x)[2:].rjust(k*2, '0')
    M = b''
    for i in range(k):
        M = M + bytes([eval('0x' + s[i*2:i*2+2])])
    return M


# 字节串到整数的转换。接受长度为k的字节串。返回值是整数x
def bytes_to_int(M):
    k = len(M)
    x = 0
    for i in range(k-1, -1, -1):
        x += pow(256, k-1-i) * M[i]
    return x


# 比特串到字节串的转换。接收长度为m的比特串s。返回长度为k的字节串M。其中k = [m/8] 向上取整。
def bits_to_bytes(s):
    k = ceil(len(s)/8)
    s = s.rjust(k*8, '0')
    M = b''
    for i in range(k):
        M = M + bytes([eval('0b' + s[i*8: i*8+8])])
    return M


# 字节串到比特串的转换。接收长度为k的字节串M，返回长度为m的比特串s，其中m = 8k。字节串逐位处理即可。
def bytes_to_bits(M):
    s_list = []
    for i in M:
        s_list.append(bin(i)[2:].rjust(8, '0'))
    s = ''.join(s_list)
    return s


# 域元素到字节串的转换。域元素是整数，转换成字节串要明确长度。
def fielde_to_bytes(e):
    q = eval('0x' + '8542D69E 4C044F18 E8B92435 BF6FF7DE 45728391 5C45517D 722EDB8B 08F1DFC3'.replace(' ', ''))
    t = ceil(log(q, 2))
    l = ceil(t / 8)
    return int_to_bytes(e, l)


# 字节串到域元素的转换。直接调用bytes_to_int()。
def bytes_to_fielde(M):
    return bytes_to_int(M)


# 域元素到整数的转换
def fielde_to_int(a):
    return a


# 点到字节串的转换。接收的参数是椭圆曲线上的点p，元组表示。输出字节串S。选用未压缩表示形式
def point_to_bytes(P):
    xp, yp = P[0], P[1]
    x = fielde_to_bytes(xp)
    y = fielde_to_bytes(yp)
    PC = bytes([0x04])
    s = PC + x + y
    return s


# 字节串到点的转换。接收的参数是字节串s，返回椭圆曲线上的点P，点P的坐标用元组表示
def bytes_to_point(s):
    if len(s) % 2 == 0:
        raise Exception("无法实现字节串到点的转换，请检查字节串是否为未压缩形式！")
    l = (len(s) - 1) // 2
    PC = s[0]
    if PC != 4:
        raise Exception("无法实现字节串到点的转换，请检查PC是否为b'04'！")
    x = s[1: l+1]
    y = s[l+1: 2*l+1]
    xp = bytes_to_fielde(x)
    yp = bytes_to_fielde(y)
    P = (xp, yp)
    return P


# 附加数据类型转换
# 域元素到比特串
def fielde_to_bits(a):
    a_bytes = fielde_to_bytes(a)
    a_bits = bytes_to_bits(a_bytes)
    return a_bits


# 点到比特串
def point_to_bits(P):
    p_bytes = point_to_bytes(P)
    p_bits = bytes_to_bits(p_bytes)
    return p_bits


# 整数到比特串
def int_to_bits(x):
    x_bits = bin(x)[2:]
    k = ceil(len(x_bits)/8)
    x_bits = x_bits.rjust(k*8, '0')
    return x_bits


# 字节串到十六进制串
def bytes_to_hex(m):
    h_list = []
    for i in m:
        e = hex(i)[2:].rjust(2, '0')
        h_list.append(e)
    h = ''.join(h_list)
    return h


# 比特串到十六进制
def bits_to_hex(s):
    s_bytes = bits_to_bytes(s)
    s_hex = bytes_to_hex(s_bytes)
    return s_hex


# 十六进制串到比特串
def hex_to_bits(h):
    b_list = []
    for i in h:
        b = bin(eval('0x' + i))[2:].rjust(4, '0')           # 增强型for循环，是i不是h
        b_list.append(b)
    b = ''.join(b_list)
    return b


# 十六进制到字节串
def hex_to_bytes(h):
    h_bits = hex_to_bits(h)
    h_bytes = bits_to_bytes(h_bits)
    return h_bytes


# 域元素到十六进制串
def fielde_to_hex(e):
    h_bytes = fielde_to_bytes(e)
    h = bytes_to_hex(h_bytes)
    return h


# 密钥派生函数KDF。接收的参数是比特串Z和要获得的密钥数据的长度klen。返回klen长度的密钥数据比特串K
def KDF(Z, klen):
    v = 256
    if klen >= (pow(2, 32) - 1) * v:
        raise Exception("密钥派生函数KDF出错，请检查klen的大小！")
    ct = 0x00000001
    if klen % v == 0:
        l = klen // v
    else:
        l = klen // v + 1
    Ha = []
    for i in range(l):
        # s存储 Z || ct 的比特串形式 # 注意，ct要填充为32位
        s = Z + int_to_bits(ct).rjust(32, '0')
        s_bytes = bits_to_bytes(s)
        s_list = [i for i in s_bytes]
        hash_hex = sm3.sm3_hash(s_list)
        hash_bin = hex_to_bits(hash_hex)
        Ha.append(hash_bin)
        ct += 1
    if klen % v != 0:
        Ha[-1] = Ha[-1][:klen - v*(klen//v)]
    k = ''.join(Ha)
    return k


# 模逆算法。返回M模m的逆。在将分式模运算转换为整数时用，分子分母同时乘上分母的模逆。
def calc_inverse(M, m):
    if gcd(M, m) != 1:
        return None
    u1, u2, u3 = 1, 0, M
    v1, v2, v3 = 0, 1, m
    while v3 != 0:
        q = u3 // v3
        v1, v2, v3, u1, u2, u3 = (u1 - q * v1), (u2 - q * v2), (u3 - q * v3), v1, v2, v3
    return u1 % m


# 将分式模运算转换为整数。输入 up/down mod m, 返回该分式在模m意义下的整数。点加和二倍点运算时求λ用。
def frac_to_int(up, down, p):
    num = gcd(up, down)
    up //= num
    down //= num         # 分子分母约分
    return up * calc_inverse(down, p) % p

# 椭圆曲线上的点加运算。接收的参数是元组P和Q，表示相加的两个点，p为模数。返回二者的点加和
def add_point(P, Q, p):
    if P == 0:
        return Q
    if Q == 0:
        return P
    x1, y1, x2, y2 = P[0], P[1], Q[0], Q[1]
    e = frac_to_int(y2 - y1, x2 - x1, p)
    x3 = (e*e - x1 - x2) % p
    y3 = (e * (x1 - x3) - y1) % p
    ans = (x3, y3)
    return ans


# 二倍点算法。不能直接用点加算法，否则会发生除零错误。接收的参数是点P，素数p，椭圆曲线参数a。返回P的二倍点。
def double_point(P, p, a):
    if P == 0:
        return P
    x1, y1 = P[0], P[1]
    e = frac_to_int(3 * x1 * x1 + a, 2 * y1, p)
    x3 = (e * e - 2 * x1) % p
    y3 = (e * (x1 - x3) - y1) % p
    Q = (x3, y3)
    return Q


# 多倍点算法。通过二进制展开法实现。
# 优化
def mult_point(P, k, p, a):
    s = bin(k)[2:]          # s是k的二进制串形式
    Q = 0
    for i in s:
        Q = double_point(Q, p, a)
        if i == '1':
            Q = add_point(P, Q, p)
    return Q



# 验证某个点是否在椭圆曲线上。接收的参数是椭圆曲线系统参数args和要验证的点P(x, y)。
def on_curve(args, P):
    p, a, b, h, G, n = args
    x, y = P
    if pow(y, 2, p) == ((pow(x, 3, p) + a*x + b) % p):
        return True
    return False


def ecdsa_sign(args, d, M, hash_func='sha256'):
    """
    ECDSA数字签名生成算法
    参数:
        args - 椭圆曲线参数(p, a, b, h, G, n)
        d - 签名方私钥
        M - 待签名的消息(字符串或字节)
        hash_func - 使用的哈希算法，默认sha256
    返回:
        数字签名(r, s)
    """
    p, a, b, h, G, n = args

    # 1. 计算消息哈希
    if isinstance(M, str):
        M = M.encode('utf-8')

    if hash_func.lower() == 'sha256':

        e = int.from_bytes(hashlib.sha256(M).digest(), 'big')
    elif hash_func.lower() == 'sm3':
        hash_list = [i for i in M]
        e_hex = sm3.sm3_hash(hash_list)
        e = int(e_hex, 16)
    else:
        raise ValueError("不支持的哈希算法")

    # 取哈希值的左边n的位长度
    e_bits = n.bit_length()
    e = e >> (256 - e_bits) if e_bits < 256 else e
    e = e % n

    # 2. 签名生成循环
    while True:
        # 3. 生成随机数k ∈ [1, n-1]
        k = random.randint(1, n- 1)

        # 4. 计算椭圆曲线点(x1, y1) = [k]G
        P = mult_point(G, k, p, a)
        x1 = P[0]

        # 5. 计算 r = x1 mod n
        r = x1 % n
        if r == 0:
            continue

        # 6. 计算 s = (k⁻¹ * (e + r*d)) mod n
        k_inv = calc_inverse(k, n)
        if k_inv is None:
            continue
        s = (k_inv * (e + r * d)) % n
        if s == 0:
            continue

        return (r, s)


def ecdsa_verify(args, Q, M, signature, hash_func='sha256'):
    """
    ECDSA数字签名验证算法
    参数:
        args - 椭圆曲线参数(p, a, b, h, G, n)
        Q - 签名方公钥
        M - 原始消息
        signature - 待验证的签名(r, s)
        hash_func - 使用的哈希算法，默认sha256
    返回:
        签名有效返回True，否则返回False
    """
    p, a, b, h, G, n = args
    r, s = signature

    # 1. 验证r和s范围
    if not (1 <= r <= n - 1) or not (1 <= s <= n - 1):
        return False

    # 2. 计算消息哈希
    if isinstance(M, str):
        M = M.encode('utf-8')

    if hash_func.lower() == 'sha256':
        import hashlib
        e = int.from_bytes(hashlib.sha256(M).digest(), 'big')
    elif hash_func.lower() == 'sm3':
        hash_list = [i for i in M]
        e_hex = sm3.sm3_hash(hash_list)
        e = int(e_hex, 16)
    else:
        raise ValueError("不支持的哈希算法")

    # 取哈希值的左边n的位长度
    e_bits = n.bit_length()
    e = e >> (256 - e_bits) if e_bits < 256 else e
    e = e % n

    # 3. 计算 s⁻¹ mod n
    s_inv = calc_inverse(s, n)
    if s_inv is None:
        return False

    # 4. 计算 u1 = e * s⁻¹ mod n 和 u2 = r * s⁻¹ mod n
    u1 = (e * s_inv) % n
    u2 = (r * s_inv) % n

    # 5. 计算 (x1, y1) = [u1]G + [u2]Q
    u1G = mult_point(G, u1, p, a)
    u2Q = mult_point(Q, u2, p, a)
    P = add_point(u1G, u2Q, p)

    if P == 0:  # 无穷远点
        return False

    # 6. 验证 r ≡ x1 mod n
    return r == P[0] % n



# 椭圆曲线系统参数args(p, a, b, h, G, n)的获取。
def get_args():
    p = eval('0x' + '8542D69E 4C044F18 E8B92435 BF6FF7DE 45728391 5C45517D 722EDB8B 08F1DFC3'.replace(' ', ''))
    a = eval('0x' + '787968B4 FA32C3FD 2417842E 73BBFEFF 2F3C848B 6831D7E0 EC65228B 3937E498'.replace(' ', ''))
    b = eval('0x' + '63E4C6D3 B23B0C84 9CF84241 484BFE48 F61D59A5 B16BA06E 6E12D1DA 27C5249A'.replace(' ', ''))
    h = 1
    xG = eval('0x' + '421DEBD6 1B62EAB6 746434EB C3CC315E 32220B3B ADD50BDC 4C4E6C14 7FEDD43D'.replace(' ', ''))
    yG = eval('0x' + '0680512B CBB42C07 D47349D2 153B70C4 E5D7FDFC BFA36EA1 A85841B9 E46E09A2'.replace(' ', ''))
    G = (xG, yG)            # G 是基点
    n = eval('0x' + '8542D69E 4C044F18 E8B92435 BF6FF7DD 29772063 0485628D 5AE74EE7 C32E79B7'.replace(' ', ''))
    args = (p, a, b, h, G, n)           # args是存储椭圆曲线参数的元组。
    return args


# 密钥获取。本程序中主要是消息接收方B的公私钥的获取。
def get_key():
    xA = eval('0x' + '0AE4C779 8AA0F119 471BEE11 825BE462 02BB79E2 A5844495 E97C04FF 4DF2548A'.replace(' ', ''))
    yA = eval('0x' + '7C0240F8 8F1CD4E1 6352A73C 17B7F16F 07353E53 A176D684 A9FE0C6B B798E857'.replace(' ', ''))
    PA = (xA, yA)           # PB是B的公钥
    dA = eval('0x' + '128B2FA8 BD433C6C 068C8D80 3DFF7979 2A519A55 171B1B65 0C23661D 15897263'.replace(' ', ''))
    # dB是B的私钥
    key_A = (PA, dA)
    return key_A


# 示例用法
if __name__ == "__main__":
    # 使用与SM2相同的椭圆曲线参数(实际中ECDSA常用secp256k1)
    args = get_args()
    p, a, b, h, G, n = args

    # 生成密钥对
    d = random.randint(1, n - 1)  # 私钥
    Q = mult_point(G, d, p, a)  # 公钥

    M = "This is a test message"

    print("\nECDSA签名示例(r,s):")
    signature = ecdsa_sign(args, d, M)
    r,s=signature
    print(f"签名(r, s): {hex(signature[0])},{hex(signature[1])}")

    print("\nECDSA验证:")
    is_valid = ecdsa_verify(args, Q, M, signature)
    print(f"验证结果: {is_valid}")

    print("\nECDSA签名伪造(r,-s):")
    signature1=(r,(-s)%n)
    print(f"签名(r, -s): {hex(signature1[0])},{hex(signature1[1])}")
    is_valid = ecdsa_verify(args, Q, M, signature1)
    print(f"伪造签名的验证结果: {is_valid}")

