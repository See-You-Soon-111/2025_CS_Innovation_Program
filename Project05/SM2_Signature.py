import random
from math import gcd, ceil, log
from gmssl import sm3
import time
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


def sm2_sign(args, dA, M, ZA=None, IDA=None):
    """
    SM2数字签名生成算法
    参数:
        args - 椭圆曲线参数(p, a, b, h, G, n)
        dA - 签名方私钥
        M - 待签名的消息(字符串或字节)
        ZA - 用户哈希身份标识(可选)
        IDA - 用户身份字符串(可选)
    返回:
        数字签名(r, s)
    """
    # 解包椭圆曲线参数
    p, a, b, h, G, n = args

    # 步骤1：如果未提供ZA，则根据IDA计算ZA
    if ZA is None:
        if IDA is None:
            raise ValueError("ZA和IDA不能为空")

        # ZA = H256(ENTLA || IDA || a || b || xG || yG || xA || yA)
        # 计算ENTLA(IDA的比特长度，2字节)
        entlen = len(IDA.encode('ascii')) * 8
        ENTLA = int_to_bytes(entlen, 2)

        # 通过私钥计算公钥PA = [dA]G
        PA = mult_point(G, dA, p, a)
        xA, yA = PA[0], PA[1]

        # 将所有参数转换为字节串
        a_bytes = fielde_to_bytes(a)
        b_bytes = fielde_to_bytes(b)
        xG_bytes = fielde_to_bytes(G[0])
        yG_bytes = fielde_to_bytes(G[1])
        xA_bytes = fielde_to_bytes(xA)
        yA_bytes = fielde_to_bytes(yA)

        # 拼接所有数据
        data = ENTLA + IDA.encode('ascii') + a_bytes + b_bytes + xG_bytes + yG_bytes + xA_bytes + yA_bytes

        # 计算SM3哈希得到ZA
        hash_list = [i for i in data]
        ZA = sm3.sm3_hash(hash_list)
        ZA_bits = hex_to_bits(ZA)
    else:
        ZA_bits = hex_to_bits(ZA)

    # 步骤2：计算 e = Hv(M')，其中 M' = ZA || M
    M_bytes = M.encode('ascii') if isinstance(M, str) else M
    ZA_bytes = bits_to_bytes(ZA_bits)
    M_prime = ZA_bytes + M_bytes

    # 计算M'的SM3哈希值
    hash_list = [i for i in M_prime]
    e_hex = sm3.sm3_hash(hash_list)
    e_bits = hex_to_bits(e_hex)
    e = bytes_to_int(bits_to_bytes(e_bits))

    # 步骤3-6：签名生成循环
    while True:
        # 步骤3：生成随机数k ∈ [1, n-1]
        # k = random.randint(1, n - 1)
        k=eval('0x'+'6CB28D99 385C175C 94F94E93 4817663F C176D925 DD72B727 260DBAAE 1FB2F96F'.replace(' ',''))
        # 步骤4：计算椭圆曲线点(x1, y1) = [k]G
        P = mult_point(G, k, p, a)
        x1, y1 = P[0], P[1]

        # 步骤5：计算 r = (e + x1) mod n
        r = (e + x1) % n

        # 检查r和r+k是否为0(需要重新生成k)
        if r == 0 or r + k == n:
            continue

        # 步骤6：计算 s = ((1 + dA)^-1 * (k - r*dA)) mod n
        dA_plus_1 = (1 + dA) % n
        dA_plus_1_inv = calc_inverse(dA_plus_1, n)
        if dA_plus_1_inv is None:  # 理论上不会发生，因为n是素数
            continue

        s = (dA_plus_1_inv * (k - r * dA)) % n

        # 检查s是否为0(需要重新生成k)
        if s == 0:
            continue

        # 步骤7：返回有效的签名(r, s)
        return (r,s)


def sm2_verify(args, PA, M, signature, ZA=None, IDA=None):
    """
    SM2数字签名验证算法
    参数:
        args - 椭圆曲线参数(p, a, b, h, G, n)
        PA - 签名方公钥
        M - 原始消息(字符串或字节)
        signature - 待验证的签名(r, s)
        ZA - 用户哈希身份标识(可选)
        IDA - 用户身份字符串(可选)
    返回:
        签名有效返回True，否则返回False
    """
    # 解包椭圆曲线参数和签名
    p, a, b, h, G, n = args
    r, s = signature
    # 步骤1：验证r和s是否在[1, n-1]范围内
    if not (1 <= r <= n - 1) or not (1 <= s <= n - 1):
        return False

    # 步骤2：如果未提供ZA，则根据IDA计算ZA
    if ZA is None:
        if IDA is None:
            raise ValueError("必须提供ZA或IDA")

        # ZA = H256(ENTLA || IDA || a || b || xG || yG || xA || yA)
        entlen = len(IDA.encode('ascii')) * 8
        ENTLA = int_to_bytes(entlen, 2)

        # 从公钥PA中提取坐标
        xA, yA = PA[0], PA[1]

        # 将所有参数转换为字节串
        a_bytes = fielde_to_bytes(a)
        b_bytes = fielde_to_bytes(b)
        xG_bytes = fielde_to_bytes(G[0])
        yG_bytes = fielde_to_bytes(G[1])
        xA_bytes = fielde_to_bytes(xA)
        yA_bytes = fielde_to_bytes(yA)

        # 拼接所有数据
        data = ENTLA + IDA.encode('ascii') + a_bytes + b_bytes + xG_bytes + yG_bytes + xA_bytes + yA_bytes

        # 计算SM3哈希得到ZA
        hash_list = [i for i in data]
        ZA = sm3.sm3_hash(hash_list)

        ZA_bits = hex_to_bits(ZA)
    else:
        ZA_bits = hex_to_bits(ZA)

    # 步骤3：计算 e = Hv(M')，其中 M' = ZA || M
    M_bytes = M.encode('ascii') if isinstance(M, str) else M
    ZA_bytes = bits_to_bytes(ZA_bits)
    M_prime = ZA_bytes + M_bytes

    # 计算M'的SM3哈希值
    hash_list = [i for i in M_prime]
    e_hex = sm3.sm3_hash(hash_list)
    e_bits = hex_to_bits(e_hex)
    e = bytes_to_int(bits_to_bytes(e_bits))

    # 步骤4：计算 t = (r + s) mod n
    t = (r + s) % n
    if t == 0:  # t为0时直接验证失败
        return False

    # 步骤5：计算 (x1', y1') = [s]G + [t]PA
    sG = mult_point(G, s, p, a)  # [s]G
    tPA = mult_point(PA, t, p, a)  # [t]PA
    P = add_point(sG, tPA, p)  # 点相加
    x1_prime, y1_prime = P[0], P[1]

    # 步骤6：计算 R = (e + x1') mod n
    R = (e + x1_prime) % n

    # 步骤7：验证 R == r
    return R == r


# k泄露时，可恢复出私钥
def sm2_sign_leaking_k(args,signature,k):
    p, a, b, h, G, n = args
    r,s=signature
    r_plus_s_inv=calc_inverse((s+r)%n,n)
    return hex((r_plus_s_inv*(k-s))%n)

#k重用时，可恢复出私钥
def sm2_sign_reusing_k(args,signature1,signature2):
    p, a, b, h, G, n = args
    r1,s1=signature1
    r2,s2=signature2
    return hex(frac_to_int(s2-s1,s1-s2+r1-r2,n))

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


if __name__ == "__main__":

    args = get_args()
    p, a, b, h, G, n = args

    key_A = get_key()
    PA, dA = key_A
    IDA = "ALICE123@YAHOO.COM"
    M = "message digest"

    print("\nSM2签名...")
    signature = sm2_sign(args, dA, M, IDA=IDA)
    r, s = signature
    print(f"生成签名 (r, s): {hex(r)}, {hex(s)}")

    print("\nSM2验签...")
    is_valid = sm2_verify(args, PA, M, signature, IDA=IDA)
    print(f"签名是否有效: {is_valid}")

    # 测试错误信息
    print("\n测试错误信息...")
    is_valid = sm2_verify(args, PA, "wrong message", signature, IDA=IDA)
    print(f"签名是否有效 (应该为无效): {is_valid}")

    print("\n随机数k泄露时，可恢复出私钥：")
    leaking_k= eval('0x' + '6CB28D99 385C175C 94F94E93 4817663F C176D925 DD72B727 260DBAAE 1FB2F96F'.replace(' ', ''))
    print(sm2_sign_leaking_k(args,signature,leaking_k))

    signature2=sm2_sign(args,dA,"message digest2",IDA=IDA)
    print("\n随机数k重用时，可恢复出私钥：")
    print(sm2_sign_reusing_k(args,signature,signature2))
