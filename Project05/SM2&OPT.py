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

'''
def mult_point(P, k, p,a):
    if k == 1:
        return P
    result = P
    for _ in range(k - 1):
        if result==P:
            result=double_point(P,p,a)
        else:
            result = add_point(result, P, p)
    return result
'''

# 验证某个点是否在椭圆曲线上。接收的参数是椭圆曲线系统参数args和要验证的点P(x, y)。
def on_curve(args, P):
    p, a, b, h, G, n = args
    x, y = P
    if pow(y, 2, p) == ((pow(x, 3, p) + a*x + b) % p):
        return True
    return False


def encry_sm2(args, PB, M):
    """
    SM2加密算法
    参数:
        args - 椭圆曲线系统参数(p, a, b, h, G, n)
        PB - 接收方公钥
        M - 待加密的明文消息
    返回:
        十六进制格式的密文字符串
    """
    p, a, b, h, G, n = args

    # 将明文转换为ASCII字节串
    M_bytes = bytes(M, encoding='ascii')

    # 生成随机数k（实际使用时应取消注释随机生成，此处为测试固定值）
    # k = random.randint(1, n-1)
    k = eval('0x' + '4C62EEFD 6ECFC2B9 5B92FD6C 3D957514 8AFA1742 5546D490 18E5388D 49DD7B4F'.replace(' ', ''))

    # 计算椭圆曲线点C1 = [k]G (生成临时公钥)
    C1 = mult_point(G, k, p, a)
    C1_bits = point_to_bits(C1)

    # 计算共享密钥S = [h]PB (h是余因子)
    S = mult_point(PB, h, p, a)
    if S == 0:
        raise Exception("计算得到的S是无穷远点")

    # 计算椭圆曲线点(x2, y2) = [k]PB (用于密钥派生)
    x2, y2 = mult_point(PB, k, p, a)
    x2_bits = fielde_to_bits(x2)
    y2_bits = fielde_to_bits(y2)

    # 将明文转换为十六进制
    M_hex = bytes_to_hex(M_bytes)
    klen = 4 * len(M_hex)  # 计算需要的密钥长度(位)

    # 使用KDF函数从(x2, y2)派生对称密钥t
    t = KDF(x2_bits + y2_bits, klen)
    if eval('0b' + t) == 0:
        raise Exception("KDF返回了全零串，请检查KDF算法！")

    # 将t转换为十六进制并与明文异或得到C2
    t_hex = bits_to_hex(t)
    C2 = eval('0x' + M_hex + '^' + '0b' + t)

    # 计算消息认证码C3 = SM3(x2 || M || y2)
    x2_bytes = bits_to_bytes(x2_bits)
    y2_bytes = bits_to_bytes(y2_bits)
    hash_list = [i for i in x2_bytes + M_bytes + y2_bytes]
    C3 = sm3.sm3_hash(hash_list)

    # 拼接密文C = C1 || C2 || C3
    C1_hex = bits_to_hex(C1_bits)
    C2_hex = hex(C2)[2:]
    C3_hex = C3
    C_hex = C1_hex + C2_hex + C3_hex

    print("加密得到的密文是：", C_hex)
    return C_hex


def decry_sm2(args, dB, C):
    """
    SM2解密算法
    参数:
        args - 椭圆曲线系统参数(p, a, b, h, G, n)
        dB - 接收方私钥
        C - 待解密的密文消息(十六进制字符串)
    返回:
        解密后的明文字符串
    """
    p, a, b, h, G, n = args

    # 计算椭圆曲线坐标的字节长度
    l = ceil(log(p, 2) / 8)
    bytes_l1 = 2 * l + 1  # C1部分的字节长度
    hex_l1 = bytes_l1 * 2  # C1部分的十六进制长度

    # 将密文转换为字节并提取C1部分
    C_bytes = hex_to_bytes(C)
    C1_bytes = C_bytes[0:2 * l + 1]

    # 将C1字节转换为椭圆曲线点并验证在曲线上
    C1 = bytes_to_point(C1_bytes)
    if not on_curve(args, C1):
        raise Exception("在解密算法B1中，取得的C1不在椭圆曲线上！")

    # 计算共享密钥S = [h]C1 (h是余因子)
    S = mult_point(C1, h, p, a)
    if S == 0:
        raise Exception("在解密算法B2中，S是无穷远点！")

    # 计算(x2, y2) = [dB]C1 (用于密钥派生)
    temp = mult_point(C1, dB, p, a)
    x2, y2 = temp[0], temp[1]
    x2_hex, y2_hex = fielde_to_hex(x2), fielde_to_hex(y2)

    # 计算各部分长度
    hex_l3 = 64  # C3部分固定长度(SM3输出为256位=64十六进制字符)
    hex_l2 = len(C) - hex_l1 - hex_l3  # C2部分长度

    # 使用KDF从(x2, y2)派生对称密钥t
    klen = hex_l2 * 4  # 计算需要的密钥长度(位)
    x2_bits, y2_bits = hex_to_bits(x2_hex), hex_to_bits(y2_hex)
    t = KDF(x2_bits + y2_bits, klen)
    if eval('0b' + t) == 0:
        raise Exception("在解密算法B4中，得到的t是全0串！")

    # 将t转换为十六进制并与C2异或得到明文
    t_hex = bits_to_hex(t)
    C2_hex = C[hex_l1: -hex_l3]  # 提取C2部分
    M1 = eval('0x' + C2_hex + '^' + '0x' + t_hex)
    M1_hex = hex(M1)[2:].rjust(hex_l2, '0')  # 转换为固定长度的十六进制

    # 验证消息认证码u = SM3(x2 || M || y2) 是否等于C3
    M1_bits = hex_to_bits(M1_hex)
    cmp_bits = x2_bits + M1_bits + y2_bits
    cmp_bytes = bits_to_bytes(cmp_bits)
    cmp_list = [i for i in cmp_bytes]
    u = sm3.sm3_hash(cmp_list)

    # 提取并比较C3部分
    C3_hex = C[-hex_l3:]
    if u != C3_hex:
        raise Exception("在解密算法B6中，计算的u与C3不同！")

    # 将十六进制明文转换为ASCII字符串
    M_bytes = hex_to_bytes(M1_hex)
    M = str(M_bytes, encoding='ascii')

    return M


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
    xB = eval('0x' + '435B39CC A8F3B508 C1488AFC 67BE491A 0F7BA07E 581A0E48 49A5CF70 628A7E0A'.replace(' ', ''))
    yB = eval('0x' + '75DDBA78 F15FEECB 4C7895E2 C1CDF5FE 01DEBB2C DBADF453 99CCF77B BA076A42'.replace(' ', ''))
    PB = (xB, yB)           # PB是B的公钥
    dB = eval('0x' + '1649AB77 A00637BD 5E2EFE28 3FBF3535 34AA7F7C B89463F2 08DDBC29 20BB0DA0'.replace(' ', ''))
    # dB是B的私钥
    key_B = (PB, dB)
    return key_B

if __name__=='__main__':
    print("SM2椭圆曲线公钥密码算法".center(100, '='))
    print("本算法采用256位素数域上的椭圆曲线。椭圆曲线方程为：")
    print("y^2 = x^3 + ax + b")

    print("下面获取椭圆曲线系统参数")
    args = get_args()
    p, a, b, h, G, n = args
    p, a, b, h, xG, yG, n = tuple(map(lambda a: hex(a)[2:], (p, a, b, h, G[0], G[1], n)))
    print("椭圆曲线系统所在素域的p是：", p)
    print("椭圆曲线系统的参数a是：", a)
    print("椭圆曲线系统的参数b是：", b)
    print("椭圆曲线系统的余因子h是：", h)
    print("椭圆曲线系统的基点G的横坐标xG是：", xG)
    print("椭圆曲线系统的基点G的纵坐标yG是：", yG)

    print("下面获取接收方B的公私钥")
    key_B = get_key()
    PB, dB = key_B
    xB, yB, dB = tuple(map(lambda a: hex(a)[2:], (PB[0], PB[1], dB)))
    print("接收方B的公钥PB的横坐标xB是：", xB)
    print("接收方B的公钥PB的纵坐标yB是：", yB)
    print("接收方B的私钥dB是：", dB)
    print("下面获取明文")
    M = input('请输入要加密的明文(明文应为ascii字符组成的字符串)：')
    print("获取的ascii字符串明文是：", M)

    encrypt_time = time.time()
    C = encry_sm2(args, key_B[0], M)
    encrypt_time = time.time() - encrypt_time
    print("加密时间：", encrypt_time, "s")

    de_time = time.time()
    de_M = decry_sm2(args, key_B[1], C)
    de_time = time.time() - de_time
    print("解密时间：", de_time, "s")

    print("原始明文是：", M)
    print("解密得到的明文是：", de_M)
    if M == de_M:
        print("恭喜您，解密成功！")
    else:
        print("解密失败，请检查算法！")
