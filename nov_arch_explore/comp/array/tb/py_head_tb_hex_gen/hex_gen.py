import numpy as np
GQA = 1
EMBD_SIZE = 512
SEQ_LENGTH = 1024
N_HEADS = 8
NUM_USER = 4


# input_x gen
users,rows, cols = NUM_USER, SEQ_LENGTH, EMBD_SIZE
matrix = np.random.randint(-128, 128, size=(users,rows, cols), dtype=np.int8)
hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(matrix)


users,rows, cols = NUM_USER, SEQ_LENGTH, EMBD_SIZE
with open('input_x.hex', 'w') as f:
    for user in hex_matrix:
        for row in user:
            for hex_value in row:
                f.write(hex_value + '\n')
print("input_x.hex")



QW_matrix = np.random.randint(-128, 128, size=(N_HEADS, EMBD_SIZE, EMBD_SIZE//N_HEADS), dtype=np.int8)
QW_hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(QW_matrix)

if GQA == 0:
    KW_matrix = np.random.randint(-128, 128, size=(N_HEADS, EMBD_SIZE, EMBD_SIZE//N_HEADS), dtype=np.int8)
    KW_hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(KW_matrix)

    VW_matrix = np.random.randint(-128, 128, size=(N_HEADS, EMBD_SIZE, EMBD_SIZE//N_HEADS), dtype=np.int8)
    VW_hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(VW_matrix)
else:
    # 生成 N_HEADS//2 个随机矩阵，只为偶数编号的 heads 生成权重
    KW_matrix = np.random.randint(-128, 128, size=(N_HEADS//2, EMBD_SIZE, EMBD_SIZE//N_HEADS), dtype=np.int8)
    VW_matrix = np.random.randint(-128, 128, size=(N_HEADS//2, EMBD_SIZE, EMBD_SIZE//N_HEADS), dtype=np.int8)

    # 将偶数编号的权重复制给相邻的奇数编号 heads
    KW_matrix = np.repeat(KW_matrix, 2, axis=0)
    VW_matrix = np.repeat(VW_matrix, 2, axis=0)

    # 将矩阵元素转换为十六进制
    KW_hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(KW_matrix)
    VW_hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(VW_matrix)

PROJW_matrix = np.random.randint(-128, 128, size=(N_HEADS, EMBD_SIZE//N_HEADS, EMBD_SIZE), dtype=np.int8)
PROJW_hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(PROJW_matrix)

FFN0W_matrix = np.random.randint(-128, 128, size=(N_HEADS, EMBD_SIZE, 4*EMBD_SIZE//N_HEADS), dtype=np.int8)
FFN0W_hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(FFN0W_matrix)

FFN1W_matrix = np.random.randint(-128, 128, size=(N_HEADS, 4*EMBD_SIZE//N_HEADS, EMBD_SIZE), dtype=np.int8)
FFN1W_hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(FFN1W_matrix)

with open(f'Q_weights.hex', 'w') as f:
    for head in QW_hex_matrix:
        for row in head:
            for hex_value in row:
                f.write(hex_value + '\n')
print(f"Q_weights.hex")

with open(f'K_weights.hex', 'w') as f:
    for head in KW_hex_matrix:
        for row in head:
            for hex_value in row:
                f.write(hex_value + '\n')
print(f"K_weights.hex")

with open(f'V_weights.hex', 'w') as f:
    for head in VW_hex_matrix:
        for row in head:
            for hex_value in row:
                f.write(hex_value + '\n')
print(f"V_weights.hex")

with open(f'PROJ_weights.hex', 'w') as f:
    for head in PROJW_hex_matrix:
        for row in head:
            for hex_value in row:
                f.write(hex_value + '\n')
print(f"PROJ_weights.hex")

with open(f'FFN0_weights.hex', 'w') as f:
    for head in FFN0W_hex_matrix:
        for row in head:
            for hex_value in row:
                f.write(hex_value + '\n')
print(f"FFN0_weights.hex")

with open(f'FFN1_weights.hex', 'w') as f:
    for head in FFN1W_hex_matrix:
        for row in head:
            for hex_value in row:
                f.write(hex_value + '\n')
print(f"FFN1_weights.hex")
