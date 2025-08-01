import numpy as np
EMBD_SIZE = 512
SEQ_LENGTH = 512
N_HEADS = 8
HEAD_CORE_NUM = 16
HEAD_DIM = EMBD_SIZE//N_HEADS
QKV_WEIGHT_COLS_PER_CORE = HEAD_DIM//HEAD_CORE_NUM
NUM_USER = 4


# input_x gen
users,rows, cols = NUM_USER, SEQ_LENGTH, EMBD_SIZE
matrix = np.random.randint(-128, 128, size=(users,rows, cols), dtype=np.int8)
hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(matrix)





rows, cols = EMBD_SIZE, EMBD_SIZE
QW_matrix = np.random.randint(-128, 128, size=(rows, cols), dtype=np.int8)
QW_hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(QW_matrix)

KW_matrix = np.random.randint(-128, 128, size=(rows, cols), dtype=np.int8)
KW_hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(KW_matrix)

VW_matrix = np.random.randint(-128, 128, size=(rows, cols), dtype=np.int8)
VW_hex_matrix = np.vectorize(lambda x: format(x & 0xFF, '02X'))(VW_matrix)






users,rows, cols = NUM_USER, SEQ_LENGTH, EMBD_SIZE
with open('input_x.hex', 'w') as f:
    for user in hex_matrix:
        for row in user:
            for hex_value in row:
                f.write(hex_value + '\n')
print("input_x.hex")





rows, cols = EMBD_SIZE, EMBD_SIZE
with open('Q_weights_one_head.hex', 'w') as f:
    for row in QW_hex_matrix:
        for hex_value in row:
            f.write(hex_value + '\n')
print("Q_weights_one_head.hex")

with open('K_weights_one_head.hex', 'w') as f:
    for row in KW_hex_matrix:
        for hex_value in row:
            f.write(hex_value + '\n')
print("K_weights_one_head.hex")

with open('V_weights_one_head.hex', 'w') as f:
    for row in VW_hex_matrix:
        for hex_value in row:
            f.write(hex_value + '\n')
print("V_weights_one_head.hex")