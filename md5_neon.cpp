#include "md5_neon.h"
#include <arm_neon.h>
#include <cstring>
#include <cstdint>

//  NEON 辅助函数
static inline uint32x4_t VROTL(uint32x4_t v, int n) {
    return vorrq_u32(vshlq_n_u32(v, n), vshrq_n_u32(v, 32 - n));
}
static inline uint32x4_t VF(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return vorrq_u32(vandq_u32(x, y), vandq_u32(vmvnq_u32(x), z));
}
static inline uint32x4_t VG(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return vorrq_u32(vandq_u32(x, z), vandq_u32(y, vmvnq_u32(z)));
}
static inline uint32x4_t VH(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return veorq_u32(veorq_u32(x, y), z);
}
static inline uint32x4_t VI(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return veorq_u32(y, vorrq_u32(x, vmvnq_u32(z)));
}

static inline void VFF(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d,
                       uint32x4_t x, int s, uint32_t ac) {
    a = vaddq_u32(a, vaddq_u32(vaddq_u32(VF(b, c, d), x), vdupq_n_u32(ac)));
    a = VROTL(a, s);
    a = vaddq_u32(a, b);
}
static inline void VGG(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d,
                       uint32x4_t x, int s, uint32_t ac) {
    a = vaddq_u32(a, vaddq_u32(vaddq_u32(VG(b, c, d), x), vdupq_n_u32(ac)));
    a = VROTL(a, s);
    a = vaddq_u32(a, b);
}
static inline void VHH(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d,
                       uint32x4_t x, int s, uint32_t ac) {
    a = vaddq_u32(a, vaddq_u32(vaddq_u32(VH(b, c, d), x), vdupq_n_u32(ac)));
    a = VROTL(a, s);
    a = vaddq_u32(a, b);
}
static inline void VII(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d,
                       uint32x4_t x, int s, uint32_t ac) {
    a = vaddq_u32(a, vaddq_u32(vaddq_u32(VI(b, c, d), x), vdupq_n_u32(ac)));
    a = VROTL(a, s);
    a = vaddq_u32(a, b);
}

// 字节序反转
static inline uint32x4_t vbswap32q(uint32x4_t v) {
    uint32x4_t b0 = vshlq_n_u32(vandq_u32(v, vdupq_n_u32(0x000000ff)), 24);
    uint32x4_t b1 = vshlq_n_u32(vandq_u32(v, vdupq_n_u32(0x0000ff00)), 8);
    uint32x4_t b2 = vshrq_n_u32(vandq_u32(v, vdupq_n_u32(0x00ff0000)), 8);
    uint32x4_t b3 = vshrq_n_u32(vandq_u32(v, vdupq_n_u32(0xff000000)), 24);
    return vorrq_u32(vorrq_u32(b0, b1), vorrq_u32(b2, b3));
}


//  动态内存 Padding
static Byte *StringProcessDynamic(const std::string &input, int *n_byte) {
    const Byte *blocks = (const Byte *)input.c_str();
    int length = (int)input.length();
    int bitLength = length * 8;

    int paddingBits = bitLength % 512;
    if (paddingBits > 448)       paddingBits = 512 - (paddingBits - 448);
    else if (paddingBits < 448)  paddingBits = 448 - paddingBits;
    else                         paddingBits = 512;

    int paddingBytes = paddingBits / 8;
    int paddedLength = length + paddingBytes + 8;
    Byte *paddedMessage = new Byte[paddedLength];

    memcpy(paddedMessage, blocks, length);
    paddedMessage[length] = 0x80;
    memset(paddedMessage + length + 1, 0, paddingBytes - 1);
    for (int i = 0; i < 8; ++i)
        paddedMessage[length + paddingBytes + i] =
            ((uint64_t)length * 8 >> (i * 8)) & 0xFF;

    *n_byte = paddedLength;
    return paddedMessage;
}


//  NEON 加载 + 转置：从 4 条消息的同一块构造 vx[0..15]
//  msgs 指向当前块的起始地址
static void LoadTransposedBlock(const Byte *block0, const Byte *block1,
                                const Byte *block2, const Byte *block3,
                                uint32x4_t vx[16]) {
    // 前 16 字节转置
    uint32x4_t col0 = vld1q_u32((const uint32_t *)block0);
    uint32x4_t col1 = vld1q_u32((const uint32_t *)block1);
    uint32x4_t col2 = vld1q_u32((const uint32_t *)block2);
    uint32x4_t col3 = vld1q_u32((const uint32_t *)block3);
    uint32x4x2_t t0 = vzipq_u32(col0, col2);
    uint32x4x2_t t1 = vzipq_u32(col1, col3);
    uint32x4x2_t u0 = vzipq_u32(t0.val[0], t1.val[0]);
    uint32x4x2_t u1 = vzipq_u32(t0.val[1], t1.val[1]);
    vx[0] = u0.val[0]; vx[1] = u0.val[1];
    vx[2] = u1.val[0]; vx[3] = u1.val[1];

    // 16-31 字节
    col0 = vld1q_u32((const uint32_t *)(block0 + 16));
    col1 = vld1q_u32((const uint32_t *)(block1 + 16));
    col2 = vld1q_u32((const uint32_t *)(block2 + 16));
    col3 = vld1q_u32((const uint32_t *)(block3 + 16));
    t0 = vzipq_u32(col0, col2); t1 = vzipq_u32(col1, col3);
    u0 = vzipq_u32(t0.val[0], t1.val[0]); u1 = vzipq_u32(t0.val[1], t1.val[1]);
    vx[4] = u0.val[0]; vx[5] = u0.val[1];
    vx[6] = u1.val[0]; vx[7] = u1.val[1];

    // 32-47 字节
    col0 = vld1q_u32((const uint32_t *)(block0 + 32));
    col1 = vld1q_u32((const uint32_t *)(block1 + 32));
    col2 = vld1q_u32((const uint32_t *)(block2 + 32));
    col3 = vld1q_u32((const uint32_t *)(block3 + 32));
    t0 = vzipq_u32(col0, col2); t1 = vzipq_u32(col1, col3);
    u0 = vzipq_u32(t0.val[0], t1.val[0]); u1 = vzipq_u32(t0.val[1], t1.val[1]);
    vx[8] = u0.val[0]; vx[9] = u0.val[1];
    vx[10] = u1.val[0]; vx[11] = u1.val[1];

    // 48-63 字节
    col0 = vld1q_u32((const uint32_t *)(block0 + 48));
    col1 = vld1q_u32((const uint32_t *)(block1 + 48));
    col2 = vld1q_u32((const uint32_t *)(block2 + 48));
    col3 = vld1q_u32((const uint32_t *)(block3 + 48));
    t0 = vzipq_u32(col0, col2); t1 = vzipq_u32(col1, col3);
    u0 = vzipq_u32(t0.val[0], t1.val[0]); u1 = vzipq_u32(t0.val[1], t1.val[1]);
    vx[12] = u0.val[0]; vx[13] = u0.val[1];
    vx[14] = u1.val[0]; vx[15] = u1.val[1];
}

//  MD5Hash_NEON 主体
void MD5Hash_NEON(std::string inputs[4], bit32 states[16]) {
    // 1. 动态分配 padding 缓冲区
    Byte *msgs[4];
    int   lens[4];
    for (int i = 0; i < 4; i++)
        msgs[i] = StringProcessDynamic(inputs[i], &lens[i]);

    // 记录每条消息的原始块数
    int orig_n_blocks[4];
    bool all_equal = true;
    for (int i = 0; i < 4; i++) {
        orig_n_blocks[i] = lens[i] / 64;
        if (i > 0 && lens[i] != lens[0])
            all_equal = false;
    }

    // 计算最大块数，并将短消息扩展到相同长度
    int max_blocks = orig_n_blocks[0];
    for (int i = 1; i < 4; i++)
        if (orig_n_blocks[i] > max_blocks) max_blocks = orig_n_blocks[i];

    for (int i = 0; i < 4; i++) {
        if (orig_n_blocks[i] < max_blocks) {
            Byte *extended = new Byte[max_blocks * 64]();
            memcpy(extended, msgs[i], lens[i]);
            delete[] msgs[i];
            msgs[i] = extended;
        }
    }

    // 2. 初始化 4 路状态
    uint32x4_t va = vdupq_n_u32(0x67452301);
    uint32x4_t vb = vdupq_n_u32(0xefcdab89);
    uint32x4_t vc = vdupq_n_u32(0x98badcfe);
    uint32x4_t vd = vdupq_n_u32(0x10325476);

    // 3. 逐块处理
    for (int blk = 0; blk < max_blocks; blk++) {
        // 加载转置后的消息向量
        uint32x4_t vx[16];
        LoadTransposedBlock(msgs[0] + blk*64, msgs[1] + blk*64,
                            msgs[2] + blk*64, msgs[3] + blk*64, vx);

        uint32x4_t va0 = va, vb0 = vb, vc0 = vc, vd0 = vd;

        /* Round 1 */
        VFF(va, vb, vc, vd, vx[ 0], 7 , 0xd76aa478);
        VFF(vd, va, vb, vc, vx[ 1], 12, 0xe8c7b756);
        VFF(vc, vd, va, vb, vx[ 2], 17, 0x242070db);
        VFF(vb, vc, vd, va, vx[ 3], 22, 0xc1bdceee);
        VFF(va, vb, vc, vd, vx[ 4], 7 , 0xf57c0faf);
        VFF(vd, va, vb, vc, vx[ 5], 12, 0x4787c62a);
        VFF(vc, vd, va, vb, vx[ 6], 17, 0xa8304613);
        VFF(vb, vc, vd, va, vx[ 7], 22, 0xfd469501);
        VFF(va, vb, vc, vd, vx[ 8], 7 , 0x698098d8);
        VFF(vd, va, vb, vc, vx[ 9], 12, 0x8b44f7af);
        VFF(vc, vd, va, vb, vx[10], 17, 0xffff5bb1);
        VFF(vb, vc, vd, va, vx[11], 22, 0x895cd7be);
        VFF(va, vb, vc, vd, vx[12], 7 , 0x6b901122);
        VFF(vd, va, vb, vc, vx[13], 12, 0xfd987193);
        VFF(vc, vd, va, vb, vx[14], 17, 0xa679438e);
        VFF(vb, vc, vd, va, vx[15], 22, 0x49b40821);

        /* Round 2 */
        VGG(va, vb, vc, vd, vx[ 1], 5 , 0xf61e2562);
        VGG(vd, va, vb, vc, vx[ 6], 9 , 0xc040b340);
        VGG(vc, vd, va, vb, vx[11], 14, 0x265e5a51);
        VGG(vb, vc, vd, va, vx[ 0], 20, 0xe9b6c7aa);
        VGG(va, vb, vc, vd, vx[ 5], 5 , 0xd62f105d);
        VGG(vd, va, vb, vc, vx[10], 9 , 0x02441453);
        VGG(vc, vd, va, vb, vx[15], 14, 0xd8a1e681);
        VGG(vb, vc, vd, va, vx[ 4], 20, 0xe7d3fbc8);
        VGG(va, vb, vc, vd, vx[ 9], 5 , 0x21e1cde6);
        VGG(vd, va, vb, vc, vx[14], 9 , 0xc33707d6);
        VGG(vc, vd, va, vb, vx[ 3], 14, 0xf4d50d87);
        VGG(vb, vc, vd, va, vx[ 8], 20, 0x455a14ed);
        VGG(va, vb, vc, vd, vx[13], 5 , 0xa9e3e905);
        VGG(vd, va, vb, vc, vx[ 2], 9 , 0xfcefa3f8);
        VGG(vc, vd, va, vb, vx[ 7], 14, 0x676f02d9);
        VGG(vb, vc, vd, va, vx[12], 20, 0x8d2a4c8a);

        /* Round 3 */
        VHH(va, vb, vc, vd, vx[ 5], 4 , 0xfffa3942);
        VHH(vd, va, vb, vc, vx[ 8], 11, 0x8771f681);
        VHH(vc, vd, va, vb, vx[11], 16, 0x6d9d6122);
        VHH(vb, vc, vd, va, vx[14], 23, 0xfde5380c);
        VHH(va, vb, vc, vd, vx[ 1], 4 , 0xa4beea44);
        VHH(vd, va, vb, vc, vx[ 4], 11, 0x4bdecfa9);
        VHH(vc, vd, va, vb, vx[ 7], 16, 0xf6bb4b60);
        VHH(vb, vc, vd, va, vx[10], 23, 0xbebfbc70);
        VHH(va, vb, vc, vd, vx[13], 4 , 0x289b7ec6);
        VHH(vd, va, vb, vc, vx[ 0], 11, 0xeaa127fa);
        VHH(vc, vd, va, vb, vx[ 3], 16, 0xd4ef3085);
        VHH(vb, vc, vd, va, vx[ 6], 23, 0x04881d05);
        VHH(va, vb, vc, vd, vx[ 9], 4 , 0xd9d4d039);
        VHH(vd, va, vb, vc, vx[12], 11, 0xe6db99e5);
        VHH(vc, vd, va, vb, vx[15], 16, 0x1fa27cf8);
        VHH(vb, vc, vd, va, vx[ 2], 23, 0xc4ac5665);

        /* Round 4 */
        VII(va, vb, vc, vd, vx[ 0], 6 , 0xf4292244);
        VII(vd, va, vb, vc, vx[ 7], 10, 0x432aff97);
        VII(vc, vd, va, vb, vx[14], 15, 0xab9423a7);
        VII(vb, vc, vd, va, vx[ 5], 21, 0xfc93a039);
        VII(va, vb, vc, vd, vx[12], 6 , 0x655b59c3);
        VII(vd, va, vb, vc, vx[ 3], 10, 0x8f0ccc92);
        VII(vc, vd, va, vb, vx[10], 15, 0xffeff47d);
        VII(vb, vc, vd, va, vx[ 1], 21, 0x85845dd1);
        VII(va, vb, vc, vd, vx[ 8], 6 , 0x6fa87e4f);
        VII(vd, va, vb, vc, vx[15], 10, 0xfe2ce6e0);
        VII(vc, vd, va, vb, vx[ 6], 15, 0xa3014314);
        VII(vb, vc, vd, va, vx[13], 21, 0x4e0811a1);
        VII(va, vb, vc, vd, vx[ 4], 6 , 0xf7537e82);
        VII(vd, va, vb, vc, vx[11], 10, 0xbd3af235);
        VII(vc, vd, va, vb, vx[ 2], 15, 0x2ad7d2bb);
        VII(vb, vc, vd, va, vx[ 9], 21, 0xeb86d391);

        // 状态累加
        if (all_equal) {
            va = vaddq_u32(va, va0);
            vb = vaddq_u32(vb, vb0);
            vc = vaddq_u32(vc, vc0);
            vd = vaddq_u32(vd, vd0);
        } else {
            uint32_t m[4];
            for (int l = 0; l < 4; l++)
                m[l] = (blk < orig_n_blocks[l]) ? 0xFFFFFFFF : 0;
            uint32x4_t mask = vld1q_u32(m);
            va = vbslq_u32(mask, vaddq_u32(va, va0), va0);
            vb = vbslq_u32(mask, vaddq_u32(vb, vb0), vb0);
            vc = vbslq_u32(mask, vaddq_u32(vc, vc0), vc0);
            vd = vbslq_u32(mask, vaddq_u32(vd, vd0), vd0);
        }
    }

    // 4. 字节序反转
    va = vbswap32q(va);
    vb = vbswap32q(vb);
    vc = vbswap32q(vc);
    vd = vbswap32q(vd);

    // 5. 写回结果
    uint32_t tmp[4][4];
    vst1q_u32(tmp[0], va);
    vst1q_u32(tmp[1], vb);
    vst1q_u32(tmp[2], vc);
    vst1q_u32(tmp[3], vd);
    for (int i = 0; i < 4; i++) {
        states[i*4 + 0] = tmp[0][i];
        states[i*4 + 1] = tmp[1][i];
        states[i*4 + 2] = tmp[2][i];
        states[i*4 + 3] = tmp[3][i];
    }

    // 6. 释放动态内存
    for (int i = 0; i < 4; i++)
        delete[] msgs[i];
}