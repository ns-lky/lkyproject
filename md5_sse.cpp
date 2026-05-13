#include "md5_sse.h"
#include <immintrin.h>   // SSE2 / SSE4.1
#include <cstring>
#include <cstdint>

//  SSE 辅助函数

// 循环左移
static inline __m128i VROTL(__m128i v, int n) {
    return _mm_or_si128(_mm_slli_epi32(v, n), _mm_srli_epi32(v, 32 - n));
}

// F(x,y,z) = (x & y) | (~x & z)
static inline __m128i VF(__m128i x, __m128i y, __m128i z) {
    return _mm_or_si128(_mm_and_si128(x, y), _mm_andnot_si128(x, z));
}

// G(x,y,z) = (x & z) | (y & ~z)
static inline __m128i VG(__m128i x, __m128i y, __m128i z) {
    return _mm_or_si128(_mm_and_si128(x, z), _mm_andnot_si128(z, y));
}

// H(x,y,z) = x ^ y ^ z
static inline __m128i VH(__m128i x, __m128i y, __m128i z) {
    return _mm_xor_si128(_mm_xor_si128(x, y), z);
}

// I(x,y,z) = y ^ (x | ~z)
static inline __m128i VI(__m128i x, __m128i y, __m128i z) {
    return _mm_xor_si128(y, _mm_or_si128(x, _mm_xor_si128(z, _mm_set1_epi32(-1))));
}

static inline void VFF(__m128i &a, __m128i b, __m128i c, __m128i d,
                       __m128i x, int s, uint32_t ac) {
    a = _mm_add_epi32(a, _mm_add_epi32(_mm_add_epi32(VF(b, c, d), x),
                                        _mm_set1_epi32((int)ac)));
    a = VROTL(a, s);
    a = _mm_add_epi32(a, b);
}

static inline void VGG(__m128i &a, __m128i b, __m128i c, __m128i d,
                       __m128i x, int s, uint32_t ac) {
    a = _mm_add_epi32(a, _mm_add_epi32(_mm_add_epi32(VG(b, c, d), x),
                                        _mm_set1_epi32((int)ac)));
    a = VROTL(a, s);
    a = _mm_add_epi32(a, b);
}

static inline void VHH(__m128i &a, __m128i b, __m128i c, __m128i d,
                       __m128i x, int s, uint32_t ac) {
    a = _mm_add_epi32(a, _mm_add_epi32(_mm_add_epi32(VH(b, c, d), x),
                                        _mm_set1_epi32((int)ac)));
    a = VROTL(a, s);
    a = _mm_add_epi32(a, b);
}

static inline void VII(__m128i &a, __m128i b, __m128i c, __m128i d,
                       __m128i x, int s, uint32_t ac) {
    a = _mm_add_epi32(a, _mm_add_epi32(_mm_add_epi32(VI(b, c, d), x),
                                        _mm_set1_epi32((int)ac)));
    a = VROTL(a, s);
    a = _mm_add_epi32(a, b);
}

//  字节序反转
static inline __m128i vbswap32q(__m128i v) {
    // 每个 32-bit 元素：将 byte 顺序从 [B3 B2 B1 B0] 变为 [B0 B1 B2 B3]
    __m128i b0 = _mm_slli_epi32(_mm_and_si128(v, _mm_set1_epi32(0x000000ff)), 24);
    __m128i b1 = _mm_slli_epi32(_mm_and_si128(v, _mm_set1_epi32(0x0000ff00)), 8);
    __m128i b2 = _mm_srli_epi32(_mm_and_si128(v, _mm_set1_epi32(0x00ff0000)), 8);
    __m128i b3 = _mm_srli_epi32(_mm_and_si128(v, _mm_set1_epi32((int)0xff000000u)), 24);
    return _mm_or_si128(_mm_or_si128(b0, b1), _mm_or_si128(b2, b3));
}


//  动态 Padding
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

//  SSE 加载 + 转置：从 4 条消息的同一块构造 vx[0..15]
static void LoadTransposedBlock(const Byte *block0, const Byte *block1,
                                const Byte *block2, const Byte *block3,
                                __m128i vx[16]) {
    // 每次加载 4 个 lane 的同位置 128-bit 块，然后做 4×4 转置
    // 转置方法：先按 32-bit 解交错，得到 [A0A2 B0B2 A1A3 B1B3] 形式，
    //           再做一层解交错完成 4×4 转置

    auto transpose4x4 = [&](__m128i c0, __m128i c1, __m128i c2, __m128i c3,
                             __m128i &r0, __m128i &r1, __m128i &r2, __m128i &r3) {
        // step1: interleave 32-bit words between pairs
        __m128i t0 = _mm_unpacklo_epi32(c0, c1); // [A0 B0 A1 B1]
        __m128i t1 = _mm_unpackhi_epi32(c0, c1); // [A2 B2 A3 B3]
        __m128i t2 = _mm_unpacklo_epi32(c2, c3); // [C0 D0 C1 D1]
        __m128i t3 = _mm_unpackhi_epi32(c2, c3); // [C2 D2 C3 D3]
        // step2: interleave 64-bit words
        r0 = _mm_unpacklo_epi64(t0, t2); // [A0 B0 C0 D0]
        r1 = _mm_unpackhi_epi64(t0, t2); // [A1 B1 C1 D1]
        r2 = _mm_unpacklo_epi64(t1, t3); // [A2 B2 C2 D2]
        r3 = _mm_unpackhi_epi64(t1, t3); // [A3 B3 C3 D3]
    };

    // 前 16 字节
    __m128i col0 = _mm_loadu_si128((const __m128i *)block0);
    __m128i col1 = _mm_loadu_si128((const __m128i *)block1);
    __m128i col2 = _mm_loadu_si128((const __m128i *)block2);
    __m128i col3 = _mm_loadu_si128((const __m128i *)block3);
    transpose4x4(col0, col1, col2, col3, vx[0], vx[1], vx[2], vx[3]);

    // 16-31 字节
    col0 = _mm_loadu_si128((const __m128i *)(block0 + 16));
    col1 = _mm_loadu_si128((const __m128i *)(block1 + 16));
    col2 = _mm_loadu_si128((const __m128i *)(block2 + 16));
    col3 = _mm_loadu_si128((const __m128i *)(block3 + 16));
    transpose4x4(col0, col1, col2, col3, vx[4], vx[5], vx[6], vx[7]);

    // 32-47 字节
    col0 = _mm_loadu_si128((const __m128i *)(block0 + 32));
    col1 = _mm_loadu_si128((const __m128i *)(block1 + 32));
    col2 = _mm_loadu_si128((const __m128i *)(block2 + 32));
    col3 = _mm_loadu_si128((const __m128i *)(block3 + 32));
    transpose4x4(col0, col1, col2, col3, vx[8], vx[9], vx[10], vx[11]);

    // 48-63 字节
    col0 = _mm_loadu_si128((const __m128i *)(block0 + 48));
    col1 = _mm_loadu_si128((const __m128i *)(block1 + 48));
    col2 = _mm_loadu_si128((const __m128i *)(block2 + 48));
    col3 = _mm_loadu_si128((const __m128i *)(block3 + 48));
    transpose4x4(col0, col1, col2, col3, vx[12], vx[13], vx[14], vx[15]);
}


//  MD5Hash_SSE 主体
void MD5Hash_SSE(std::string inputs[4], bit32 states[16]) {
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
    __m128i va = _mm_set1_epi32((int)0x67452301);
    __m128i vb = _mm_set1_epi32((int)0xefcdab89);
    __m128i vc = _mm_set1_epi32((int)0x98badcfe);
    __m128i vd = _mm_set1_epi32((int)0x10325476);

    // 3. 逐块处理
    for (int blk = 0; blk < max_blocks; blk++) {
        __m128i vx[16];
        LoadTransposedBlock(msgs[0] + blk*64, msgs[1] + blk*64,
                            msgs[2] + blk*64, msgs[3] + blk*64, vx);

        __m128i va0 = va, vb0 = vb, vc0 = vc, vd0 = vd;

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
            va = _mm_add_epi32(va, va0);
            vb = _mm_add_epi32(vb, vb0);
            vc = _mm_add_epi32(vc, vc0);
            vd = _mm_add_epi32(vd, vd0);
        } else {
            // 对长度不等时，只累加已处理块对应的 lane
            uint32_t m[4];
            for (int l = 0; l < 4; l++)
                m[l] = (blk < orig_n_blocks[l]) ? 0xFFFFFFFFu : 0u;
            __m128i mask = _mm_loadu_si128((const __m128i *)m);
            // 用 mask 选择：mask 为全 1 的 lane 取 (reg + reg0)，否则取 reg0
            va = _mm_or_si128(_mm_and_si128(mask, _mm_add_epi32(va, va0)),
                              _mm_andnot_si128(mask, va0));
            vb = _mm_or_si128(_mm_and_si128(mask, _mm_add_epi32(vb, vb0)),
                              _mm_andnot_si128(mask, vb0));
            vc = _mm_or_si128(_mm_and_si128(mask, _mm_add_epi32(vc, vc0)),
                              _mm_andnot_si128(mask, vc0));
            vd = _mm_or_si128(_mm_and_si128(mask, _mm_add_epi32(vd, vd0)),
                              _mm_andnot_si128(mask, vd0));
        }
    }

    // 4. 字节序反转
    va = vbswap32q(va);
    vb = vbswap32q(vb);
    vc = vbswap32q(vc);
    vd = vbswap32q(vd);

    // 5. 写回结果
    uint32_t tmp[4][4];
    _mm_storeu_si128((__m128i *)tmp[0], va);
    _mm_storeu_si128((__m128i *)tmp[1], vb);
    _mm_storeu_si128((__m128i *)tmp[2], vc);
    _mm_storeu_si128((__m128i *)tmp[3], vd);
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
