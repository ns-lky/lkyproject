#pragma once
#include "md5.h"
#include <string>

/**
 * MD5Hash_SSE: 一次并行处理 4 个字符串，基于 x86 SSE2/SSE4.1 128-bit 指令集
 * @param inputs   4 个输入字符串
 * @param states   输出，4 组 MD5 结果，每组 4 个 bit32，共 16 个元素
 *                 states[i*4 .. i*4+3] 对应 inputs[i] 的结果
 */
void MD5Hash_SSE(std::string inputs[4], bit32 states[16]);
