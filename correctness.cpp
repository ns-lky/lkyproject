#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include "md5_neon.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp md5_neon.cpp -o test.exe


// state[4] 转 hex 字符串
static string stateToHex(const bit32 *state)
{
    stringstream ss;
    for (int i = 0; i < 4; i++)
        ss << std::setw(8) << std::setfill('0') << hex << state[i];
    return ss.str();
}


// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    // 原始串行测试（长字符串）
    bit32 state[4];
    MD5Hash("bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva", state);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    cout << endl;

    // NEON 并行版本验证 
    // 测试用例：覆盖短/中/长字符串，含纯数字、字母、混合
    // 每次取连续 4 条作为一组送入 NEON 版
    const string all_cases[] = {
        // 组 0
        "123456",
        "password",
        "12345678",
        "qwerty",
        // 组 1
        "123456789",
        "12345",
        "1234",
        "111111",
        // 组 2（长字符串压力测试，含跨 block 情形）
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",   // 64 bytes
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva",
    };
    const int N = sizeof(all_cases) / sizeof(all_cases[0]);
    int groups = N / 4;

    int pass = 0, fail = 0;

    cout << "=== MD5Hash_NEON correctness test ===" << endl;

    for (int g = 0; g < groups; g++)
    {
        string inputs[4];
        for (int i = 0; i < 4; i++)
            inputs[i] = all_cases[g * 4 + i];

        // 串行参考结果
        bit32 ref_states[4][4];
        for (int i = 0; i < 4; i++)
            MD5Hash(inputs[i], ref_states[i]);

        // NEON 并行结果
        bit32 neon_states[16];
        MD5Hash_NEON(inputs, neon_states);

        // 逐条对比
        for (int i = 0; i < 4; i++)
        {
            string ref_hex  = stateToHex(ref_states[i]);
            string neon_hex = stateToHex(neon_states + i * 4);

            if (ref_hex == neon_hex)
            {
                cout << "[PASS] \"" << inputs[i] << "\"" << endl;
                cout << "  serial: " << ref_hex  << endl;
                cout << "  NEON  : " << neon_hex << endl;
                pass++;
            }
            else
            {
                cout << "[FAIL] \"" << inputs[i] << "\"" << endl;
                cout << "  serial: " << ref_hex  << endl;
                cout << "  NEON  : " << neon_hex << endl;
                fail++;
            }
        }
        cout << endl;
    }

    cout << "========================================" << endl;
    cout << "Result: " << pass << " passed, " << fail << " failed." << endl;

    return (fail == 0) ? 0 : 1;
}
