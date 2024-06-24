#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "scalar.hpp"
#include "config.hpp"
#include "utils.hpp"
#include "zklog.hpp"

/* Global scalar variables */

mpz_class ScalarMask4   ("F", 16);
mpz_class ScalarMask8   ("FF", 16);
mpz_class ScalarMask16  ("FFFF", 16);
mpz_class ScalarMask20  ("FFFFF", 16);
mpz_class ScalarMask32  ("FFFFFFFF", 16);
mpz_class ScalarMask64  ("FFFFFFFFFFFFFFFF", 16);
mpz_class ScalarMask160 ("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
mpz_class ScalarMask256 ("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
mpz_class ScalarTwoTo8  ("100", 16);
mpz_class ScalarTwoTo16 ("10000", 16);
mpz_class ScalarTwoTo18 ("40000", 16);
mpz_class ScalarTwoTo32 ("100000000", 16);
mpz_class ScalarTwoTo64 ("10000000000000000", 16);
mpz_class ScalarTwoTo128("100000000000000000000000000000000", 16);
mpz_class ScalarTwoTo192("1000000000000000000000000000000000000000000000000", 16);
mpz_class ScalarTwoTo254("4000000000000000000000000000000000000000000000000000000000000000", 16);
mpz_class ScalarTwoTo255("8000000000000000000000000000000000000000000000000000000000000000", 16);
mpz_class ScalarTwoTo256("10000000000000000000000000000000000000000000000000000000000000000", 16);
mpz_class ScalarTwoTo257("20000000000000000000000000000000000000000000000000000000000000000", 16);
mpz_class ScalarTwoTo258("40000000000000000000000000000000000000000000000000000000000000000", 16);
mpz_class ScalarTwoTo259("80000000000000000000000000000000000000000000000000000000000000000", 16);

mpz_class ScalarZero    ("0", 16);
mpz_class ScalarOne     ("1", 16);

/* Normalized strings */

string Remove0xIfPresent(const string &s)
{
    if ( (s.size() >= 2) && (s.at(1) == 'x') && (s.at(0) == '0') ) return s.substr(2);
    return s;
}

void Remove0xIfPresentNoCopy(string &s)
{
    if ( (s.size() >= 2) && (s.at(1) == 'x') && (s.at(0) == '0') ) s = s.substr(2);
}

string Add0xIfMissing(const string &s)
{
    if ( (s.size() >= 2) && (s.at(1) == 'x') && (s.at(0) == '0') ) return s;
    return "0x" + s;
}


// A set of strings with zeros is available in memory for performance reasons
string sZeros[65] = {
    "",
    "0",
    "00",
    "000",
    "0000",
    "00000",
    "000000",
    "0000000",
    "00000000",
    "000000000",
    "0000000000",
    "00000000000",
    "000000000000",
    "0000000000000",
    "00000000000000",
    "000000000000000",
    "0000000000000000",
    "00000000000000000",
    "000000000000000000",
    "0000000000000000000",
    "00000000000000000000",
    "000000000000000000000",
    "0000000000000000000000",
    "00000000000000000000000",
    "000000000000000000000000",
    "0000000000000000000000000",
    "00000000000000000000000000",
    "000000000000000000000000000",
    "0000000000000000000000000000",
    "00000000000000000000000000000",
    "000000000000000000000000000000",
    "0000000000000000000000000000000",
    "00000000000000000000000000000000",
    "000000000000000000000000000000000",
    "0000000000000000000000000000000000",
    "00000000000000000000000000000000000",
    "000000000000000000000000000000000000",
    "0000000000000000000000000000000000000",
    "00000000000000000000000000000000000000",
    "000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000"
};

string PrependZeros (const string &s, uint64_t n)
{
    // Check that n is not too big
    if (n > 64)
    {
        zklog.error("PrependZeros() called with an that is too big n=" + to_string(n));
        exitProcess();
    }
    // Check that string size is not too big
    uint64_t stringSize = s.size();
    if ( (stringSize > n) || (stringSize > 64) )
    {
        zklog.error("PrependZeros() called with a string with too large s.size=" + to_string(stringSize) + " n=" + to_string(n));
        exitProcess();
    }

    // Prepend zeros if needed
    if (stringSize < n) return sZeros[n-stringSize] + s;

    return s;
}

void PrependZerosNoCopy (string &s, uint64_t n)
{
    // Check that n is not too big
    if (n > 64)
    {
        zklog.error("PrependZerosNoCopy() called with an n that is too big n=" + to_string(n));
        exitProcess();
    }
    // Check that string size is not too big
    uint64_t stringSize = s.size();
    if ( (stringSize > n) || (stringSize > 64) )
    {
        zklog.error("PrependZerosNoCopy() called with a string with too large s.size=" + to_string(stringSize) + " n=" + to_string(n));
        exitProcess();
    }

    // Prepend zeros if needed
    if (stringSize < n) s = sZeros[n-stringSize] + s;
}

string NormalizeToNFormat (const string &s, uint64_t n)
{
    return PrependZeros(Remove0xIfPresent(s), n);
}

string NormalizeTo0xNFormat (const string &s, uint64_t n)
{
    return "0x" + NormalizeToNFormat(s, n);
}

string stringToLower (const string &s)
{
    string result = s;
    transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

bool stringIsHex (const string &s)
{
    for (uint64_t i=0; i<s.size(); i++)
    {
        if (!charIsHex(s.at(i))) return false;
    }
    return true;
}

bool stringIsDec (const string &s)
{
    for (uint64_t i=0; i<s.size(); i++)
    {
        if (!charIsDec(s.at(i))) return false;
    }
    return true;
}

bool stringIs0xHex (const string &s)
{
    if (s.size() < 2)
    {
        return false;
    }
    if (s.at(0) != '0')
    {
        return false;
    }
    if (s.at(1) != 'x')
    {
        return false;
    }
    for (uint64_t i=2; i<s.size(); i++)
    {
        if (!charIsHex(s.at(i))) return false;
    }
    return true;
}

/* Byte to/from char conversion */

uint8_t char2byte (char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    zklog.error("char2byte() called with an invalid, non-hex char: " + to_string(c));
    exitProcess();
    return 0;
}

char byte2char (uint8_t b)
{
    if (b < 10) return '0' + b;
    if (b < 16) return 'a' + b - 10;
    zklog.error("byte2char() called with an invalid byte: " + to_string(b));
    exitProcess();
    return 0;
}

string byte2string(uint8_t b)
{
    string result;
    result.push_back(byte2char(b >> 4));
    result.push_back(byte2char(b & 0x0F));
    return result;
}

/* Strint to/from byte array conversion
   s must be even sized, and must not include the leading "0x"
   pData buffer must be big enough to store converted data */

uint64_t string2ba (const string &os, uint8_t *pData, uint64_t &dataSize)
{
    string s = Remove0xIfPresent(os);

    if (s.size()%2 != 0)
    {
        s = "0" + s;
    }

    uint64_t dsize = s.size()/2;
    if (dsize > dataSize)
    {
        zklog.error("string2ba() called with a too short buffer: " + to_string(dsize) + ">" + to_string(dataSize));
        exitProcess();
    }

    const char *p = s.c_str();
    for (uint64_t i=0; i<dsize; i++)
    {
        pData[i] = char2byte(p[2*i])*16 + char2byte(p[2*i + 1]);
    }
    return dsize;
}

void string2ba (const string &textString, string &baString)
{
    baString.clear();

    string s = Remove0xIfPresent(textString);

    if (s.size()%2 != 0)
    {
        s = "0" + s;
    }

    uint64_t dsize = s.size()/2;

    const char *p = s.c_str();
    for (uint64_t i=0; i<dsize; i++)
    {
        uint8_t aux = char2byte(p[2*i])*16 + char2byte(p[2*i + 1]);
        baString.push_back(aux);
    }
}

string string2ba (const string &textString)
{
    string result;
    string2ba(textString, result);
    return result;
}

void string2ba(const string os, vector<uint8_t> &data)
{
    string s = Remove0xIfPresent(os);

    if (s.size()%2 != 0)
    {
        s = "0" + s;
    }

    uint64_t dsize = s.size()/2;
    const char *p = s.c_str();
    for (uint64_t i=0; i<dsize; i++)
    {
        data.push_back(char2byte(p[2*i])*16 + char2byte(p[2*i + 1]));
    }
}

void ba2string (string &s, const uint8_t *pData, uint64_t dataSize)
{
    s = "";
    for (uint64_t i=0; i<dataSize; i++)
    {
        s.append(1, byte2char(pData[i] >> 4));
        s.append(1, byte2char(pData[i] & 0x0F));
    }
}

string ba2string (const uint8_t *pData, uint64_t dataSize)
{
    string result;
    ba2string(result, pData, dataSize);
    return result;
}

void ba2string (const string &baString, string &textString)
{
    ba2string(textString, (const uint8_t *)baString.c_str(), baString.size());
}

string ba2string (const string &baString)
{
    string result;
    ba2string(result, (const uint8_t *)baString.c_str(), baString.size());
    return result;
}

void ba2ba (const string &baString, vector<uint8_t> (&baVector))
{
    baVector.clear();
    baVector.reserve(baString.size());
    for (uint64_t i=0; i<baString.size(); i++)
    {
        uint8_t aux = (uint8_t)baString[i];
        baVector.emplace_back(aux);
    }
}

void ba2ba (const vector<uint8_t> (&baVector), string &baString)
{
    baString.clear();
    baString.reserve(baVector.size());
    for (uint64_t i=0; i<baVector.size(); i++)
    {
        baString.append(1, baVector[i]);
    }
}

void ba2ba (string &baString, const uint64_t ba)
{
    baString = "";
    for (uint64_t i=0; i<8; i++)
    {
        uint8_t byte = (ba >> (56 - i*8));
        baString.append(1, byte);
    }
}

uint64_t ba2ba (const string &baString)
{
    if (baString.size() != 8)
    {
        zklog.error("ba2ba() found invalid baString.size()=" + to_string(baString.size()) + "!=2");
        exitProcess();
    }
    uint64_t result;
    result = (uint64_t(uint8_t(baString[0]))<<56) |
             (uint64_t(uint8_t(baString[1]))<<48) |
             (uint64_t(uint8_t(baString[2]))<<40) |
             (uint64_t(uint8_t(baString[3]))<<32) |
             (uint64_t(uint8_t(baString[4]))<<24) |
             (uint64_t(uint8_t(baString[5]))<<16) |
             (uint64_t(uint8_t(baString[6]))<< 8) |
             (uint64_t(uint8_t(baString[7]))    );
    return result;
}

/* Byte array of exactly 2 bytes conversion */

void ba2u16 (const uint8_t *pData, uint16_t &n)
{
    n = pData[0]*256 + pData[1];
}

void ba2u32 (const uint8_t *pData, uint32_t &n)
{
    n = uint32_t(pData[0])*256*256*256 + uint32_t(pData[1])*256*256 + uint32_t(pData[2])*256 + uint32_t(pData[3]);
}

void ba2scalar (const uint8_t *pData, uint64_t dataSize, mpz_class &s)
{
    s = 0;
    for (uint64_t i=0; i<dataSize; i++)
    {
        s *= ScalarTwoTo8;
        s += pData[i];
    }
}

/* Scalar to byte array conversion (up to dataSize bytes) */

void scalar2ba (uint8_t *pData, uint64_t &dataSize, mpz_class s)
{
    uint64_t i=0;
    for (; i<dataSize; i++)
    {
        // Shift left 1B the byte array content
        for (uint64_t j=i; j>0; j--) pData[j] = pData[j-1];

        // Add the next byte to the byte array
        mpz_class auxScalar = s & ScalarMask8;
        pData[0] = auxScalar.get_ui();

        // Shift right 1B the scalar content
        s = s >> 8;

        // When we run out of significant bytes, break
        if (s == ScalarZero) break;
    }
    if (s != ScalarZero)
    {
        zklog.error("scalar2ba() run out of buffer of " + to_string(dataSize) + " bytes");
        exitProcess();
    }
    dataSize = i+1;
}

void scalar2ba16(uint64_t *pData, uint64_t &dataSize, mpz_class s)
{
    memset(pData, 0, dataSize*sizeof(uint64_t));
    uint64_t i=0;
    for (; i<dataSize; i++)
    {
        // Add the next byte to the byte array
        mpz_class auxScalar = s & ( (i<(dataSize-1)) ? ScalarMask16 : ScalarMask20 );
        pData[i] = auxScalar.get_ui();

        // Shift right 2 bytes the scalar content
        s = s >> 16;

        // When we run out of significant bytes, break
        if (s == ScalarZero) break;
    }
    if (s > ScalarMask4)
    {
        zklog.error("scalar2ba16() run out of buffer of " + to_string(dataSize) + " bytes");
        exitProcess();
    }
    dataSize = i+1;
}

string scalar2ba32(const mpz_class &_s)
{
    mpz_class s(_s);
    string result;
    result.append(32, 0);
    for (uint64_t i=0; i<32; i++)
    {
        result[31-i] = s.get_ui();

        // Shift right 1 byte the scalar content
        s = s >> 8;

        // When we run out of significant bytes, break
        if (s == ScalarZero)
        {
            return result;
        }
    }
    if (s != 0)
    {
        zklog.error("scalar2ba32() run out of buffer of 32 bytes");
        exitProcess();
    }
    return result;
}

void scalar2bytes(mpz_class s, uint8_t (&bytes)[32])
{
    for (uint64_t i=0; i<32; i++)
    {
        mpz_class aux = s & ScalarMask8;
        bytes[i] = aux.get_ui();
        s = s >> 8;
    }
    if (s != ScalarZero)
    {
        zklog.error("scalar2bytes() run out of space of 32 bytes");
        exitProcess();
    }
}

void scalar2bytesBE(mpz_class s, uint8_t *pBytes)
{
    for (uint64_t i=0; i<32; i++)
    {
        mpz_class aux = s & ScalarMask8;
        pBytes[31 - i] = aux.get_ui();
        s = s >> 8;
    }
    if (s != ScalarZero)
    {
        zklog.error("scalar2bytesBE() run out of space of 32 bytes");
        exitProcess();
    }
}

/* Scalar to byte array string conversion */

string scalar2ba(const mpz_class &s)
{
    uint64_t size = mpz_sizeinbase(s.get_mpz_t(), 256);
    if (size > 32)
    {
        zklog.error("scalar2ba() failed, size=" + to_string(size) + " is > 32");
        exitProcess();
    }

    uint8_t buffer[32] = {0};
    mpz_export(buffer, NULL, 1, 1, 1, 0, s.get_mpz_t());

    string result;
    for (uint64_t i = 0; i < size; i++)
    {
        result.push_back(buffer[i]);
    }
    
    return result;
}

/* Converts a scalar to a vector of bits of the scalar, with value 1 or 0; bits[0] is least significant bit */

void scalar2bits(mpz_class s, vector<uint8_t> &bits)
{
    while (s > ScalarZero)
    {
        if ((s & 1) == ScalarOne)
        {
            bits.push_back(1);
        }
        else
        {
            bits.push_back(0);
        }
        s = s >> 1;
    }
}

/* Converts an unsigned 32 to a vector of bits, with value 1 or 0; bits[0] is least significant bit */

void u322bits(uint32_t value, vector<uint8_t> &bits)
{
    // Call scalar2bits()
    mpz_class s(value);
    scalar2bits(s, bits);

    // Append any missing zeros
    while (bits.size() < 32)
    {
        bits.push_back(0);
    }
}

uint32_t bits2u32(const vector<uint8_t> &bits)
{
    if (bits.size() != 32)
    {
        zklog.error("bits2u32() got invalid bits size=" + to_string(bits.size()));
        exitProcess();
    }
    uint32_t result = 0;
    for (int64_t i=31; i>=0; i--)
    {
        result = result << 1;
        switch (bits[i])
        {
            case 0:
                break;
            case 1:
                result += 1;
                break;
            default:
                zklog.error("bits2u32() got invalid bit i=" + to_string(i) + " value=" + to_string(bits[i]));
                exitProcess();
                break;
        }
    }
    return result;
}

/* Converts an unsigned 64 to a vector of bits, with value 1 or 0; bits[0] is least significant bit */

void u642bits(uint64_t value, vector<uint8_t> &bits)
{
    // Call scalar2bits()
    mpz_class s(value);
    scalar2bits(s, bits);

    // Append any missing zeros
    while (bits.size() < 64)
    {
        bits.push_back(0);
    }
}

uint64_t bits2u64(const vector<uint8_t> &bits)
{
    if (bits.size() != 64)
    {
        zklog.error("bits2u64() got invalid bits size=" + to_string(bits.size()));
        exitProcess();
    }
    uint64_t result = 0;
    for (int64_t i=63; i>=0; i--)
    {
        result = result << 1;
        switch (bits[i])
        {
            case 0:
                break;
            case 1:
                result += 1;
                break;
            default:
                zklog.error("bits2u64() got invalid bit i=" + to_string(i) + " value=" + to_string(bits[i]));
                exitProcess();
                break;
        }
    }
    return result;
}

/* Byte to/from bits array conversion, with value 1 or 0; bits[0] is the least significant bit */

void byte2bits(uint8_t byte, uint8_t *pBits)
{
    for (uint64_t i=0; i<8; i++)
    {
        if ((byte&1) == 1)
        {
            pBits[i] = 1;
        }
        else
        {
            pBits[i] = 0;
        }
        byte = byte >> 1;
    }
}

void bits2byte(const uint8_t *pBits, uint8_t &byte)
{
    byte = 0;
    for (uint64_t i=0; i<8; i++)
    {
        byte = byte << 1;
        if ((pBits[7-i]&0x01) == 1)
        {
            byte |= 1;
        }
    }
}

void u642bytes (uint64_t input, uint8_t * pOutput, bool bBigEndian)
{
    for (uint64_t i=0; i<8; i++)
    {
        pOutput[bBigEndian ? (7-i) : i] = input & 0x00000000000000FF;
        if (i != 7) input = input >> 8;
    }
}

void bytes2u32 (const uint8_t * pInput, uint32_t &output, bool bBigEndian)
{
    output = 0;
    for (uint64_t i=0; i<4; i++)
    {
        if (i != 0) output = output << 8;
        output |= pInput[bBigEndian ? i : (3-i)];
    }
}

void bytes2u64 (const uint8_t * pInput, uint64_t &output, bool bBigEndian)
{
    output = 0;
    for (uint64_t i=0; i<8; i++)
    {
        if (i != 0) output = output << 8;
        output |= pInput[bBigEndian ? i : (7-i)];
    }
}

uint64_t swapBytes64 (uint64_t input)
{
    return ((((input) & 0xff00000000000000ull) >> 56) |
            (((input) & 0x00ff000000000000ull) >> 40) |
            (((input) & 0x0000ff0000000000ull) >> 24) |
            (((input) & 0x000000ff00000000ull) >> 8 ) |
            (((input) & 0x00000000ff000000ull) << 8 ) |
            (((input) & 0x0000000000ff0000ull) << 24) |
            (((input) & 0x000000000000ff00ull) << 40) |
            (((input) & 0x00000000000000ffull) << 56));
}

