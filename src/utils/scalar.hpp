#ifndef SCALAR_HPP
#define SCALAR_HPP

#include <gmpxx.h>
#include <string>
#include <vector>
#include "exit_process.hpp"
#include "zklog.hpp"
#include "constants.hpp"

using namespace std;

/* Global scalar variables */
extern mpz_class ScalarMask4;
extern mpz_class ScalarMask8;
extern mpz_class ScalarMask16;
extern mpz_class ScalarMask20;
extern mpz_class ScalarMask32;
extern mpz_class ScalarMask64;
extern mpz_class ScalarMask160;
extern mpz_class ScalarMask256;
extern mpz_class ScalarTwoTo8;
extern mpz_class ScalarTwoTo16;
extern mpz_class ScalarTwoTo18;
extern mpz_class ScalarTwoTo32;
extern mpz_class ScalarTwoTo64;
extern mpz_class ScalarTwoTo128;
extern mpz_class ScalarTwoTo192;
extern mpz_class ScalarTwoTo254;
extern mpz_class ScalarTwoTo255;
extern mpz_class ScalarTwoTo256;
extern mpz_class ScalarTwoTo257;
extern mpz_class ScalarTwoTo258;
extern mpz_class ScalarTwoTo259;
extern mpz_class ScalarZero;
extern mpz_class ScalarOne;
extern mpz_class ScalarGoldilocksPrime;
extern mpz_class Scalar4xGoldilocksPrime;

/* Normalized strings */
string Remove0xIfPresent      (const string &s);
void   Remove0xIfPresentNoCopy(      string &s);
string Add0xIfMissing         (const string &s);
string PrependZeros           (const string &s, uint64_t n);
void   PrependZerosNoCopy     (      string &s, uint64_t n);
string NormalizeTo0xNFormat   (const string &s, uint64_t n);
string NormalizeToNFormat     (const string &s, uint64_t n);
string stringToLower          (const string &s);

// Check that a char is an hex character
inline bool charIsHex (char c)
{
    if ( (c >= '0') && (c <= '9') ) return true;
    if ( (c >= 'a') && (c <= 'f') ) return true;
    if ( (c >= 'A') && (c <= 'F') ) return true;
    return false;
}

// Check that a char is a decimal character
inline bool charIsDec (char c)
{
    if ( (c >= '0') && (c <= '9') ) return true;
    return false;
}

// Check that the string contains only hex characters
bool stringIsHex (const string &s);

// Check that the string contains only decimal characters
bool stringIsDec (const string &s);

// Check that the string contains only 0x + hex characters
bool stringIs0xHex (const string &s);

/* Keccak */
void   keccak256 (const uint8_t *pInputData, uint64_t inputDataSize, uint8_t *pOutputData, uint64_t outputDataSize);
void   keccak256 (const uint8_t *pInputData, uint64_t inputDataSize, uint8_t (&hash)[32]);
void   keccak256 (const uint8_t *pInputData, uint64_t inputDataSize, mpz_class &hash);
string keccak256 (const uint8_t *pInputData, uint64_t inputDataSize);
void   keccak256 (const vector<uint8_t> &input, mpz_class &hash);

/* Byte to/from char conversion */
uint8_t char2byte (char c);
char    byte2char (uint8_t b);
string  byte2string(uint8_t b);

/* Strint to/from byte array conversion
   s must be even sized, and must not include the leading "0x"
   pData buffer must be big enough to store converted data */
uint64_t string2ba (const string &s, uint8_t *pData, uint64_t &dataSize);
void     string2ba (const string &textString, string &baString);
string   string2ba (const string &textString);
void     string2ba (const string os, vector<uint8_t> &data);

void     ba2string (string &s, const uint8_t *pData, uint64_t dataSize);
string   ba2string (const uint8_t *pData, uint64_t dataSize);
void     ba2string (const string &baString, string &textString);
string   ba2string (const string &baString);
void     ba2ba     (const string &baString, vector<uint8_t> (&baVector));
void     ba2ba     (const vector<uint8_t> (&baVector), string &baString);
void     ba2ba     (string &baString, const uint64_t ba);
uint64_t ba2ba     (const string &baString);

/* Byte array of exactly 2 bytes conversion */
void ba2u16(const uint8_t *pData, uint16_t &n);
void ba2u32(const uint8_t *pData, uint32_t &n);
void ba2scalar(const uint8_t *pData, uint64_t dataSize, mpz_class &s);

/* Scalar to byte array conversion (up to dataSize bytes) */
void scalar2ba(uint8_t *pData, uint64_t &dataSize, mpz_class s);
void scalar2ba16(uint64_t *pData, uint64_t &dataSize, mpz_class s);
string scalar2ba32(const mpz_class &s); // Returns exactly 32 bytes
void scalar2bytes(mpz_class s, uint8_t (&bytes)[32]);
void scalar2bytesBE(mpz_class s, uint8_t *pBytes); // pBytes must be a 32-bytes array


/* Scalar to byte array string conversion */
string scalar2ba(const mpz_class &s);

inline void ba2scalar(mpz_class &s, const string &ba)
{
    mpz_import(s.get_mpz_t(), ba.size(), 1, 1, 0, 0, ba.c_str());
}

inline void ba2scalar(mpz_class &s, const uint8_t (&hash)[32])
{
    mpz_import(s.get_mpz_t(), 32, 1, 1, 0, 0, hash);
}

/* Converts a scalar to a vector of bits of the scalar, with value 1 or 0; bits[0] is least significant bit */
void scalar2bits(mpz_class s, vector<uint8_t> &bits);

/* Converts an unsigned 32 to a vector of bits, with value 1 or 0; bits[0] is least significant bit */
void u322bits(uint32_t value, vector<uint8_t> &bits);
uint32_t bits2u32(const vector<uint8_t> &bits);

/* Converts an unsigned 64 to a vector of bits, with value 1 or 0; bits[0] is least significant bit */
void u642bits(uint64_t value, vector<uint8_t> &bits);
uint64_t bits2u64(const vector<uint8_t> &bits);

/* Byte to/from bits array conversion, with value 1 or 0; bits[0] is the least significant bit */
void byte2bits(uint8_t byte, uint8_t *pBits);
void bits2byte(const uint8_t *pBits, uint8_t &byte);

/* Scalar to/from fec conversion */
//void fec2scalar(RawFec &fec, const RawFec::Element &fe, mpz_class &s);
//void scalar2fec(RawFec &fec, RawFec::Element &fe, const mpz_class &s);

/* Less than 4
*  Computes comparation of 256 bits, these values (a,b) are divided in 4 chunks of 64 bits
*  and compared one-to-one, 4 comparations, lt4 return 1 if ALL chunks of a are less than b.
*  lt = a[0] < b[0] && a[1] < b[1] && a[2] < b[2] && a[3] < b[3]
*/
inline mpz_class lt4(const mpz_class& a, const mpz_class& b) {
     
    mpz_class a_=a;
    mpz_class b_=b;
    mpz_class mask(0xffffffffffffffff);
    for (int i = 0; i < 4; i++) {    
        if ((a_ & mask) >= (b_ & mask) ) {
            return 0;
        }
        a_ = a_ >> 64;
        b_ = b_ >> 64;
    }
    return 1;
}

/* Unsigned 64 to an array of bytes.  pOutput must be 8 bytes long */
void u642bytes (uint64_t input, uint8_t * pOutput, bool bBigEndian);

/* Array of bytes to unsigned 32.  pInput must be 4 bytes long */
void bytes2u32 (const uint8_t * pInput, uint32_t &output, bool bBigEndian);

/* Array of bytes to unsigned 64.  pInput must be 8 bytes long */
void bytes2u64 (const uint8_t * pInput, uint64_t &output, bool bBigEndian);

/* unsigned64 to string*/
inline void U64toString(std::string &result, const uint64_t in1, const int radix)
{
    mpz_class aux = in1;
    result = aux.get_str(radix);
}

/* unsigned64 to string*/
inline std::string U64toString( const uint64_t in1, const int radix)
{
    mpz_class aux = in1;
    string result = aux.get_str(radix);
    return result;
}
/* Swap bytes, e.g. little to big endian, and vice-versa */
uint64_t swapBytes64 (uint64_t input);

/* Rotate */
uint32_t inline rotateRight32( uint32_t input, uint64_t bits) { return (input >> bits) | (input << (32-bits)); }
uint32_t inline rotateLeft32( uint32_t input, uint64_t bits) { return (input << bits) | (input >> (32-bits)); }
uint64_t inline rotateRight64( uint64_t input, uint64_t bits) { return (input >> bits) | (input << (64-bits)); }
uint64_t inline rotateLeft64( uint64_t input, uint64_t bits) { return (input << bits) | (input >> (64-bits)); }

#endif