#include "fr_goldilocks.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <assert.h>
#include <string>


static mpz_t q;
static mpz_t zero;
static mpz_t one;
static mpz_t mask;
static size_t nBits;
static bool initialized = false;


void FrG_toMpz(mpz_t r, PFrGElement pE) {
    FrGElement tmp;
    FrG_toNormal(&tmp, pE);
    if (!(tmp.type & FrG_LONG)) {
        mpz_set_si(r, tmp.shortVal);
        if (tmp.shortVal<0) {
            mpz_add(r, r, q);
        }
    } else {
        mpz_import(r, FrG_N64, -1, 8, -1, 0, (const void *)tmp.longVal);
    }
}

void FrG_fromMpz(PFrGElement pE, mpz_t v) {
    if (mpz_fits_sint_p(v)) {
        pE->type = FrG_SHORT;
        pE->shortVal = mpz_get_si(v);
    } else {
        pE->type = FrG_LONG;
        for (int i=0; i<FrG_N64; i++) pE->longVal[i] = 0;
        mpz_export((void *)(pE->longVal), NULL, -1, 8, -1, 0, v);
    }
}


bool FrG_init() {
    if (initialized) return false;
    initialized = true;
    mpz_init(q);
    mpz_import(q, FrG_N64, -1, 8, -1, 0, (const void *)FrG_q.longVal);
    mpz_init_set_ui(zero, 0);
    mpz_init_set_ui(one, 1);
    nBits = mpz_sizeinbase (q, 2);
    mpz_init(mask);
    mpz_mul_2exp(mask, one, nBits);
    mpz_sub(mask, mask, one);
    return true;
}

void FrG_str2element(PFrGElement pE, char const *s) {
    mpz_t mr;
    mpz_init_set_str(mr, s, 10);
    mpz_fdiv_r(mr, mr, q);
    FrG_fromMpz(pE, mr);
    mpz_clear(mr);
}

void FrG_str2element(PFrGElement pE, char const *s, uint base) {
    mpz_t mr;
    mpz_init_set_str(mr, s, base);
    mpz_fdiv_r(mr, mr, q);
    FrG_fromMpz(pE, mr);
    mpz_clear(mr);
}

char *FrG_element2str(PFrGElement pE) {
    FrGElement tmp;
    mpz_t r;
    if (!(pE->type & FrG_LONG)) {
        if (pE->shortVal>=0) {
            char *r = new char[32];
            sprintf(r, "%d", pE->shortVal);
            return r;
        } else {
            mpz_init_set_si(r, pE->shortVal);
            mpz_add(r, r, q);
        }
    } else {
        FrG_toNormal(&tmp, pE);
        mpz_init(r);
        mpz_import(r, FrG_N64, -1, 8, -1, 0, (const void *)tmp.longVal);
    }
    char *res = mpz_get_str (0, 10, r);
    mpz_clear(r);
    return res;
}

void FrG_idiv(PFrGElement r, PFrGElement a, PFrGElement b) {
    mpz_t ma;
    mpz_t mb;
    mpz_t mr;
    mpz_init(ma);
    mpz_init(mb);
    mpz_init(mr);

    FrG_toMpz(ma, a);
    // char *s1 = mpz_get_str (0, 10, ma);
    // printf("s1 %s\n", s1);
    FrG_toMpz(mb, b);
    // char *s2 = mpz_get_str (0, 10, mb);
    // printf("s2 %s\n", s2);
    mpz_fdiv_q(mr, ma, mb);
    // char *sr = mpz_get_str (0, 10, mr);
    // printf("r %s\n", sr);
    FrG_fromMpz(r, mr);

    mpz_clear(ma);
    mpz_clear(mb);
    mpz_clear(mr);
}

void FrG_mod(PFrGElement r, PFrGElement a, PFrGElement b) {
    mpz_t ma;
    mpz_t mb;
    mpz_t mr;
    mpz_init(ma);
    mpz_init(mb);
    mpz_init(mr);

    FrG_toMpz(ma, a);
    FrG_toMpz(mb, b);
    mpz_fdiv_r(mr, ma, mb);
    FrG_fromMpz(r, mr);

    mpz_clear(ma);
    mpz_clear(mb);
    mpz_clear(mr);
}

void FrG_pow(PFrGElement r, PFrGElement a, PFrGElement b) {
    mpz_t ma;
    mpz_t mb;
    mpz_t mr;
    mpz_init(ma);
    mpz_init(mb);
    mpz_init(mr);

    FrG_toMpz(ma, a);
    FrG_toMpz(mb, b);
    mpz_powm(mr, ma, mb, q);
    FrG_fromMpz(r, mr);

    mpz_clear(ma);
    mpz_clear(mb);
    mpz_clear(mr);
}

void FrG_inv(PFrGElement r, PFrGElement a) {
    mpz_t ma;
    mpz_t mr;
    mpz_init(ma);
    mpz_init(mr);

    FrG_toMpz(ma, a);
    mpz_invert(mr, ma, q);
    FrG_fromMpz(r, mr);
    mpz_clear(ma);
    mpz_clear(mr);
}

void FrG_div(PFrGElement r, PFrGElement a, PFrGElement b) {
    FrGElement tmp;
    FrG_inv(&tmp, b);
    FrG_mul(r, a, &tmp);
}

void FrG_fail() {
    assert(false);
}


RawFrG::RawFrG() {
    FrG_init();
    set(fZero, 0);
    set(fOne, 1);
    neg(fNegOne, fOne);
}

RawFrG::~RawFrG() {
}

void RawFrG::fromString(Element &r, const std::string &s, uint32_t radix) {
    mpz_t mr;
    mpz_init_set_str(mr, s.c_str(), radix);
    mpz_fdiv_r(mr, mr, q);
    for (int i=0; i<FrG_N64; i++) r.v[i] = 0;
    mpz_export((void *)(r.v), NULL, -1, 8, -1, 0, mr);
    FrG_rawToMontgomery(r.v,r.v);
    mpz_clear(mr);
}

void RawFrG::fromUI(Element &r, unsigned long int v) {
    mpz_t mr;
    mpz_init(mr);
    mpz_set_ui(mr, v);
    for (int i=0; i<FrG_N64; i++) r.v[i] = 0;
    mpz_export((void *)(r.v), NULL, -1, 8, -1, 0, mr);
    FrG_rawToMontgomery(r.v,r.v);
    mpz_clear(mr);
}

RawFrG::Element RawFrG::set(int value) {
  Element r;
  set(r, value);
  return r;
}

void RawFrG::set(Element &r, int value) {
  mpz_t mr;
  mpz_init(mr);
  mpz_set_si(mr, value);
  if (value < 0) {
      mpz_add(mr, mr, q);
  }

  mpz_export((void *)(r.v), NULL, -1, 8, -1, 0, mr);
      
  for (int i=0; i<FrG_N64; i++) r.v[i] = 0;
  mpz_export((void *)(r.v), NULL, -1, 8, -1, 0, mr);
  FrG_rawToMontgomery(r.v,r.v);
  mpz_clear(mr);
}

std::string RawFrG::toString(const Element &a, uint32_t radix) {
    Element tmp;
    mpz_t r;
    FrG_rawFromMontgomery(tmp.v, a.v);
    mpz_init(r);
    mpz_import(r, FrG_N64, -1, 8, -1, 0, (const void *)(tmp.v));
    char *res = mpz_get_str (0, radix, r);
    mpz_clear(r);
    std::string resS(res);
    free(res);
    return resS;
}

void RawFrG::inv(Element &r, const Element &a) {
    mpz_t mr;
    mpz_init(mr);
    mpz_import(mr, FrG_N64, -1, 8, -1, 0, (const void *)(a.v));
    mpz_invert(mr, mr, q);


    for (int i=0; i<FrG_N64; i++) r.v[i] = 0;
    mpz_export((void *)(r.v), NULL, -1, 8, -1, 0, mr);

    FrG_rawMMul(r.v, r.v,FrG_rawR3);
    mpz_clear(mr);
}

void RawFrG::div(Element &r, const Element &a, const Element &b) {
    Element tmp;
    inv(tmp, b);
    mul(r, a, tmp);
}

#define BIT_IS_SET(s, p) (s[p>>3] & (1 << (p & 0x7)))
void RawFrG::exp(Element &r, const Element &base, uint8_t* scalar, unsigned int scalarSize) {
    bool oneFound = false;
    Element copyBase;
    copy(copyBase, base);
    for (int i=scalarSize*8-1; i>=0; i--) {
        if (!oneFound) {
            if ( !BIT_IS_SET(scalar, i) ) continue;
            copy(r, copyBase);
            oneFound = true;
            continue;
        }
        square(r, r);
        if ( BIT_IS_SET(scalar, i) ) {
            mul(r, r, copyBase);
        }
    }
    if (!oneFound) {
        copy(r, fOne);
    }
}

void RawFrG::toMpz(mpz_t r, const Element &a) {
    Element tmp;
    FrG_rawFromMontgomery(tmp.v, a.v);
    mpz_import(r, FrG_N64, -1, 8, -1, 0, (const void *)tmp.v);
}

void RawFrG::fromMpz(Element &r, const mpz_t a) {
    for (int i=0; i<FrG_N64; i++) r.v[i] = 0;
    mpz_export((void *)(r.v), NULL, -1, 8, -1, 0, a);
    FrG_rawToMontgomery(r.v, r.v);
}

int RawFrG::toRprBE(const Element &element, uint8_t *data, int bytes)
{
    if (bytes < FrG_N64 * 8) {
      return -(FrG_N64 * 8);
    }

    mpz_t r;
    mpz_init(r);
  
    toMpz(r, element);
    
    mpz_export(data, NULL, 1, 8, 1, 0, r);
  
    return FrG_N64 * 8;
}

int RawFrG::fromRprBE(Element &element, const uint8_t *data, int bytes)
{
    if (bytes < FrG_N64 * 8) {
      return -(FrG_N64* 8);
    }
    mpz_t r;
    mpz_init(r);

    mpz_import(r, FrG_N64 * 8, 0, 1, 0, 0, data);
    fromMpz(element, r);
    return FrG_N64 * 8;
}

static bool init = FrG_init();

RawFrG RawFrG::field;

#ifdef __aarch64__
FrGElement FrG_q = {
    .shortVal = FrG_SHORT,
    .type = FrG_LONG
};
FrGElement FrG_R3 = {
    .shortVal = FrG_SHORT,
    .type = FrG_LONG
};
FrGRawElement FrG_rawq = {0xffffffff00000001};
FrGRawElement FrG_rawR3 = {0x0000000000000001};

void FrG_copy(PFrGElement r, PFrGElement a){}
void FrG_copyn(PFrGElement r, PFrGElement a, int n){}
void FrG_add(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_sub(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_neg(PFrGElement r, PFrGElement a){}
void FrG_mul(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_square(PFrGElement r, PFrGElement a){}
void FrG_band(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_bor(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_bxor(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_bnot(PFrGElement r, PFrGElement a){}
void FrG_shl(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_shr(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_eq(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_neq(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_lt(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_gt(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_leq(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_geq(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_land(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_lor(PFrGElement r, PFrGElement a, PFrGElement b){}
void FrG_lnot(PFrGElement r, PFrGElement a){}
void FrG_toNormal(PFrGElement r, PFrGElement a){}
void FrG_toLongNormal(PFrGElement r, PFrGElement a){}
void FrG_toMontgomery(PFrGElement r, PFrGElement a){}

int FrG_isTrue(PFrGElement pE){return 0;}
int FrG_toInt(PFrGElement pE){return 0;}

void FrG_rawCopy(FrGRawElement pRawResult, const FrGRawElement pRawA){}
void FrG_rawSwap(FrGRawElement pRawResult, FrGRawElement pRawA){}
void FrG_rawAdd(FrGRawElement pRawResult, const FrGRawElement pRawA, const FrGRawElement pRawB){}
void FrG_rawSub(FrGRawElement pRawResult, const FrGRawElement pRawA, const FrGRawElement pRawB){}
void FrG_rawNeg(FrGRawElement pRawResult, const FrGRawElement pRawA){}
void FrG_rawMMul(FrGRawElement pRawResult, const FrGRawElement pRawA, const FrGRawElement pRawB){}
void FrG_rawMSquare(FrGRawElement pRawResult, const FrGRawElement pRawA){}
void FrG_rawMMul1(FrGRawElement pRawResult, const FrGRawElement pRawA, uint64_t pRawB){}
void FrG_rawToMontgomery(FrGRawElement pRawResult, const FrGRawElement &pRawA){}
void FrG_rawFromMontgomery(FrGRawElement pRawResult, const FrGRawElement &pRawA){}
int FrG_rawIsEq(const FrGRawElement pRawA, const FrGRawElement pRawB){return 0;}
int FrG_rawIsZero(const FrGRawElement pRawB){return 0;}
#endif
