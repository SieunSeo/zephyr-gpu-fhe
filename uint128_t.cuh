#pragma once

#include <assert.h>
#include <stdio.h>
#include <inttypes.h>
#include <cstdio>

//----------------------------------------------------
// 
//----------------------------------------------------
struct uint128_t {
	//----------------------------------------------------
	// member variables
	//----------------------------------------------------
	uint64_t hi, lo;
	uint128_t() { hi = lo = 0; }
	void print()const { printf("%" PRIu64 "*beta+%" PRIu64 " ", hi, lo); }
	//----------------------------------------------------
	// conversion with uint64_t
	//----------------------------------------------------
	uint128_t(uint64_t lo_) { hi = 0;lo = lo_; }
	operator uint64_t   ()const { return lo; }
	operator float()const {
		const float beta = 1.8446744073709552e19;
		return float(hi) * beta + float(lo);
	}
	//----------------------------------------------------
	// arithmetric operations
	//----------------------------------------------------
	uint128_t operator+(uint128_t A) const { uint128_t out;out.lo = lo + A.lo;int c = out.lo < lo;out.hi = hi + A.hi + c;return out; }
	uint128_t operator-(uint128_t A) const { uint128_t out;out.lo = lo - A.lo;int b = out.lo > lo;out.hi = hi - A.hi - b;return out; }
	uint128_t operator*(uint64_t) const;
	uint64_t operator/(uint64_t) const;
	uint64_t operator%(uint64_t) const;
	//----------------------------------------------------
	// comparison operations
	//----------------------------------------------------
	bool operator>=(uint128_t A) const { if (hi > A.hi) return true;else if (hi < A.hi) return false;else return lo >= A.lo; }
	bool operator> (uint128_t A) const { if (hi > A.hi) return true;else if (hi < A.hi) return false;else return lo > A.lo; }
	bool operator<=(uint128_t A) const { if (hi > A.hi) return false;else if (hi < A.hi) return true;else return lo <= A.lo; }
	bool operator< (uint128_t A) const { if (hi > A.hi) return false;else if (hi < A.hi) return true;else return lo < A.lo; }
	bool operator==(uint128_t A) const { return (hi == A.hi) && (lo == A.lo); }
	//----------------------------------------------------
	// shift operations
	//----------------------------------------------------
	uint128_t operator<<(int i) const { uint128_t out;out.hi = (hi << i) + (lo >> (64 - i));  out.lo = lo << i;return out; }
	uint128_t operator>>(int i) const { uint128_t out;out.hi = (hi >> i); out.lo = (hi << (64 - i)) + (lo >> i);return out; }
};



//----------------------------------------------------
// 
//----------------------------------------------------
struct uint256_t {
	//----------------------------------------------------
	// member variables
	//----------------------------------------------------
	uint128_t hi, lo;
	uint256_t() {}
	void print()const { printf("%" PRIu64 "*beta^3 +%" PRIu64 "*beta^2+%" PRIu64 "*beta +%" PRIu64 "\n", hi.hi, hi.lo, lo.hi, lo.lo); }
	//----------------------------------------------------
	// conversion with uint64_t
	//----------------------------------------------------
	uint256_t(uint128_t lo_) { hi.hi = 0; hi.lo = 0;lo = lo_; }
	operator uint128_t   ()const { return lo; }
	operator uint64_t   ()const { return lo.lo; }
	operator float()const {
		const float beta = 1.8446744073709552e19;
		return float(hi) * beta * beta + float(lo);
	}
	//----------------------------------------------------
	// arithmetric operations
	//----------------------------------------------------
	uint256_t operator+(uint256_t A) const { uint256_t out;out.lo = lo + A.lo;int c = out.lo < lo;out.hi = hi + A.hi + uint128_t(c);return out; }
	uint256_t operator-(uint256_t A) const { uint256_t out;out.lo = lo - A.lo;int b = out.lo > lo;out.hi = hi - A.hi - uint128_t(b);return out; }
	uint256_t operator*(uint32_t b) const {
		assert((hi.hi >> 32) == 0); uint256_t out;
		out.lo = uint128_t(lo.lo) * uint64_t(b);
		uint128_t lhb = uint128_t(lo.hi) * uint64_t(b);
		out.hi = uint128_t(hi.lo) * uint64_t(b);
		out.lo.hi = out.lo.hi + lhb.lo; int c = out.lo.hi < lhb.lo;
		out.hi = out.hi + uint128_t(lhb.hi + c);
		return out;
	}
	//----------------------------------------------------
	// comparison operations
	//----------------------------------------------------
	bool operator>=(uint256_t A) const { if (hi > A.hi) return true;else if (hi < A.hi) return false;else return lo >= A.lo; }
	bool operator> (uint256_t A) const { if (hi > A.hi) return true;else if (hi < A.hi) return false;else return lo > A.lo; }
	bool operator<=(uint256_t A) const { if (hi > A.hi) return false;else if (hi < A.hi) return true;else return lo <= A.lo; }
	bool operator< (uint256_t A) const { if (hi > A.hi) return false;else if (hi < A.hi) return true;else return lo < A.lo; }
	bool operator==(uint256_t A) const { return (hi == A.hi) && (lo == A.lo); }
	//----------------------------------------------------
	// shift operations
	//----------------------------------------------------
	uint256_t operator<<(int i) const { uint256_t out;out.hi = (hi << i) + (lo >> (128 - i));  out.lo = lo << i;return out; }
	uint256_t operator>>(int i) const { uint256_t out;out.hi = (hi >> i); out.lo = (hi << (128 - i)) + (lo >> i);return out; }
};



