#include <iostream>
#include <assert.h>
#include <cstdlib>

using namespace std;

#define N_FEAT			617	//feature per input (e.g., isolet: 617)
#define N_CLASS			26	//number of classes. (e.g., isolet: 26, ucihar 12)
#define Dhv				2048 //hypervectors length
#define COL				8 //number of columns of a matrix-vector multiplication window (keep fixed 8)
// NOTE: For current Hetero-C++ implementation, ROW ***must*** be kept as 32. The original FPGA code uses ROW as the length of many ap_ints, which is not "standard" C++.
#define ROW				32 //number of rows of a matrix-vector multiplication window (32, 64, 128, 256, 512)


#define PAD_			(N_FEAT & (COL - 1))
#if PAD_ == 0
	#define PAD 		0
#else
	#define PAD 		(COL - PAD_)
#endif

#define N_FEAT_PAD		(N_FEAT + PAD)	//feature per input (e.g., isolet: 624, ucihar 568)

struct HyperVector512 {
	uint32_t buf[512 / 32];
};

void hd(int *input_gmem, std::size_t input_gmem_size, int *ID_gmem, std::size_t ID_gmem_size, int *classHV_gmem, std::size_t classHV_gmem_size, int *labels_gmem, std::size_t labels_gmem_size, HyperVector512 *encHV_gmem, std::size_t encHV_gmem_size, int train, int size);
