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

template <typename T, std::size_t size = 64>
struct FIFO {
        std::size_t head = 0, tail = 0;
        std::size_t len = 0;
	std::size_t total_pushes = 0, max_len = 0;
        T buf[size] = {0};

        void push(T t) {
                assert(len < size);
                ++len;
		max_len = max_len < len ? len : max_len;
		++total_pushes;
                buf[head] = t;
                head = (head + 1) % size;
        }

        T pop() {
                assert(len > 0);
                --len;
                T t = buf[tail];
                tail = (tail + 1) % size;
                return t;
        }

	void clear() {
		len = 0;
		head = 0;
		tail = 0;
	}

	friend FIFO<T, size> &operator<<(FIFO<T, size> &fifo, const T &t) {
		fifo.push(t);
		return fifo;
	}

	friend FIFO<T, size> &operator>>(FIFO<T, size> &fifo, T &t) {
		t = fifo.pop();
		return fifo;
	}
};

struct HyperVector512 {
	uint32_t buf[512 / 32];
};

void hd(int *input_gmem, int *ID_gmem, int *classHV_gmem, int *labels_gmem, HyperVector512 *encHV_gmem, int *trainScore, int train, int size);
