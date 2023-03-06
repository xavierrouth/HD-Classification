#ifdef HPVM
#include <heterocc.h>
#endif
#include "hd.h"

/*
 * inputStream fetches input features as ints, and streames to the next functions.
 *
 * input_gmem (input): input data port; each feature is quantized to an integer.
 * feature_stream (output): N_FEAT_PAD parallel streams to stream the data to the next module.
 * size (input): number of data sampels.
 */
void inputStream(int *__restrict input_gmem, int feature_stream[N_FEAT_PAD], int size, int iter_read) {

	 //Need to move the pointer by intPerInput after each input
	int offset = iter_read * N_FEAT;
	loop_features:
	for (int i = 0; i < N_FEAT; i++) {
		feature_stream[i] = input_gmem[offset + i];
	}
	for (int i = 0; i < PAD; i++) {
		feature_stream[N_FEAT + i] = 0;
	}
}

/*
 * encodeUnit reads input features from the stream and obtains encoding hypervector using random projection (RP) algorithm.
 * RP is all about a matrix-vector multplicatoin. The matrix is ID hypervectors arrangaed as Dhv*Div (Dhv = HV dimensions, Div = #input vector elements).
 * We break this A=(Dhv*Div) * B=(Div*1) multiplicaton into slinding windows of ROW*COL on the matrix A.
 * It takes Div/COL cycles for the sliding window to reach the right-most column, when we accomplish ROW dimensions. Total cycles = (Dhv*Div)/(ROW*COL)*(latency of accumulating COL partials).
 * Note that we only have a single seed ID (Dhv bits) in a (Dhv/ROW) x ROW array for column 0, and generate the ID of column k by k-bit rotating (circular shift).
 *
 * feature_stream (input): N_FEAT_PAD parallel streams from the previous module to read input features of a data sample.
 * ID (input): seed ID hypervector, packed as Dhv/ROW (rows) of ROW bit (total Dhv bits).
 * enc_stream (output): streams ROW encoded dimensions per (Div/COL) cycles to the next module.
 * size (input): number of data samples.
 *
 */
void encodeUnit(int *__restrict feature_stream, uint32_t *__restrict ID, int *__restrict enc_stream, int size, int iter_read) {

	//Operate on ROW encoding dimension per cycle
	int encHV_partial[ROW];

	//Factor the feature memory into COL, as we read COL elements of it in parallel.

	//ID register to keep ROW+COL bits for a ROW*COL window.
	//ID memory has ROW bits per cell, so we use 2*ROW bit register (extra bits will be used in the next window).
	//It might look a little tricky. See the report for visualization.
	uint64_t ID_reg;

	//Probe ROW rows simultanously for mat-vec multplication (result = r encoding dimension).
	//Each row block has Dhv/ROW rows.
	loop_mat_row:
	for (int r = 0; r < Dhv/ROW; r++) {
		//Clear the partial encoding buffer when the window starts the new rows.
		loop_clear:
		for (int i = 0; i < ROW; i++) {
			encHV_partial[i] = 0;
		}
		//We need to figure out which ID bits should be read.
		//At the beginning of row block r, we read bits of the block r and r+1 (each block has Dhv/ROW bits).
		int cycle = 0;
		int addr1 = r;
		int addr2 = r+1;
		//In the last block, r+1 becomes Dhv/ROW, so we start from 0 (ID bits are stored circular).
		if (addr2 == Dhv/ROW)
			addr2 = 0;
		ID_reg = (((uint64_t) ID[addr2]) << 32) | ((uint64_t) ID[addr1]);

		//Divide each of row blocks into columns (tiles) of COL, i.e., multiply a ROW*COL tile to COL features at a given cycle.
		loop_mat_col:
		for (int c = 0; c < N_FEAT_PAD/COL; c++) {
			//Iterate over the rows and columns of the ROW*COL tile to perform matrix-vector multplication.
			loop_tile_row:
			for (int i = 0; i < ROW; i++) {
				//In each ID register of 2*ROW bits, bits [0-COL) are for the first row, [1, COL+1) for the second, and so on.
				uint8_t ID_row = (ID_reg >> i) & 0xFF;
				loop_tile_col:
				for (int j = 0; j < COL; j++) {
					//For column group c, we read features c*COL to (c+1)*COL.
					int feature = feature_stream[c*COL + j];
					if (ID_row & (1 << j))
						encHV_partial[i] += feature;
					else
						encHV_partial[i] -= feature;
				}
			}
			//After the first window, we move the window by right.
			//The initial 2*ROW ID block has enough bits for ROW/COL consecutive windows (as each window needs ROW+COL bits, not 2*ROW bits).
			//Otherwise, we update the ID address to get the new required ID bits.
			cycle += 1;
			if (cycle == ROW/COL) {
				cycle = 0;
				addr1 = addr1 + 1;
				addr2 = addr2 + 1;
				if (addr1 == Dhv/ROW)
					addr1 = 0;
				if (addr2 == Dhv/ROW)
					addr2 = 0;
				ID_reg = (((uint64_t) ID[addr2]) << 32) | ((uint64_t) ID[addr1]);
			}
			//We have not reached the bound of ROW/COL, so the ID register contains the needed bits.
			//Just shift right by COL, so 'ID_reg.range(i+COL-1, i)' gives the correct ID bits per each row i of the ID block.
			//E.g., in a 4x2 window, in the first cycle we need bits 0-1 for row 1, while in the next cycle we need bits 2-3, so shift by COL=2 is needed.
			else {
				ID_reg = (ID_reg >> COL);
			}
		}
		//Output the ROW generated dimensions for subsequent pipelined search.
		//Note that we use quantized random projection. Otherwise, we will need higher bit-width for classes (and tmp resgiter during dot-product).
		loop_enc_stream:
		for (int i = 0; i < ROW; i++) {
			if (encHV_partial[i] >= 0)
				enc_stream[i * Dhv / ROW + r] = 1;
			else
				enc_stream[i * Dhv / ROW + r] = -1;
		}
	}
}

/*
 * searchUnit is the major component of our implementation. It runs EPOCH times over the data (EPOCH=1 for inference).
 * In the first epoch, it reads encoding elements ROW by ROW (ROW elements in Div/COL cycles) from the encodeUnit unit, and stores them in the global memory to reuse in later epochs (in case of training).
 * In the remaining epochs, it reads the encoded hypervector from the global memory. Upon reading ROW dimensions, it compares (similarity checking) them with the corresponding dimensions of all classes.
 * After finding the class with highest score during retraining, the model updates in case of misprediction.
 *
 * enc_stream (input): ROW parallel stream of bipolar (+1, -1) dimensions from encoding unit.
 * classHV_gmem (input/output): class hypervectors; output in case of training, and input in case of inference.
 * labels_gmem (input/output): label of data samples; input in case of training, and output in case of inference.
 * encHV_gmem (input/output): interface to write/read encoded hypervectors to/from the DRAM to reuse encoded data.
 * train (input): number of training epochs (0 = inference)
 * size (input): number of data samples.
 *
 * searchUnitFirstEpoch runs searchUnit for the first epoch, searchUnitRestEpochs runs searchUnit for the rest of the epochs.
 * These are kept separate because of how searchUnit reads from the FIFOs.
 */
void searchUnitFirstEpoch(int *__restrict enc_stream, int *__restrict classHV_gmem, int *__restrict labels_gmem, HyperVector512 *__restrict encHV_gmem, int train, int size, int iter_read, int *__restrict encHV_partial, int *__restrict dotProductRes, float *__restrict norm2_inv, uint32_t *__restrict encHV_full) {
	if (iter_read == 0) {
		//Initialize the class hypervectors.
		loop_initClass:
		for (int i = 0; i < N_CLASS; i++) {
			for (int dim = 0; dim < Dhv; dim++) {
				//For inference, class hypervectors are given.
				//For training, initialize to zero.
				if (train > 0)
					classHV_gmem[i * Dhv + dim] = 0;
			}
		}
		
		//At the beginning of each epoch, calculate 1/|C|_2 (we call "1/|C|_2" as norm2).
		loop_norm_1:
		for (int i_class = 0; i_class < N_CLASS; i_class++) {
			uint64_t total = 0;
			loop_norm_2:
			for (int dim = 0; dim < Dhv; dim++) {
				total += classHV_gmem[i_class * Dhv + dim] * classHV_gmem[i_class * Dhv + dim];
			}
			//Total might be 0 before the first round of training, or if some class didn't have any sample,
			//So we use 1/|C|_2 = 0 to make its similarity (H*C*1/|C|_2) score 0 (although similarity checking won't be actually used in the first round of training).
			if (total == 0)
				norm2_inv[i_class] = 0;
			else {
				norm2_inv[i_class] = 1.0 / float(total);
			}
		}
	}

	//cout << "norm2_inv[0]: " << norm2_inv[0] << endl;

	int label;

	//For inference we do not need to read the label.
	if (train > 0)
		label = labels_gmem[iter_read];

	//Reset the dotProductRes (score buffer) before each input sample.
	loop_clear:
	for (int i_class = 0; i_class < N_CLASS; i_class++) {
		dotProductRes[i_class] = 0;
	}
	//In the first EPOCH, will read Dhv encoding dimensions, ROW by ROW (ROW dimensions per Dhv/COL cycles).
	//i_dim keeps track of the global index of classes (increases by ROW after processing a block of ROW rows).
	loop_outer:
	for (int i_dim = 0; i_dim < Dhv/ROW; ++i_dim) {
		uint32_t temp_partial = encHV_full[i_dim];
		loop_stream:
		for (int j_sub = 0; j_sub < ROW; j_sub++) {
			encHV_partial[j_sub] = enc_stream[j_sub * Dhv / ROW + i_dim];
		}
		//In the first epoch of TRAINING, initialize the classes, and store the encoded hypervector.
		if (train > 0) {
			uint32_t temp_partial;
			loop_init:
			for (int j_sub = 0; j_sub < ROW; j_sub++) {
				classHV_gmem[label * Dhv + i_dim*ROW + j_sub] += encHV_partial[j_sub];
				//store the dimensions (in a whole hypervector) and save to global memory for reuse in next epochs.
				//temp_partial[j_sub] = encHV_partial[j_sub] == 1 ? 1 : 0; //Bipolar to binary conversion.
				if (encHV_partial[j_sub] == 1) {
					temp_partial |= 1 << j_sub;
				} else {
					temp_partial &= ~(1 << j_sub);
				}
			}
			encHV_full[i_dim] = temp_partial;
		} else {
			//Multiply the generated ROW encoding dimensions to the corresponding class hypervectors.
			loop_score:
			for (int j_class = 0; j_class < N_CLASS; j_class++) {
				////#pragma HLS PIPELINE
				//#pragma HLS UNROLL
				loop_inner:
				for (int k_sub = 0; k_sub < ROW; k_sub++) {
					//#pragma HLS UNROLL factor=ROW
					//i_dim keeps track of the global index of classes (increases by ROW after processing a block of ROW rows).
					dotProductRes[j_class] += encHV_partial[k_sub] * classHV_gmem[j_class * Dhv + i_dim*ROW + k_sub];
				}
			}
		}
	}
	//Calculate max index (needed in inference and REtraining iterations, but we do it in case of initial training too, to avoid if/else...).
	int maxIndex = -1;
	float maxVal = -(1 << 15);
	loop_max:
	for (int i_class = 0; i_class < N_CLASS; i_class++) {
		//Here is the tricky part; I replace H*C/sqrt(|C|_2) by (H*C)^2/|C|_2, while considering the sign of H*C.
		float temp = dotProductRes[i_class]*norm2_inv[i_class];
		float score = temp * dotProductRes[i_class];
		if (dotProductRes[i_class] < 0)
			score = -score;
		if (score > maxVal) {
			maxIndex = i_class;
			maxVal = score;
		}
	}

	//If inference, output the index (label) of the class with maximum similarity score.
	if (train == 0) {
		labels_gmem[iter_read] = maxIndex;
	}
	//Processing an input sample ends here. Write the encoded hypervector to global memory in the FIRST epoch, in case of training.
	if (train > 0) {
		loop_writeEncHV:
		for (int i = 0; i < Dhv/512; i++) {
			HyperVector512 enc_512b;
			for (int j = 0; j < 512/ROW; j += 1) {
				//enc_512b.range(j*ROW+ROW-1, j*ROW) = encHV_full[(i*512/ROW) + j];
				enc_512b.buf[j] = encHV_full[(i*512/ROW) + j];
			}
			encHV_gmem[iter_read*(Dhv/512) + i] = enc_512b;
		}
	}

}

void searchUnitRestEpochs(int *__restrict classHV_gmem, int *__restrict labels_gmem, HyperVector512 *__restrict encHV_gmem, int train, int size, int *__restrict encHV_partial, int *__restrict dotProductRes, float *__restrict norm2_inv, uint32_t *__restrict encHV_full) {
	//At the beginning of each epoch, calculate 1/|C|_2 (we call "1/|C|_2" as norm2).
	loop_norm_1:
	for (int i_class = 0; i_class < N_CLASS; i_class++) {
		uint64_t total = 0;
		loop_norm_2:
		for (int dim = 0; dim < Dhv; dim++) {
			total += classHV_gmem[i_class * Dhv + dim] * classHV_gmem[i_class * Dhv + dim];
		}
		//Total might be 0 before the first round of training, or if some class didn't have any sample,
		//So we use 1/|C|_2 = 0 to make its similarity (H*C*1/|C|_2) score 0 (although similarity checking won't be actually used in the first round of training).
		if (total == 0)
			norm2_inv[i_class] = 0;
		else {
			norm2_inv[i_class] = 1.0 / float(total);
		}
	}

	//cout << "norm2_inv[0]: " << norm2_inv[0] << endl;

	loop_inputs:
	for (int iter_read = 0; iter_read < size; iter_read++) {

		//For inference we do not need to read the label.
		int label = labels_gmem[iter_read];

		//Reset the dotProductRes (score buffer) before each input sample.
		loop_clear:
		for (int i_class = 0; i_class < N_CLASS; i_class++) {
			dotProductRes[i_class] = 0;
		}
		//In the subsequent training epochs (i.e., retraining), we just reuse the encoding hypervectors generated in the first epoch.
		loop_read_encHV:
		for (int i = 0; i < Dhv/512; i++) {
			HyperVector512 enc_512b = encHV_gmem[iter_read*(Dhv/512) + i];
			for (int j = 0; j < 512/ROW; j += 1) {
				//encHV_full[(i*512/ROW) + j] = enc_512b.range(j*ROW+ROW-1, j*ROW);
				encHV_full[(i*512/ROW) + j] = enc_512b.buf[j];
			}
		}
		//In the first EPOCH, will read Dhv encoding dimensions, ROW by ROW (ROW dimensions per Dhv/COL cycles).
		//i_dim keeps track of the global index of classes (increases by ROW after processing a block of ROW rows).
		loop_outer:
		for (int i_dim = 0; i_dim < Dhv/ROW; i_dim += 1) {
			uint32_t temp_partial = encHV_full[i_dim];
			loop_stream:
			for (int j_sub = 0; j_sub < ROW; j_sub++) {
				encHV_partial[j_sub] = temp_partial & (1 << j_sub) ? 1 : -1; //Binary to bipolar conversion.
			}
			//In the first epoch of TRAINING, initialize the classes, and store the encoded hypervector.
			//In the next training epochs and/or in inference, calculate the similarity scores.
			//Multiply the generated ROW encoding dimensions to the corresponding class hypervectors.
			loop_score:
			for (int j_class = 0; j_class < N_CLASS; j_class++) {
				loop_inner:
				for (int k_sub = 0; k_sub < ROW; k_sub++) {
					//i_dim keeps track of the global index of classes (increases by ROW after processing a block of ROW rows).
					dotProductRes[j_class] += encHV_partial[k_sub] * classHV_gmem[j_class * Dhv + i_dim*ROW + k_sub];
				}
			}
		}
		//Calculate max index (needed in inference and REtraining iterations, but we do it in case of initial training too, to avoid if/else...).
		int maxIndex = -1;
		float maxVal = -(1 << 15);
		loop_max:
		for (int i_class = 0; i_class < N_CLASS; i_class++) {
			//Here is the tricky part; I replace H*C/sqrt(|C|_2) by (H*C)^2/|C|_2, while considering the sign of H*C.
			float temp = dotProductRes[i_class]*norm2_inv[i_class];
			float score = temp * dotProductRes[i_class];
			if (dotProductRes[i_class] < 0)
				score = -score;
			if (score > maxVal) {
				maxIndex = i_class;
				maxVal = score;
			}
		}

		//If it is a REtraining epoch, update the correct and mispredicted class.
		if (maxIndex != label) {
			loop_update:
			for (int i_sub = 0; i_sub < Dhv/ROW; i_sub++) {
				uint32_t temp_partial = encHV_full[i_sub];
				for (int j = 0; j < ROW; j++) {
					//int temp_dim = temp_partial[j] == 1 ? 1 : -1;
					int temp_dim = temp_partial & (1 << j) ? 1 : -1;
					classHV_gmem[label * Dhv + i_sub*ROW + j] += temp_dim;
					classHV_gmem[maxIndex * Dhv + i_sub*ROW + j] -= temp_dim;
					//classHV_gmem[label * Dhv + i_sub*ROW + j] = 42;
					//classHV_gmem[maxIndex * Dhv + i_sub*ROW + j] = 24;
				}
			}
		}	
	}
}

void top(int *__restrict input_gmem, std::size_t input_gmem_size, int *__restrict ID_gmem, std::size_t ID_gmem_size, int *__restrict classHV_gmem, std::size_t classHV_gmem_size, int *__restrict labels_gmem, std::size_t labels_gmem_size, HyperVector512 *__restrict encHV_gmem, std::size_t encHV_gmem_size, int train, int size) {
	int feature_stream[N_FEAT_PAD];

	//For now, the encoding stream is integer while we are using bipolar (+1, -1) encoding. Fix it later.
	int enc_stream[Dhv];

	//We have a seed ID of Dhv length, and we partition it to Dhv/ROW pieces of ROW bits as we operate on ROW rows at the same time.
	uint32_t ID[Dhv/ROW];

	//Initialize the seed ID hypervector.
	int offset = 0;
	static_assert(ROW == 32, "In the Hetero-C++ port, ROW must be 32!");
	loop_initID:
	for (int i = 0; i < Dhv/ROW; i++) {
		uint32_t ID_int = ID_gmem[i];
		ID[i] = ID_int;
		/*
		//If ROW is smaller than 32, each IDarray will fill several ID elements.
		if (ROW < 32) {
			for (int j = 0; j < 32/ROW; j++) {
				ID[i*32/ROW + j] = ID_int.range((j+1)*ROW - 1, j*ROW);
			}
		}//Otherwise, for each ID element, we need to read several IDarray elements.
		else {
			ID[i*32/ROW].range(32*offset + 31, 32*offset) = ID_int;
			offset += 1;
			if (offset == ROW/32)
				offset = 0;
		}
		*/
	}

	//Explained previously: to operate on ROW encoding dimensions per cycle.
	int encHV_partial[ROW];

	//To store the dot-product of the classes with the encoding hypervector.
	int dotProductRes[N_CLASS];

	//For cosine, we need to store 1/|C|_2, which are small fractional numbers. For now we use float, though we may change to ap_fixed.
	float norm2_inv[N_CLASS];

	//During retraining, we will need the encoded hypervector from global memory (as we generated and stored them in the first epoch).
	//I tried replacing encHV_full with a 1-d array (i.e., dt_int2 encHV_full[Dhv] with cyclic partitioning) but latency of search increased 50%.
	//As a result of using 2-d array, there will be some annoying temp variables to read/write data from/to encHV_full within the code.
	uint32_t encHV_full[Dhv/ROW];

	//We partition each class dimensions into ROW elements to match the ROW generated dimensions.

	for (int iter_read = 0; iter_read < size; iter_read++) {
		inputStream(input_gmem, feature_stream, size, iter_read);
		encodeUnit(feature_stream, ID, enc_stream, size, iter_read);
		searchUnitFirstEpoch(enc_stream, classHV_gmem, labels_gmem, encHV_gmem, train, size, iter_read, encHV_partial, dotProductRes, norm2_inv, encHV_full);
	}
	for (int epoch = 1; epoch < train; ++epoch) {
		searchUnitRestEpochs(classHV_gmem, labels_gmem, encHV_gmem, train, size, encHV_partial, dotProductRes, norm2_inv, encHV_full);
	}
}

/*
 * input_gmem (input): input data port; each feature is quantized to an integer.
 * ID_gmem (input): seed ID hypervector, packed to ints.
 * classHV_gmem (input/output): class hypervectors; output in case of training, and input in case of inference.
 * labels_gmem (input/output): label of data samples; input in case of training, and output in case of inference.
 * encHV_gmem (input/output): interface to write/read encoded hypervectors to/from the DRAM to reuse encoded data.
 * train (input): number of training epochs (0 = inference)
 * size (input): number of data samples.
 */
void hd(int *__restrict input_gmem, std::size_t input_gmem_size, int *__restrict ID_gmem, std::size_t ID_gmem_size, int *__restrict classHV_gmem, std::size_t classHV_gmem_size, int *__restrict labels_gmem, std::size_t labels_gmem_size, HyperVector512 *__restrict encHV_gmem, std::size_t encHV_gmem_size, int train, int size) {
#ifdef HPVM
	void *hd_Section = __hetero_section_begin();
	void *hd_Wrapper = __hetero_task_begin(
					/* Num Input Pairs */ 7,
					input_gmem, input_gmem_size, 
					ID_gmem, ID_gmem_size, 					
					classHV_gmem, classHV_gmem_size,
					labels_gmem, labels_gmem_size,
					encHV_gmem, encHV_gmem_size,
					train,
					size,
					/* Num Output Pairs */ 5,
					input_gmem, input_gmem_size, 
					ID_gmem, ID_gmem_size, 					
					classHV_gmem, classHV_gmem_size,
					labels_gmem, labels_gmem_size,
					encHV_gmem, encHV_gmem_size,
					/* Optional Node Name */ "hd_task");

	void *top_Section = __hetero_section_begin();
	void *top_Wrapper = __hetero_task_begin(
					/* Num Input Pairs */ 7,
					input_gmem, input_gmem_size, 
					ID_gmem, ID_gmem_size, 					
					classHV_gmem, classHV_gmem_size,
					labels_gmem, labels_gmem_size,
					encHV_gmem, encHV_gmem_size,
					train,
					size,
					/* Num Output Pairs */ 5,
					input_gmem, input_gmem_size, 
					ID_gmem, ID_gmem_size, 					
					classHV_gmem, classHV_gmem_size,
					labels_gmem, labels_gmem_size,
					encHV_gmem, encHV_gmem_size,
					/* Optional Node Name */ "top_task");
	__hpvm__hint(DEVICE);
#endif

	top(input_gmem, input_gmem_size, ID_gmem, ID_gmem_size, classHV_gmem, classHV_gmem_size, labels_gmem, labels_gmem_size, encHV_gmem, encHV_gmem_size, train, size);

#ifdef HPVM
	__hetero_task_end(top_Wrapper);
	__hetero_section_end(top_Section);

	__hetero_task_end(hd_Wrapper);
	__hetero_section_end(hd_Section);
#endif
}
