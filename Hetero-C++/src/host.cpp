#define HPVM 1

#ifdef HPVM
#include <heterocc.h>
#include <hpvm_hdc.h>
#include "DFG.hpp"
#endif
#include "host.h"
#include <vector>
#include <cassert>
#include <cmath>


#define DUMP(vec, suffix) {\
  FILE *f = fopen("dump/" #vec suffix, "w");\
  if (f) fwrite(vec.data(), sizeof(vec[0]), vec.size(), f);\
  if (f) fclose(f);\
}

template <int N, typename elemTy>
void print_hv(__hypervector__<N, elemTy> hv) {
    std::cout << "[";
    for (int i = 0; i < N-1; i++) {
        std::cout << hv[0][i] << ", ";
    }
    std::cout << hv[0][N-1] << "]\n";
    return;
}

void datasetBinaryRead(std::vector<int> &data, std::string path){
	std::ifstream file_(path, std::ios::in | std::ios::binary);
	assert(file_.is_open() && "Couldn't open file!");
	int32_t size;
	file_.read((char*)&size, sizeof(size));
	int32_t temp;
	for(int i = 0; i < size; i++){
		file_.read((char*)&temp, sizeof(temp));
		data.push_back(temp);
	}
	file_.close();
}
template <typename T>
T initialize_hv(T* datapoint_vector, size_t loop_index_var) {
	//std::cout << ((float*)datapoint_vector)[loop_index_var] << "\n";
	return datapoint_vector[loop_index_var];
}

template <typename T>
T initialize_rp_seed(size_t loop_index_var) {
	int i = loop_index_var / 32;
	int j = loop_index_var % 32;

	//std::cout << i << " " << j << "\n";
	long double temp = log2(i+2.5) * pow(2, 31);
	long long int temp2 = (long long int)(temp);
	temp2 = temp2 % 2147483648;
	//temp2 = temp2 % int(pow(2, 31));
	//2147483648;

	int ele = temp2 & (0x01 << j); //temp2 && (0x01 << j);

	//std::cout << ele << "\n";

	if (ele) {
		return (T) 1;
	}
	else {
		return (T) -1;
	}
}


int main(int argc, char** argv)
{
	__hpvm__init();

	auto t_start = std::chrono::high_resolution_clock::now();
	std::cout << "Main Starting" << std::endl;

	srand(time(NULL));

	int EPOCH = std::atoi(argv[1]);
   
	std::vector<int> X_train; // X data. 
	std::vector<int> y_train; // LABELS
	
	datasetBinaryRead(X_train, X_train_path);
	datasetBinaryRead(y_train, y_train_path);

	std::vector<int> X_test;
	std::vector<int> y_test;
	
	datasetBinaryRead(X_test, X_test_path);
	datasetBinaryRead(y_test, y_test_path);

	// FIXME, run inference on training dataset and make sure we get 100% accuracy.
	//datasetBinaryRead(X_test, X_train_path);
	//datasetBinaryRead(y_test, y_train_path);

	for (int i = 0; i < y_test.size(); i++) {
		std::cout << y_test[i] << " ";
	}
	std::cout << "Read Data Starting" << std::endl;

	srand(0);
	
	assert(N_SAMPLE == y_train.size());
	assert(N_TEST == y_test.size());

	// std::cout << y_test.size();
	
	// TRAINING DATA INITIALZIATION
	// FIXME: Should probably remove padding here, not durin gnode launches. 
	std::vector<hvtype> temp_vec(X_train.begin(), X_train.end());
	hvtype* training_input_vectors = temp_vec.data();
	// N_FEAT is number of entries per vector
	size_t input_vector_size = N_FEAT * sizeof(hvtype); // Size of a single vector

	int* training_labels = y_train.data(); // Get your training labels.
	// N_SAMPLE is number of input vectors
	size_t training_labels_size = N_SAMPLE * sizeof(int);

	// INFERENCE DATA / TEST DATA
	int inference_labels[N_TEST];
	size_t inference_labels_size = N_TEST * sizeof(int);

	// TRAINING DATA INITIALZIATION
	std::vector<hvtype> temp_vec2(X_test.begin(), X_test.end());
	hvtype* inference_input_vectors = temp_vec.data();
	// N_FEAT is number of entries per vector



	auto t_elapsed = std::chrono::high_resolution_clock::now() - t_start;
	long mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_elapsed).count();
	long mSec1 = mSec;
	std::cout << "Reading data took " << mSec << " mSec" << std::endl;

	t_start = std::chrono::high_resolution_clock::now();

	// Host allocated memory 
	__hypervector__<Dhv, hvtype> encoded_hv = __hetero_hdc_hypervector<Dhv, hvtype>();
	hvtype* encoded_hv_buffer = new hvtype[Dhv];
	size_t encoded_hv_size = Dhv * sizeof(hvtype);

	__hypervector__<Dhv, hvtype> update_hv = __hetero_hdc_hypervector<Dhv, hvtype>();
	//hvtype* update_hv_ptr = new hvtype[Dhv];
	size_t update_hv_size = Dhv * sizeof(hvtype);
	
	// Used to store a temporary class_hv for initializion
	__hypervector__<Dhv, hvtype> class_hv = __hetero_hdc_hypervector<Dhv, hvtype>();
	hvtype* class_buffer = new hvtype[Dhv];
	size_t class_size = Dhv * sizeof(hvtype);

	// Read from during classification
	__hypermatrix__<N_CLASS, Dhv, hvtype> classes = __hetero_hdc_create_hypermatrix<N_CLASS, Dhv, hvtype>(0, (void*) zero_hv<hvtype>);
	hvtype* classes_buffer = new hvtype[N_CLASS * Dhv];
	size_t classes_size = N_CLASS * Dhv * sizeof(hvtype);

	// Temporarily store scores, allows us to split score calcuation into a separte task.
	__hypervector__<Dhv, hvtype> scores = __hetero_hdc_hypervector<Dhv, hvtype>();
	hvtype* scores_buffer = new hvtype[N_CLASS];
	size_t scores_size = N_CLASS * sizeof(hvtype);

	// Encoding matrix: First we write into rp_matrix_transpose, then transpose it to get rp_matrix,
	// which is the correct dimensions for encoding input features.


	size_t rp_matrix_size = N_FEAT * Dhv * sizeof(hvtype);

	__hypervector__<Dhv, hvtype> rp_seed = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) initialize_rp_seed<hvtype>);	

	std::cout << "Dimension over 32: " << Dhv/32 << std::endl;
	//We need a seed ID. To generate in a random yet determenistic (for later debug purposes) fashion, we use bits of log2 as some random stuff.

	std::cout << "Seed hv:\n";
	//print_hv<Dhv, hvtype>(rp_seed);
	std::cout << "After seed generation\n";

	// Dhv needs to be greater than N_FEAT for the orthognality to hold.
	
#ifdef OFFLOAD_RP_GEN

	hvtype* rp_matrix_buffer = new hvtype[N_FEAT * Dhv];
	hvtype* shifted_buffer = new hvtype[N_FEAT * Dhv];
	hvtype* row_buffer = new hvtype[Dhv];

    void* GenRPMatDAG = __hetero_launch(
        (void*) gen_rp_matrix<Dhv,  N_FEAT>,
        4,
        /* Input Buffers: 3*/ 
        &rp_seed, sizeof(hvtype) * Dhv,
        row_buffer, sizeof(hvtype) * Dhv,
        shifted_buffer, sizeof(hvtype) * (N_FEAT * Dhv),
        rp_matrix_buffer, sizeof(hvtype) * (N_FEAT * Dhv),
        /* Output Buffers: 1*/ 
        1,
        rp_matrix_buffer, sizeof(hvtype) * (N_FEAT * Dhv)
    );

    __hetero_wait(GenRPMatDAG);

    free(shifted_buffer);
    free(row_buffer);



    //rp_matrix =   *  (__hypermatrix__<Dhv, N_FEAT, hvtype>*) rp_matrix_buffer;

#else

	// Generate the random projection matrix. Dhv rows, N_FEAT cols, so Dhv x N_FEAT.
	__hypermatrix__<N_FEAT, Dhv, hvtype> rp_matrix_transpose = __hetero_hdc_hypermatrix<N_FEAT, Dhv, hvtype>();
	__hypervector__<Dhv, hvtype> row = __hetero_hdc_hypervector<Dhv, hvtype>();

	// Each row is just a wrap shift of the seed.
	for (int i = 0; i < N_FEAT; i++) {
		row = __hetero_hdc_wrap_shift<Dhv, hvtype>(rp_seed, i);
		//print_hv<Dhv, hvtype>(row);
		__hetero_hdc_set_matrix_row<N_FEAT, Dhv, hvtype>(rp_matrix_transpose, row, i);
	} 

	// Now transpose in order to be able to multiply with input hv in DFG.
	__hypermatrix__<Dhv, N_FEAT, hvtype> rp_matrix = __hetero_hdc_matrix_transpose<N_FEAT, Dhv, hvtype>(rp_matrix_transpose, N_FEAT, Dhv);

	auto rp_matrix_buffer = &rp_matrix;
#endif

	// Confirm that there are equal amounts of each label:
	#if 0
	int counts [N_CLASS];

	for (int i = 0; i < N_CLASS; i++) {
		counts[i] = 0;
	}

	for (int i = 0; i < N_SAMPLE; i++) {
		int idx = training_labels[i];
		counts[idx] += 1;
	}

	for (int i = 0; i < N_CLASS; i++) {
		std::cout << i << " " << counts[i] <<std::endl;
	}
	#endif


	// ============ Training ===============

	// Initialize class hvs.
	// FIXME: Verify that the classes matrix is set to 0 by the compiler.
	std::cout << "Init class hvs:" << std::endl;
	// TODO: Move to DAG.
	for (int i = 0; i < N_SAMPLE; i++) {
		__hypervector__<N_FEAT, hvtype> datapoint_hv = __hetero_hdc_create_hypervector<N_FEAT, hvtype>(1, (void*) initialize_hv<hvtype>, training_input_vectors + i * N_FEAT_PAD);

		// Encode each input datapoitn
		void* initialize_DFG = __hetero_launch(
			(void*) InitialEncodingDFG<Dhv, N_FEAT>, //FIXME: Make this a copy. 
			2 + 1,
			/* Input Buffers: 2*/ 
			&rp_matrix, rp_matrix_size, //false,
			&datapoint_hv, input_vector_size,
			/* Output Buffers: 1*/ 
			&encoded_hv, class_size,  //false,
			1,
			&encoded_hv, class_size //false
		);

		__hetero_wait(initialize_DFG);

		int label = training_labels[i];

		// rp_encoding_node encodes a single encoded_hv, which we then have to accumulate to our big group of classes in class_hv[s].

		//print_hv<Dhv, hvtype>(encoded_hv);

		// accumulate each encoded hv to its corresponding class.
		// FIXME: Should this be a dfg?? 
		update_hv =  __hetero_hdc_get_matrix_row<N_CLASS, Dhv, hvtype>(classes, N_CLASS, Dhv, label);
		update_hv = __hetero_hdc_sum<Dhv, hvtype>(update_hv, encoded_hv); 
		__hetero_hdc_set_matrix_row<N_CLASS, Dhv, hvtype>(classes, encoded_hv, label); 
		//print_hv<Dhv, hvtype>(update_hv); //TODO: Maybe there should be a _hdc_sign applied here.
	}

	std::cout << "Done init class hvs:" << std::endl;

	#if 0
	for (int i = 0; i < N_CLASS; i++) {
		__hypervector__<Dhv, hvtype> class_temp = __hetero_hdc_get_matrix_row<N_CLASS, Dhv, hvtype>(classes, N_CLASS, Dhv, i);
		std::cout << i << " ";
		print_hv<Dhv, hvtype>(class_temp);
	}
	#endif

	int argmax[1];

	// Training generates classes from labeled data. 
	// ======= Training Rest Epochs ======= 
	for (int i = 0; i < EPOCH; i++) {
		// Can we normalize the hypervectors here or do we have to do that in the DFG.
		std::cout << "Epoch: #" << i << std::endl;
		for (int j = 0; j < N_SAMPLE; j++) {

			//printf("before creat hv\n");
			__hypervector__<N_FEAT, hvtype> datapoint_hv = __hetero_hdc_create_hypervector<N_FEAT, hvtype>(1, (void*) initialize_hv<hvtype>, training_input_vectors + j * N_FEAT_PAD);

			//printf("before root launch\n");
			// Root node is: Encoding -> classing for a single HV.
			void *DFG = __hetero_launch(

				(void*) training_root_node<Dhv, N_CLASS, N_SAMPLE, N_FEAT>,

				/* Input Buffers: 4*/ 8,
				&rp_matrix, rp_matrix_size, //false,
				&datapoint_hv, input_vector_size, //true,
				&classes, classes_size, //false,
				/* Local Var Buffers 4*/
				encoded_hv_buffer, encoded_hv_size,// false,
				scores_buffer, scores_size,
                &update_hv, update_hv_size,
				&argmax[0], sizeof(int),
				training_labels[j], 
				/* Output Buffers: 1*/ 
				1,
				&classes, classes_size
			);
			__hetero_wait(DFG); 

			// Print out the class that this hv is labeled as. 

			//printf("after training root launch\n");
	
		}

		#if 0
		for (int i = 0; i < N_CLASS; i++) {
			__hypervector__<Dhv, hvtype> class_temp = __hetero_hdc_get_matrix_row<N_CLASS, Dhv, hvtype>(classes, N_CLASS, Dhv, i);
			std::cout << i << " ";
			print_hv<Dhv, hvtype>(class_temp);
		}
		#endif
	}

	std::ofstream training_file("training-classes.txt");

	#if DEBUG
	for(int i = 0; i < N_CLASS; i++) {

		// Basically, these should be CHANGING for the same class. 
		update_hv = __hetero_hdc_get_matrix_row<N_CLASS, Dhv, hvtype>(classes, N_CLASS, Dhv, i);
		print_hv<Dhv, hvtype>(update_hv);
		printf("label: %d\n", i);

	}
	#endif
	std::cout << "inference starting" << std::endl;

	// ============ Inference =============== //
	
	/* For each hypervector, inference calculates what class it is closest to and labels the hypervectorit accordingly.*/
	for (int j = 0; j < N_TEST; j++) {

			//std::cout << "Inference vec: #" << j << std::endl;

			__hypervector__<N_FEAT, hvtype> datapoint_hv = __hetero_hdc_create_hypervector<N_FEAT, hvtype>(1, (void*) initialize_hv<hvtype>, inference_input_vectors + j * N_FEAT_PAD);

			// Root node is: Encoding -> classing for a single HV.
			void *DFG = __hetero_launch(
				(void*) inference_root_node<Dhv, N_CLASS, N_TEST, N_FEAT>,
				/* Input Buffers: 3*/ 7,
				&rp_matrix, rp_matrix_size, //false,
				&datapoint_hv, input_vector_size, //true,
				&classes, classes_size, //false,
				/* Local Var Buffers 2*/
				encoded_hv_buffer, encoded_hv_size,// false,
				scores_buffer, scores_size,
				j, 
				/* Output Buffers: 1*/ 
				inference_labels + j, sizeof(int),
				1,
				inference_labels + j, sizeof(int) //, false
			);
			__hetero_wait(DFG); 

			//std::cout << "after root launch" << std::endl;
	
		}
	
	std::cout << "After Inference" << std::endl;

	t_elapsed = std::chrono::high_resolution_clock::now() - t_start;
	
	mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_elapsed).count();

	std::ofstream myfile("out.txt");

	int correct = 0;
	for(int i = 0; i < N_TEST; i++) {
		myfile << y_test[i] << " " << inference_labels[i] << std::endl;
		if(inference_labels[i] == y_test[i])
			correct += 1;
	}
    
	std::cout << "Test accuracy = " << float(correct)/N_TEST << std::endl;


	__hpvm__cleanup();
	return 0;
}




