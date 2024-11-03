#include <heterocc.h>
#include <hpvm_hdc.h>
#include "DFG.hpp"
#include "host.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include "mnist_reader.hpp"

#ifdef NODFG
#define POSTFIX ".cublas.txt"
#else
#define POSTFIX ".cpu.txt"
#endif

using namespace std;

// #define MNIST

extern "C" void cu_rt_dump_float_hv(void *hv, size_t row, const char * filename);
extern "C" void cu_rt_dump_float_hm(void *hv, size_t row, size_t col, const char * filename);

template <typename T>
void datasetRead(std::vector<T> &data, std::string path){
	std::ifstream file_(path, std::ios::in | std::ios::binary);
	assert(file_.is_open() && "Couldn't open file!");
	T size;
	file_.read((char*)&size, sizeof(size));
	T temp;
	for(int i = 0; i < size; i++){
		file_.read((char*)&temp, sizeof(temp));
		data.push_back(temp);
	}
	file_.close();
}

template <typename T>
T initialize_rp_seed(size_t loop_index_var) {
	int i = loop_index_var / 32;
	int j = loop_index_var % 32;

	//std::cout << i << " " << j << "\n";
    
	long double temp = log2(i+2.5) * pow(2, 31);
	long long int temp2 = (long long int)(temp);
	temp2 = temp2 % 2147483648;

	int ele = temp2 & (0x01 << j); //temp2 && (0x01 << j);
	return ele ? (T) 1 : (T) -1;
}

template <typename T>
T copy(T* vec, size_t loop_index_var) {
	return vec[loop_index_var];
}

template <typename T>
T copy_no_padding(T* vec, size_t loop_index_var) {
	// The given data vectors are strided by N_FEAT_PAD elements, even though the vectors are each N_FEAT elements in length.
    return vec[loop_index_var + (N_FEAT_PAD - N_FEAT) * (loop_index_var / N_FEAT)];
}

template <typename T>
T zero(size_t loop_index_var) {
	return 0;
}

extern "C" float run_hd_classification(int EPOCH, __hypermatrix__<Dhv, N_FEAT, hvtype>* rp_matrix_buffer, __hypermatrix__<N_SAMPLE, N_FEAT, hvtype>* training_input_vectors, __hypermatrix__<N_TEST, N_FEAT, hvtype>* inference_input_vectors, int* training_labels, int* y_test);

int main(int argc, char** argv) {
#ifndef NODFG
	__hpvm__init();
#endif

	auto t_start = std::chrono::high_resolution_clock::now();

	srand(time(NULL));

	int EPOCH = std::atoi(argv[1]);
   
#ifdef QUANT
	std::vector<int> X_train; // X data. 
	std::vector<int> y_train; // LABELS
	
	datasetRead(X_train, X_train_path);
	datasetRead(y_train, y_train_path);

	std::vector<int> X_test;
	std::vector<int> y_test;
	
	datasetRead(X_test, X_test_path);
	datasetRead(y_test, y_test_path);
#else
	std::vector<float> X_train; // X data. 
	std::vector<int> y_train; // LABELS
	
	datasetRead(X_train, X_train_path);
	datasetRead(y_train, y_train_path);

	std::vector<float> X_test;
	std::vector<int> y_test;
	
	datasetRead(X_test, X_test_path);
	datasetRead(y_test, y_test_path);

#endif


#ifdef MNIST
	std::cout << "mnist dataset" << std::endl;

	auto dataset =
      mnist::read_dataset<std::vector, std::vector, int16_t, uint8_t>("../dataset/mnist");

	std::vector<int16_t> test_vec;

	for (const auto& row : dataset.test_images) {
		test_vec.insert(test_vec.end(), row.begin(), row.end());
	}

	std::vector<int16_t> training_vec;

	for (const auto& row : dataset.training_images) {
		training_vec.insert(training_vec.end(), row.begin(), row.end());
	}

	int16_t* inference_input_vectors_cpu = test_vec.data();
	size_t inference_input_size = training_vec.size() * sizeof(int16_t);

	int16_t* training_input_vectors_cpu = training_vec.data();

	// N_FEAT is number of entries per vector
	size_t input_vector_size = N_FEAT * sizeof(int16_t); // Size of a single vector

	// ===LABELS===
	// TODO: Check Datatype

	std::cout << "casting labels" << std::endl;
	std::vector<int> temp_training_labels;

	// Training samples
	for (int i = 0; i < N_SAMPLE; i++) {
		temp_training_labels.push_back((int) dataset.training_labels[i]);
	}

	std::vector<int> temp_test_labels;

	// Training samples
	for (int i = 0; i < N_TEST; i++) {
		temp_test_labels.push_back((int) dataset.test_labels[i]);
	}

	assert(N_SAMPLE == 60000);
	assert(N_TEST == 10000);
	assert(N_FEAT == 28 * 28);
	
	int* training_labels = temp_training_labels.data(); // Get your training labels.
	int* inference_test_labels = temp_test_labels.data();

	size_t training_labels_size = N_SAMPLE * sizeof(int16_t);

	// INFERENCE DATA / TEST DATA
	int16_t inference_labels[N_TEST];
	size_t inference_labels_size = N_TEST * sizeof(int16_t);

	std::cout << "done dataset" << std::endl;

#else

	size_t X_train_samples = X_train.size() / N_FEAT_PAD;

	assert(X_train_samples == y_train.size() && "Incorrect number of training labels");
	assert(N_SAMPLE == y_train.size());
	assert(N_TEST == y_test.size());
	
	std::vector<hvtype> temp_vec(X_train.begin(), X_train.end());
	hvtype* training_input_vectors_cpu = temp_vec.data();

	// N_FEAT is number of entries per vector
	size_t input_vector_size = N_FEAT * sizeof(hvtype); // Size of a single vector

	int* training_labels = y_train.data(); // Get your training labels.
	// N_SAMPLE is number of input vectors
	size_t training_labels_size = N_SAMPLE * sizeof(int);

	// INFERENCE DATA / TEST DATA
	//int inference_test_labels[N_TEST];
	size_t inference_labels_size = N_TEST * sizeof(int);
	int* inference_test_labels = y_test.data();


	std::vector<hvtype> temp_vec2(X_test.begin(), X_test.end());
	hvtype* inference_input_vectors_cpu = temp_vec2.data();
	assert((temp_vec2.size() / N_FEAT_PAD) == N_TEST && "Incorrect number of tests");

#endif

	auto t_elapsed = std::chrono::high_resolution_clock::now() - t_start;
	long mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_elapsed).count();
	long mSec1 = mSec;

	t_start = std::chrono::high_resolution_clock::now();

	__hypermatrix__<N_SAMPLE, N_FEAT, hvtype> training_input_vectors = __hetero_hdc_create_hypermatrix<N_SAMPLE, N_FEAT, hvtype>(1, (void*) copy_no_padding<hvtype>, training_input_vectors_cpu);
	__hypermatrix__<N_TEST, N_FEAT, hvtype> inference_input_vectors = __hetero_hdc_create_hypermatrix<N_TEST, N_FEAT, hvtype>(1, (void*) copy_no_padding<hvtype>, inference_input_vectors_cpu);

	// Encoding matrix: First we write into rp_matrix_transpose, then transpose it to get rp_matrix,
	// which is the correct dimensions for encoding input features.
	size_t rp_matrix_size = N_FEAT * Dhv * sizeof(hvtype);
	__hypervector__<Dhv, hvtype> rp_seed = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) initialize_rp_seed<hvtype>);	
	// Dhv needs to be greater than N_FEAT for the orthognality to hold.
	__hypermatrix__<Dhv, N_FEAT, hvtype> rp_matrix_buffer = __hetero_hdc_create_hypermatrix<Dhv, N_FEAT, hvtype>(0, (void *) zero<hvtype>);
	__hypermatrix__<N_FEAT, Dhv, hvtype> shifted_buffer = __hetero_hdc_create_hypermatrix<N_FEAT, Dhv, hvtype>(0, (void *) zero<hvtype>);
#ifndef NODFG
    void* GenRPMatDAG = __hetero_launch((void*) gen_rp_matrix<Dhv,  N_FEAT>, 3, /* Input Buffers: 3*/ __hetero_hdc_get_handle(rp_seed), sizeof(hvtype) * Dhv, __hetero_hdc_get_handle(shifted_buffer), sizeof(hvtype) * (N_FEAT * Dhv), __hetero_hdc_get_handle(rp_matrix_buffer), sizeof(hvtype) * (N_FEAT * Dhv), /* Output Buffers: 1*/ 1, __hetero_hdc_get_handle(rp_matrix_buffer), sizeof(hvtype) * (N_FEAT * Dhv));

    __hetero_wait(GenRPMatDAG);
#else
    gen_rp_matrix<Dhv, N_FEAT>(__hetero_hdc_get_handle(rp_seed), sizeof(hvtype) * Dhv, __hetero_hdc_get_handle(shifted_buffer), sizeof(hvtype) * (N_FEAT * Dhv), __hetero_hdc_get_handle(rp_matrix_buffer), sizeof(hvtype) * (N_FEAT * Dhv));
#endif

	float test_accuracy = run_hd_classification(EPOCH, 
		__hetero_hdc_get_handle(rp_matrix_buffer), 
		__hetero_hdc_get_handle(training_input_vectors),
		__hetero_hdc_get_handle(inference_input_vectors), 
		training_labels, 
		inference_test_labels);
    
	t_elapsed = std::chrono::high_resolution_clock::now() - t_start;
	mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_elapsed).count();

	std::cout << "Overall benchmark took " << mSec << " mSec" << std::endl;
	std::cout << "Test Accuracy: " << test_accuracy << std::endl;

#ifndef NODFG
	__hpvm__cleanup();
#endif	
	return 0;
}
void __attribute__ ((noinline)) l2norm(__hypervector__<N_CLASS, hvtype> *norms_buffer, __hypermatrix__<N_CLASS, Dhv, hvtype> *classes) {
#ifdef NODFG
	// *norms_buffer = __hetero_hdc_l2norm<N_CLASS, Dhv, hvtype>(*classes);
#else

        void* InitDAG = __hetero_launch(
                (void*) L2NormDFG<N_CLASS, Dhv>, 
                2,
                norms_buffer,
                sizeof(hvtype) * N_CLASS,
                classes,
                sizeof(hvtype) * N_CLASS * Dhv,
                1,
                norms_buffer,
                sizeof(hvtype) * N_CLASS
                );

        __hetero_wait(InitDAG);

#endif
}


void __attribute__ ((noinline)) InitializeClasses(__hypervector__<Dhv, hvtype> *encoded_hvs, __hypermatrix__<N_CLASS, Dhv, hvtype> *classes, int label ,int index) {
#ifdef NODFG
#else


        void* InitDAG = __hetero_launch(
                (void*) InitialUpdateDFG< N_CLASS, N_SAMPLE, Dhv>, 4, /* Input Buffers: 4*/ 
                encoded_hvs, sizeof(hvtype) * Dhv, 
                classes, sizeof(hvtype) * (N_CLASS * Dhv), 
                label, index,
                1, /* Output Buffers */ 
                classes, sizeof(hvtype) * (N_CLASS * Dhv)
                );

        __hetero_wait(InitDAG);

#endif
}

extern "C" float run_hd_classification(int EPOCH, __hypermatrix__<Dhv, N_FEAT, hvtype>* rp_matrix_buffer, __hypermatrix__<N_SAMPLE, N_FEAT, hvtype>* training_input_vectors, __hypermatrix__<N_TEST, N_FEAT, hvtype>* inference_input_vectors, int* training_labels, int* y_test) {
	size_t rp_matrix_size = N_FEAT * Dhv * sizeof(hvtype);
	size_t input_vector_size = N_FEAT * sizeof(hvtype);
	size_t class_size = Dhv * sizeof(hvtype);
	size_t classes_size = N_CLASS * Dhv * sizeof(hvtype);
	size_t training_labels_size = N_SAMPLE * sizeof(int);
	size_t inference_labels_size = N_TEST * sizeof(int);
	size_t encoded_hv_size = Dhv * sizeof(hvtype);
	size_t update_hv_size = Dhv * sizeof(hvtype);
	size_t scores_size = N_CLASS * sizeof(hvtype);
	size_t norms_size = N_CLASS * sizeof(hvtype);

	//cu_rt_dump_float_hm(rp_matrix_buffer, Dhv, N_FEAT, "rp_matrix_buffer" POSTFIX);
	//cu_rt_dump_float_hm(training_input_vectors, N_SAMPLE, N_FEAT_PAD, "training_input_vectors" POSTFIX);
	//cu_rt_dump_float_hm(inference_input_vectors, N_TEST, N_FEAT_PAD, "inference_input_vectors" POSTFIX);

	__hypervector__<Dhv, hvtype> update_hv = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);
	__hypermatrix__<N_CLASS, Dhv, hvtype> classes = __hetero_hdc_create_hypermatrix<N_CLASS, Dhv, hvtype>(0, (void*) zero<hvtype>);
    auto classes_handle =__hetero_hdc_get_handle(classes);

	__hypermatrix__<N_SAMPLE, Dhv, hvtype> encoded_hvs = __hetero_hdc_create_hypermatrix<N_SAMPLE, Dhv, hvtype>(0, (void*) zero<hvtype>);
    auto encoded_hvs_handle = __hetero_hdc_get_handle(encoded_hvs);

	__hypervector__<Dhv, hvtype> encoded_hv_buffer = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);
	__hypervector__<N_CLASS, hvtype> scores_buffer = __hetero_hdc_create_hypervector<N_CLASS, hvtype>(0, (void*) zero<hvtype>);
	__hypervector__<N_CLASS, hvtype> norms_buffer = __hetero_hdc_create_hypervector<N_CLASS, hvtype>(0, (void*) zero<hvtype>);

	int inference_labels[N_TEST];

	// ============ Training ===============

	// Initialize class hvs.
	// No need to do encoding if using simulator. 
	if (0) {
	__hetero_hdc_encoding_loop(0, (void*) InitialEncodingDFG<Dhv, N_FEAT>, N_SAMPLE, N_CLASS, N_FEAT, N_FEAT, rp_matrix_buffer, rp_matrix_size, (hvtype *) training_input_vectors, input_vector_size, encoded_hvs_handle, class_size);

	// Set classes as sum of all 
	for (int i = 0; i < N_SAMPLE; i++) {
		int label = training_labels[i];
#ifndef NODFG
		auto encoded_hv = __hetero_hdc_get_matrix_row<N_SAMPLE, Dhv, hvtype>(encoded_hvs, N_SAMPLE, Dhv, i);

        auto encoded_hv_i_handle = __hetero_hdc_get_handle(encoded_hv);
        InitializeClasses(encoded_hv_i_handle, classes_handle, label, i);
#else
		auto class_hv = __hetero_hdc_get_matrix_row<N_CLASS, Dhv, hvtype>(classes, N_CLASS, Dhv, label);
		auto encoded_hv = __hetero_hdc_get_matrix_row<N_SAMPLE, Dhv, hvtype>(encoded_hvs, N_SAMPLE, Dhv, i);
		auto sum_hv = __hetero_hdc_sum<Dhv, hvtype>(class_hv, encoded_hv); 
		__hetero_hdc_set_matrix_row<N_CLASS, Dhv, hvtype>(classes, sum_hv, label); 
#endif
	}
	}
	//cu_rt_dump_float_hm(__hetero_hdc_get_handle(classes), N_CLASS, Dhv, "classes" POSTFIX);
	//cu_rt_dump_float_hm(__hetero_hdc_get_handle(encoded_hvs), N_SAMPLE, Dhv, "encoded_hvs" POSTFIX);

	int argmax[1];
	// Training generates classes from labeled data. 
	// ======= Training Rest Epochs ======= 

	{
	auto t_start = std::chrono::high_resolution_clock::now();

	// l2norm(__hetero_hdc_get_handle(norms_buffer), __hetero_hdc_get_handle(classes));
	
	__hetero_hdc_training_loop(22, (void*) training_root_node<Dhv, N_CLASS, N_SAMPLE, N_FEAT>, 
		EPOCH, N_SAMPLE, N_FEAT, N_FEAT, 
		rp_matrix_buffer, rp_matrix_size, 
		(hvtype *) training_input_vectors, input_vector_size, 
		__hetero_hdc_get_handle(classes), classes_size, 
		training_labels, training_labels_size, 
		__hetero_hdc_get_handle(encoded_hv_buffer), encoded_hv_size, 
		__hetero_hdc_get_handle(scores_buffer), scores_size, 
		__hetero_hdc_get_handle(norms_buffer), norms_size, 
		__hetero_hdc_get_handle(update_hv), update_hv_size, &argmax[0], sizeof(int));

	auto t_end = std::chrono::high_resolution_clock::now();
	long mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
	}
	//cu_rt_dump_float_hm(__hetero_hdc_get_handle(classes), N_CLASS, Dhv, "post_train_classes" POSTFIX);
	//cu_rt_dump_float_hv(__hetero_hdc_get_handle(scores_buffer), N_CLASS, "post_train_scores_buffer" POSTFIX);
	//cu_rt_dump_float_hv(__hetero_hdc_get_handle(norms_buffer), N_CLASS, "post_train_norms_buffer" POSTFIX);

	// ============ Inference =============== //

	{
	auto t_start = std::chrono::high_resolution_clock::now();

	// l2norm(__hetero_hdc_get_handle(norms_buffer), __hetero_hdc_get_handle(classes));
	__hetero_hdc_inference_loop(17, 
		(void*) inference_root_node<Dhv, N_CLASS, N_TEST, N_FEAT>, 
		N_TEST, N_FEAT, N_FEAT, 
		rp_matrix_buffer, rp_matrix_size, 
		(hvtype *) inference_input_vectors, input_vector_size, 
		__hetero_hdc_get_handle(classes), classes_size, 
		inference_labels, inference_labels_size, 
		__hetero_hdc_get_handle(encoded_hv_buffer), encoded_hv_size, 
		__hetero_hdc_get_handle(scores_buffer), scores_size, 
		__hetero_hdc_get_handle(norms_buffer), norms_size);

	auto t_end = std::chrono::high_resolution_clock::now();
	long mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
	}
	//cu_rt_dump_float_hm(__hetero_hdc_get_handle(classes), N_CLASS, Dhv, "post_infer_classes" POSTFIX);
	//cu_rt_dump_float_hv(__hetero_hdc_get_handle(scores_buffer), N_CLASS, "post_infer_scores_buffer" POSTFIX);
	//cu_rt_dump_float_hv(__hetero_hdc_get_handle(norms_buffer), N_CLASS, "post_infer_norms_buffer" POSTFIX);

	std::ofstream myfile("out" POSTFIX);

	int correct = 0;
	for(int i = 0; i < N_TEST; i++) {
		correct += inference_labels[i] == y_test[i]; 
		myfile << y_test[i] << " " << inference_labels[i] << std::endl;
	}

	return float(correct)/N_TEST;
}
