#pragma once

#include <hpvm_hdc.h>
#include <heterocc.h>
#include <iostream>

//#define HAMMING_DIST

#undef D
#undef N_FEATURES
#undef K

typedef int binary;
typedef float hvtype;

// RANDOM PROJECTION ENCODING!!
// Matrix-vector mul
// Encodes a single vector using a random projection matrix
//
// RP encoding reduces N_features -> D 

template <typename T>
T zero_hv(size_t loop_index_var) {
	////std::cout << ((float*)datapoint_vector)[loop_index_var] << "\n";
	return 0;
}

template<int D, int N_FEATURES>
void rp_encoding_node(/* Input Buffers: 2*/
        __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
        __hypervector__<N_FEATURES, hvtype>* input_datapoint_ptr, size_t input_datapoint_size, // __hypervector__<N_FEATURES, int> 
        /* Output Buffers: 1*/
        __hypervector__<D, hvtype>* output_hv_ptr, size_t output_hv_size) { // __hypervector__<D, binary>
    
    void* section = __hetero_section_begin();

#if FGPA
    __hetero_hint(DEVICE);
#endif

    void* task = __hetero_task_begin(
        /* Input Buffers: 2*/ 3, rp_matrix_ptr, rp_matrix_size, input_datapoint_ptr, input_datapoint_size, output_hv_ptr, output_hv_size,
        /* Parameters: 0*/
        /* Output Buffers: 1*/ 1, output_hv_ptr, output_hv_size,
        "inner_rp_encoding_task"
    );

    ////std::cout << "encoding node" << std::endl;
    
#if FGPA
    __hetero_hint(DEVICE);
#endif
    
    __hypervector__<D, hvtype> encoded_hv = __hetero_hdc_create_hypervector<D, hvtype>(0, (void*) zero_hv<hvtype>);
    *output_hv_ptr = encoded_hv;

    encoded_hv = __hetero_hdc_matmul<D, N_FEATURES, hvtype>(*input_datapoint_ptr, *rp_matrix_ptr); 
    // Uses the output_hv_ptr for the buffer. So that we can lower to 
    // additional tasks. We should do an optimization in the bufferization
    // analysis to re-use the same buffer (especially those coming from the
    // formal parameters) to enable more of these tasks to become parallel loops.
    *output_hv_ptr = encoded_hv;
    
    __hypervector__<D, hvtype> bipolar_encoded_hv = __hetero_hdc_sign<D, hvtype>(encoded_hv);
    *output_hv_ptr = bipolar_encoded_hv;


    __hetero_task_end(task); 

    __hetero_section_end(section);
    return;
}

template<int D, int N_FEATURES>
void rp_encoding_node_copy(/* Input Buffers: 2*/
        __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
        __hypervector__<N_FEATURES, hvtype>* input_datapoint_ptr, size_t input_datapoint_size, // __hypervector__<N_FEATURES, int> 
        /* Output Buffers: 1*/
        __hypervector__<D, hvtype>* output_hv_ptr, size_t output_hv_size) { // __hypervector__<D, binary>
    
    void* section = __hetero_section_begin();

#if FGPA
    __hetero_hint(DEVICE);
#endif

    void* task = __hetero_task_begin(
        /* Input Buffers: 2*/ 3, rp_matrix_ptr, rp_matrix_size, input_datapoint_ptr, input_datapoint_size, output_hv_ptr, output_hv_size,
        /* Parameters: 0*/
        /* Output Buffers: 1*/ 1, output_hv_ptr, output_hv_size,
        "inner_rp_encoding_task"
    );

    ////std::cout << "encoding node" << std::endl;
    
#if FGPA
    __hetero_hint(DEVICE);
#endif
    
    __hypervector__<D, hvtype> encoded_hv = __hetero_hdc_create_hypervector<D, hvtype>(0, (void*) zero_hv<hvtype>);
    *output_hv_ptr = encoded_hv;

    encoded_hv = __hetero_hdc_matmul<D, N_FEATURES, hvtype>(*input_datapoint_ptr, *rp_matrix_ptr); 
    // Uses the output_hv_ptr for the buffer. So that we can lower to 
    // additional tasks. We should do an optimization in the bufferization
    // analysis to re-use the same buffer (especially those coming from the
    // formal parameters) to enable more of these tasks to become parallel loops.
    *output_hv_ptr = encoded_hv;
    
    __hypervector__<D, hvtype> bipolar_encoded_hv = __hetero_hdc_sign<D, hvtype>(encoded_hv);
    *output_hv_ptr = bipolar_encoded_hv;


    __hetero_task_end(task); 

    __hetero_section_end(section);
    return;
}

template<int D, int N_FEATURES>
void rp_encoding_node_copy_copy(/* Input Buffers: 2*/
        __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
        __hypervector__<N_FEATURES, hvtype>* input_datapoint_ptr, size_t input_datapoint_size, // __hypervector__<N_FEATURES, int> 
        /* Output Buffers: 1*/
        __hypervector__<D, hvtype>* output_hv_ptr, size_t output_hv_size) { // __hypervector__<D, binary>
    
    void* section = __hetero_section_begin();

#if FGPA
    __hetero_hint(DEVICE);
#endif

    void* task = __hetero_task_begin(
        /* Input Buffers: 2*/ 3, rp_matrix_ptr, rp_matrix_size, input_datapoint_ptr, input_datapoint_size, output_hv_ptr, output_hv_size,
        /* Parameters: 0*/
        /* Output Buffers: 1*/ 1, output_hv_ptr, output_hv_size,
        "inner_rp_encoding_task"
    );

    ////std::cout << "encoding node" << std::endl;
    
#if FGPA
    __hetero_hint(DEVICE);
#endif
    
    __hypervector__<D, hvtype> encoded_hv = __hetero_hdc_create_hypervector<D, hvtype>(0, (void*) zero_hv<hvtype>);
    *output_hv_ptr = encoded_hv;

    encoded_hv = __hetero_hdc_matmul<D, N_FEATURES, hvtype>(*input_datapoint_ptr, *rp_matrix_ptr); 
    // Uses the output_hv_ptr for the buffer. So that we can lower to 
    // additional tasks. We should do an optimization in the bufferization
    // analysis to re-use the same buffer (especially those coming from the
    // formal parameters) to enable more of these tasks to become parallel loops.
    *output_hv_ptr = encoded_hv;
    
    __hypervector__<D, hvtype> bipolar_encoded_hv = __hetero_hdc_sign<D, hvtype>(encoded_hv);
    *output_hv_ptr = bipolar_encoded_hv;


    __hetero_task_end(task); 

    __hetero_section_end(section);
    return;
}


// train == 0
// In the streaming implementation, this runs for each encoded HV, so N_VEC * EPOCHs times.

/* Just make guesses based on cossim  */
template<int D, int K, int N_VEC> // ONLY RUNS ONCE
void __attribute__ ((noinline)) classification_node_inference(
    __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, // __hypervector__<D, binary>
    __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, // __hypermatrix__<K, D, binary>
    __hypervector__<K, hvtype>* scores_ptr, size_t scores_size, // Used as Local var.
    int encoded_hv_idx,
    int* label_ptr, size_t label_size ) {   
    // Read classes hvs from host.

    // Calculate the similarity scores between this hv (encoded_hv_ptr) and class (classes_ptr) hypervectors.
    // Do once per hypervector:
     void* section = __hetero_section_begin();

    void* task1 = __hetero_task_begin(
        /* Input Buffers: */ 3, encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, scores_ptr, scores_size, 
        /* Output Buffers: */ 1, scores_ptr, scores_size, "inference_calculate_score_task"
    );

    
    // Class HVs are created via 'clustering' on +1, -1 encoded hypervectors. (loop 269).
    __hypervector__<D, hvtype> encoded_hv = *encoded_hv_ptr;
    __hypermatrix__<K, D, hvtype> classes = *classes_ptr;

    __hypervector__<K, hvtype> scores = *scores_ptr; // Precision of these scores might need to be increased.

    #ifdef HAMMING_DIST
    //////printf("before hamming_dist\n");
    //auto v =  __hetero_hdc_get_matrix_row<K, D, hvtype>(classes, K, D, 1);
    // FIXME: This is causing a segmentation fault. 
    *scores_ptr =  __hetero_hdc_hamming_distance<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr);
    //////printf("after hamming_dist\n");
    #else
    ////printf("before cossim\n");
    *scores_ptr = __hetero_hdc_cossim<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr);
    //////printf("after cossim\n");
    #endif

    __hetero_task_end(task1);

    void* task2 = __hetero_task_begin(
        /* Input Buffers: 1*/ 4, scores_ptr, scores_size, label_ptr, label_size, encoded_hv_ptr, encoded_hv_size,
        /* paramters: 1*/       encoded_hv_idx,
        /* Output Buffers: 1*/ 1,  label_ptr, label_size, "inference_find_max_task"
    );  
    {
    __hypervector__<K, hvtype> scores = *scores_ptr;
    int max_idx = 0;
    
    #ifdef HAMMING_DIST
    hvtype max_score = (hvtype) D - (*scores_ptr)[0][0]; // I think this is probably causing issues.
    #else
    hvtype max_score = (hvtype) (*scores_ptr)[0][0];
    #endif
    
    for (int k = 0; k < K; k++) {
        #ifdef HAMMING_DIST
        hvtype score = (hvtype) D - (*scores_ptr)[0][k];
        #else
        hvtype score = (hvtype) (*scores_ptr)[0][k];
        #endif
        ////std::cout << score << " ";
        if (score > max_score) {
            max_score = score;
            max_idx = k;
        }
    } 
    
    // Set the label to our guess 
    *label_ptr = max_idx; 
    }
    __hetero_task_end(task2);

    __hetero_section_end(section);
    return;
}

// ========= This is just inlined on teh host for now. =======
#if 0
// train > 0
// In the streaming implementation, this runs for each encoded HV, so N_VEC * EPOCHs times.
// The initialization epoch for training ,needs to set class hypervectors as combination of all input datapoints mapped to that class.
template<int D, int N_CLASSES, int N_VEC>
void classification_node_training_initial( 
    __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size,
    __hypervector__<K, D, hvtype>* classes_ptr, size_t classes_size, // Do we need a separate buffer or can we edit in place. We can edit in place. 
    __hypervector__<D, hvtype>* update_hv_ptr, size_t update_hv_size,  // Used in second stage of clustering node for extracting and accumulating
    int* labels, size_t labels_size, /* Output also*/
    int encoded_hv_idx ) {
    // Do once for all hypervectors combined:
    // Initialize class hypervectors (to 0).

    // TODO: Find out how to cache encoded hypervectors.

    // Class HVs are created via 'clustering' on +1, -1 encoded hypervectors. (loop 269).

    // In first epoch of training, initialize the classes and store the encoded hypervector. 

    // Convert encoded hypervectors (if for some reason they were bipolar ) to binary (1, 0)

    *update_hv_ptr =  __hetero_hdc_get_matrix_row<K, D, hvtype>(*classes_ptr, K, D, label);
    *update_hv_ptr = __hetero_hdc_sum<D, hvtype>(*update_hv_ptr, *encoded_hv_ptr); // May need an instrinsic for this.
    __hetero_hdc_set_matrix_row<K, D, hvtype>(*classes_ptr, *update_hv_ptr, label); // How do we normalize?


    // Binary encoding for these hypervectors. 
    void* section = __hetero_section_begin();
    return;
    
}
#endif


// Retraining epochs
// In the streaming implementation, this runs for each encoded HV, so N_VEC * EPOCHs times.
// classification_node is the hetero-c++ version of searchUnit from the original FPGA code.
template<int D, int K, int N_VEC>
void classification_node_training_rest(/* Input Buffers: 2 */
    __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size,
    __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, // Do we need a separate buffer or can we edit in place. We can edit in place. 
    __hypervector__<K, hvtype>* scores_ptr, size_t scores_size, // Used as Local var.
    __hypervector__<D, hvtype>* update_hv_ptr, size_t update_hv_size,  // Used in second stage of clustering node for extracting and accumulating
    int* argmax, size_t argmax_size,
    int label ) {
    

    // TODO: Find out how to cache encoded hypervectors.

    void* section = __hetero_section_begin();

    void* task1 = __hetero_task_begin(
        /* Input Buffers: */ 3, encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, scores_ptr, scores_size, 
        /* Output Buffers: */ 1, scores_ptr, scores_size, "training_rest_scoring_task"
    );
    // Class HVs are created via 'clustering' on +1, -1 encoded hypervectors. (loop 269).
    //std::cout << "training task 1" << std::endl;

    __hypervector__<D, hvtype> encoded_hv = *encoded_hv_ptr;
    __hypermatrix__<K, D, hvtype> classes = *classes_ptr;

    __hypervector__<K, hvtype> scores = *scores_ptr; // Precision of these scores might need to be increased.

    #ifdef HAMMING_DIST
    *scores_ptr =  __hetero_hdc_hamming_distance<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr);
    #else
    *scores_ptr = __hetero_hdc_cossim<K, D, hvtype>(encoded_hv, classes);
    #endif

    //std::cout << "after dist metric" << std::endl;


    __hetero_task_end(task1);

    void* task2 = __hetero_task_begin(
        /* Input Buffers: 1*/ 3, scores_ptr, scores_size,  classes_ptr, classes_size, argmax, argmax_size,
        /* paramters: 1*/      
        /* Output Buffers: 1*/ 2,  classes_ptr, classes_size, argmax, argmax_size, "training_rest_find_score_task"
    );  
    {
        __hypervector__<K, hvtype> scores = *scores_ptr;

        *argmax = 0;
        
        #ifdef HAMMING_DIST
        hvtype max_score = (hvtype) D - (*scores_ptr)[0][0]; // I think this is probably causing issues.
        #else
        hvtype max_score = (hvtype) (*scores_ptr)[0][0];
        #endif
        
        //std::cout << "good scores access" << std::endl;
        for (int k = 0; k < K; k++) {
            #ifdef HAMMING_DIST
            hvtype score = (hvtype) D - (*scores_ptr)[0][k];
            #else
            hvtype score = (hvtype) (*scores_ptr)[0][k];
            #endif
            ////std::cout << score << " ";
            if (score > max_score) {
                max_score = score;
                *argmax = k;
            }
        } 

        //printf("after argmax\n");

        //std::cout << "good argmax"<< std::endl;

        // FIXME:
        int correct = 0; // Count the number of correct prediction in training epochs.
        
        // Get correct label

        //std::cout << *argmax << std::endl;

        // Calculate max label / index
    }
    __hetero_task_end(task2);

    

    void* task3 = __hetero_task_begin(
        /* Input Buffers: 1*/ 5, encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, update_hv_ptr, update_hv_size, argmax, argmax_size,
        /* paramters: 1*/       label,
        /* Output Buffers: 1*/ 1,  classes_ptr, classes_size, "update_classes_task"
    );  

    int max_idx = *argmax;
    //printf("%d ", max_idx);
    // Update the correct and mispredicted class
    if (label != max_idx) {  // incorrect prediction)
        // FIXME: May need to recude to bipolar encoding? 
        // temp_dim = bipolar_encoding(encoded_hv); 

        // classHV[label]  += temp_dim;  Add to actual class.
        *update_hv_ptr =  __hetero_hdc_get_matrix_row<K, D, hvtype>(*classes_ptr, K, D, label);
        *update_hv_ptr = __hetero_hdc_sum<D, hvtype>(*update_hv_ptr, *encoded_hv_ptr); // May need an instrinsic for this.
        __hetero_hdc_set_matrix_row<K, D, hvtype>(*classes_ptr, *update_hv_ptr, label); // How do we normalize?
        //printf("increment good\n");

        // classHV[maxIndex] -= temp_dim;  Subtract from guessed class.
        *update_hv_ptr =  __hetero_hdc_get_matrix_row<K, D, hvtype>(*classes_ptr, K, D, max_idx);
        *update_hv_ptr = __hetero_hdc_sign_flip<D, hvtype>(*update_hv_ptr); // May need an instrinsic for this.
        *update_hv_ptr = __hetero_hdc_sum<D, hvtype>(*update_hv_ptr, *encoded_hv_ptr); // May need an instrinsic for this.
        __hetero_hdc_set_matrix_row<K, D, hvtype>(*classes_ptr, *update_hv_ptr, max_idx); // How do we normalize?
        //printf("decrement good\n");
    }
    else {
        //printf("correct\n");
    }
    /* 
    else { // need to accumulate to global buffer or something, ask for how to do this. 
        correct += 1; // Use to calculate score after each epoch. "cout << "Training epoch " << iter_epoch << " accuracy: " << float(correct)/size << endl;"
    } */
    __hetero_task_end(task3);

    __hetero_section_end(section);

    //printf("can i put stuff ehre??\n");

    return;
}


// Dimensionality, Clusters, data point vectors, features per.
template <int D, int K, int N_VEC, int N_FEATURES>
void training_root_node( /* Input buffers: 3*/ 
                __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
                __hypervector__<N_FEATURES, hvtype>* datapoint_vec_ptr, size_t datapoint_vec_size, // Features
                 __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, // __hypermatrix__<K, D, binary> // Also an output.
                /* Local Vars: 3 */
                __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, // // __hypervector__<D, binary>
                __hypervector__<K, hvtype>* scores_ptr, size_t scores_size,
                __hypervector__<D, hvtype>* update_hv_ptr, size_t update_hv_size,  // Used in second stage of clustering node for extracting and accumulating
                int* argmax_ptr, size_t argmax_size,
                /* Parameters: 1*/
                int label
                /* Output Buffers: 1 (Classes)*/ 
                ){

    void* root_section = __hetero_section_begin();

    
    // Re-encode each iteration.
    void* encoding_task = __hetero_task_begin(
        /* Input Buffers: 3 */ 3, rp_matrix_ptr, rp_matrix_size, datapoint_vec_ptr, datapoint_vec_size, 
                                 encoded_hv_ptr, encoded_hv_size,
        /* Output Buffers: 1 */ 1, encoded_hv_ptr, encoded_hv_size,
        "training_encoding_task"  
    );

    rp_encoding_node<D, N_FEATURES>(rp_matrix_ptr, rp_matrix_size, datapoint_vec_ptr, datapoint_vec_size, encoded_hv_ptr, encoded_hv_size);

    __hetero_task_end(encoding_task);

    void* training_task = __hetero_task_begin(
        /* Input Buffers: 5 */  5 + 1, 
                                encoded_hv_ptr, encoded_hv_size, 
                                classes_ptr, classes_size, 
                                scores_ptr, scores_size,
                                update_hv_ptr, update_hv_size,
                                argmax_ptr, argmax_size,
                                label,
        /* Output Buffers: 2 */ 1, classes_ptr, classes_size, 
        "training_task"  
    );

    classification_node_training_rest<D, K, N_VEC>(encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, scores_ptr, scores_size, update_hv_ptr, update_hv_size, argmax_ptr, argmax_size, label); 

    __hetero_task_end(training_task);

    __hetero_section_end(root_section);
    return;
}

// Dimensionality, Clusters, data point vectors, features per.
template <int D, int K, int N_VEC, int N_FEATURES>
void inference_root_node( /* Input buffers: 3*/ 
                __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
                __hypervector__<N_FEATURES, hvtype>* datapoint_vec_ptr, size_t datapoint_vec_size, // Features
                __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, // __hypermatrix__<K, D, binary>
                /* Local Vars: 2*/
                __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, // // __hypervector__<D, binary>
                __hypervector__<K, hvtype>* scores_ptr, size_t scores_size,
                // FIXME, give scores its own type.
                /* Parameters: 1*/
                int encoded_hv_idx,
                /* Output Buffers: 1*/
                int* label_ptr, size_t label_size){

    void* root_section = __hetero_section_begin();

    // Re-encode each iteration.
    void* encoding_task = __hetero_task_begin(
        /* Input Buffers: 3 */ 3, rp_matrix_ptr, rp_matrix_size, datapoint_vec_ptr, datapoint_vec_size, 
                                encoded_hv_ptr, encoded_hv_size,
        /* Output Buffers: 1 */ 1, encoded_hv_ptr, encoded_hv_size,
        "inference_encoding_task"  
    );

    ////printf("inference encoding task\n");
    ////std::cout << "inference encoding task" << std::endl;

    rp_encoding_node_copy_copy<D, N_FEATURES>(rp_matrix_ptr, rp_matrix_size, datapoint_vec_ptr, datapoint_vec_size, encoded_hv_ptr, encoded_hv_size);

    __hetero_task_end(encoding_task);

    void* inference_task = __hetero_task_begin(
        /* Input Buffers: 5 */  4 + 1, 
                                encoded_hv_ptr, encoded_hv_size, 
                                classes_ptr, classes_size, 
                                label_ptr, label_size,
                                scores_ptr, scores_size,
        /* Parameters: 1 */     encoded_hv_idx,
        /* Output Buffers: 1 */ 1, label_ptr, label_size,
        "inference_task"  
    );

    classification_node_inference<D, K, N_VEC>(encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, scores_ptr, scores_size, encoded_hv_idx, label_ptr, label_size); 

    __hetero_task_end(inference_task);

    __hetero_section_end(root_section);
    return;
}
