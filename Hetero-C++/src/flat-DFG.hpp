#pragma once

#include <hpvm_hdc.h>
#include <heterocc.h>
#include <iostream>

//#define HAMMING_DIST

#undef D
#undef N_FEATURES
#undef K

typedef int binary;
#ifndef HAMMING_DIST
typedef float hvtype;
#else
typedef int hvtype;
#endif


#ifndef DEVICE
#define DEVICE 1
#endif

// Dimensionality, Clusters, data point vectors, features per.
template <int D, int K, int N_VEC, int N_FEATURES>
void flat_inference_root_node( /* Input buffers: 3*/ 
                __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
                __hypervector__<N_FEATURES, hvtype>* datapoint_vec_ptr, size_t datapoint_vec_size, // Features
                __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, // __hypermatrix__<K, D, binary>
                /* Local Vars: 2*/
                __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, // // __hypervector__<D, binary>
                __hypervector__<K, hvtype>* scores_ptr, size_t scores_size,
                __hypervector__<K, hvtype>* norms_ptr, size_t norms_size,
                /* Parameters: 1*/
                int encoded_hv_idx,
                /* Output Buffers: 1*/
                int* label_ptr, size_t label_size){

    void* root_section = __hetero_section_begin();

    void* root_task = __hetero_task_begin(
        /* Input Buffers: 3 */ 8,
        rp_matrix_ptr, rp_matrix_size, 
        datapoint_vec_ptr, datapoint_vec_size, 
        encoded_hv_ptr, encoded_hv_size,
        classes_ptr, classes_size, 
        label_ptr, label_size,
        scores_ptr, scores_size,
        norms_ptr, norms_size,
        encoded_hv_idx,
        /* Output Buffers: 1 */ 1,
        encoded_hv_ptr, encoded_hv_size,
        "root_inference_task"
    );

    void* fpga_section = __hetero_section_begin();

    // ======= Encoding and Inference ============== 


    // ======= Encoding  ============== 
    // Re-encode each iteration.
    void* rp_encoding_task = __hetero_task_begin(
        /* Input Buffers: 2*/ 3, rp_matrix_ptr, rp_matrix_size, datapoint_vec_ptr, datapoint_vec_size, encoded_hv_ptr, encoded_hv_size,
        /* Parameters: 0*/
        /* Output Buffers: 1*/ 1, encoded_hv_ptr, encoded_hv_size,
        "inner_rp_encoding_task"
    );
    {
    __hypervector__<D, hvtype> encoded_hv = __hetero_hdc_create_hypervector<D, hvtype>(0, (void*) zero_hv<hvtype>);
    *encoded_hv_ptr = encoded_hv;

    encoded_hv = __hetero_hdc_matmul<D, N_FEATURES, hvtype>(*datapoint_vec_ptr, *rp_matrix_ptr); 
    *encoded_hv_ptr = encoded_hv;
    }
    __hetero_task_end(rp_encoding_task); 

    // ======= Encoding End ============== 

    // ========= SCORING ===========
    void* task1 = __hetero_task_begin(
        /* Input Buffers: */ 4, encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, scores_ptr, scores_size, norms_ptr, norms_size, 
        /* Output Buffers: */ 1, scores_ptr, scores_size, "inference_calculate_score_task"
    );

    {
    // Class HVs are created via 'clustering' on +1, -1 encoded hypervectors. (loop 269).
    __hypervector__<D, hvtype> encoded_hv = *encoded_hv_ptr;
    __hypermatrix__<K, D, hvtype> classes = *classes_ptr;

    __hypervector__<K, hvtype> scores = *scores_ptr; // Precision of these scores might need to be increased.

    #ifdef HAMMING_DIST
    *scores_ptr =  __hetero_hdc_hamming_distance<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr);
    #else
    *norms_ptr = __hetero_hdc_l2norm<K, D, hvtype>(*classes_ptr);
    *scores_ptr = __hetero_hdc_matmul<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr); 
    *scores_ptr = __hetero_hdc_div<K, hvtype>(*scores_ptr, *norms_ptr);
    *scores_ptr = __hetero_hdc_absolute_value<K, hvtype>(*scores_ptr);
    #endif


    }
    __hetero_task_end(task1);

    // ========= SCORING END ===========


    // ========= ARGMAX / UPDATE ===========

    void* task2 = __hetero_task_begin(
        /* Input Buffers: 1*/ 4, scores_ptr, scores_size, label_ptr, label_size, encoded_hv_ptr, encoded_hv_size,
        /* paramters: 1*/       encoded_hv_idx,
        /* Output Buffers: 1*/ 1,  label_ptr, label_size, "inference_find_max_task"
    );  
    {
    __hypervector__<K, hvtype> scores = *scores_ptr;
    int max_idx = 0;
    
    hvtype* scores_elem_ptr = (hvtype*) scores_ptr;
    #ifdef HAMMING_DIST
    hvtype max_score = (hvtype) D - scores_elem_ptr[0]; // I think this is probably causing issues.
    #else
    hvtype max_score = (hvtype) scores_elem_ptr[0];
    #endif

    
    for (int k = 0; k < K; k++) {
        #ifdef HAMMING_DIST
        hvtype score = (hvtype) D - scores_elem_ptr[k];
        #else
        hvtype score = (hvtype) scores_elem_ptr[k];
        #endif
        if (score > max_score) {
            max_score = score;
            max_idx = k;
        }
    } 
    
    // Set the label to our guess 
    *label_ptr = max_idx; 
    }
    __hetero_task_end(task2);

    // ========= ARGMAX / UPDATE END ===========

    __hetero_section_end(fpga_section);

    // ======== INFERENCE END =============

    __hetero_task_end(root_task);

    __hetero_section_end(root_section);
}

// Dimensionality, Clusters, data point vectors, features per.
template <int D, int K, int N_VEC, int N_FEATURES>
void flat_training_root_node( /* Input buffers: 3*/ 
                __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
                __hypervector__<N_FEATURES, hvtype>* datapoint_vec_ptr, size_t datapoint_vec_size, // Features
                 __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, // __hypermatrix__<K, D, binary> // Also an output.
                /* Local Vars: 3 */
                __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, // // __hypervector__<D, binary>
                __hypervector__<K, hvtype>* scores_ptr, size_t scores_size,
                __hypervector__<K, hvtype>* norms_ptr, size_t norms_size,
                __hypervector__<D, hvtype>* update_hv_ptr, size_t update_hv_size,  // Used in second stage of clustering node for extracting and accumulating
                int* argmax_ptr, size_t argmax_size,
                /* Parameters: 1*/
                int label
                /* Output Buffers: 1 (Classes)*/ 
                ){
    void* root_section = __hetero_section_begin();

    // Re-encode each iteration.
    void* root_task = __hetero_task_begin(
        /* Input Buffers: 3 */ 9, 
            rp_matrix_ptr, rp_matrix_size, 
            datapoint_vec_ptr, datapoint_vec_size, 
            encoded_hv_ptr, encoded_hv_size,  
            classes_ptr, classes_size, 
            scores_ptr, scores_size,
            norms_ptr, norms_size,
            update_hv_ptr, update_hv_size,
            argmax_ptr, argmax_size,
            label,
        /* Output Buffers: 1 */ 1, 
        encoded_hv_ptr, encoded_hv_size,
        "training_encoding_task_wrapper"  
    );

    void* fpga_section = __hetero_section_begin();

    // Re-encode each iteration.
    void* rp_encoding_task = __hetero_task_begin(
        /* Input Buffers: 3 */ 3, rp_matrix_ptr, rp_matrix_size, datapoint_vec_ptr, datapoint_vec_size, 
                                 encoded_hv_ptr, encoded_hv_size,
        /* Output Buffers: 1 */ 1, encoded_hv_ptr, encoded_hv_size,
        "training_encoding_task"  
    );

    __hetero_hint(DEVICE);

    {
    // ======= Encoding  ============== 
    // Re-encode each iteration.
    
    __hypervector__<D, hvtype> encoded_hv = __hetero_hdc_create_hypervector<D, hvtype>(0, (void*) zero_hv<hvtype>);
    *encoded_hv_ptr = encoded_hv;

    encoded_hv = __hetero_hdc_matmul<D, N_FEATURES, hvtype>(*datapoint_vec_ptr, *rp_matrix_ptr); 
    *encoded_hv_ptr = encoded_hv;
    }
    __hetero_task_end(rp_encoding_task); 

    // ======= Encoding End ============== 

void* task1 = __hetero_task_begin(
        /* Input Buffers: */ 4, encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, scores_ptr, scores_size, norms_ptr, norms_size, 
        /* Output Buffers: */ 1, scores_ptr, scores_size, "training_rest_scoring_task"
    );
    {
        __hetero_hint(DEVICE);
    __hypervector__<D, hvtype> encoded_hv = *encoded_hv_ptr;
    __hypermatrix__<K, D, hvtype> classes = *classes_ptr;

    __hypervector__<K, hvtype> scores = *scores_ptr; // Precision of these scores might need to be increased.

    #ifdef HAMMING_DIST
    *scores_ptr =  __hetero_hdc_hamming_distance<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr);
    #else
#if 0
    *scores_ptr = __hetero_hdc_cossim<K, D, hvtype>(encoded_hv, classes);
#endif
    *norms_ptr = __hetero_hdc_l2norm<K, D, hvtype>(*classes_ptr);
    *scores_ptr = __hetero_hdc_matmul<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr); 
    *scores_ptr = __hetero_hdc_div<K, hvtype>(*scores_ptr, *norms_ptr);
    *scores_ptr = __hetero_hdc_absolute_value<K, hvtype>(*scores_ptr);
    #endif


    }
    __hetero_task_end(task1);

    void* task2 = __hetero_task_begin(
        /* Input Buffers: 1*/ 3, scores_ptr, scores_size, classes_ptr, classes_size, argmax_ptr, argmax_size,
        /* paramters: 1*/      
        /* Output Buffers: 1*/ 2,  classes_ptr, classes_size, argmax_ptr, argmax_size, "training_rest_find_score_task"
    );  
    {
        __hetero_hint(DEVICE);

        __hypervector__<K, hvtype> scores = *scores_ptr;

        *argmax_ptr = 0;
        
        hvtype* scores_elem_ptr = (hvtype*) scores_ptr;
        #ifdef HAMMING_DIST
        hvtype max_score = (hvtype) D - scores_elem_ptr[0]; // I think this is probably causing issues.
        #else
        hvtype max_score = (hvtype) scores_elem_ptr[0];
        #endif
        
        for (int k = 0; k < K; k++) {
            #ifdef HAMMING_DIST
            hvtype score = (hvtype) D - scores_elem_ptr[k];
            #else
            hvtype score = (hvtype) scores_elem_ptr[k];
            #endif
            if (score > max_score) {
                max_score = score;
                *argmax_ptr = k;
            }
        } 
    }
    __hetero_task_end(task2);

    

    void* task3 = __hetero_task_begin(
        /* Input Buffers: 1*/ 5, encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, update_hv_ptr, update_hv_size, argmax_ptr, argmax_size,
        /* paramters: 1*/       label,
        /* Output Buffers: 1*/ 1,  classes_ptr, classes_size, "update_classes_task"
    );  
    {
        __hetero_hint(DEVICE);
    int max_idx = *argmax_ptr;
    // Update the correct and mispredicted class
    if (label != max_idx) {  // incorrect prediction)
        
        *update_hv_ptr =  __hetero_hdc_get_matrix_row<K, D, hvtype>(*classes_ptr, K, D, label);
        *update_hv_ptr = __hetero_hdc_sum<D, hvtype>(*update_hv_ptr, *encoded_hv_ptr); // May need an instrinsic for this.
        __hetero_hdc_set_matrix_row<K, D, hvtype>(*classes_ptr, *update_hv_ptr, label); // How do we normalize?

        *update_hv_ptr =  __hetero_hdc_get_matrix_row<K, D, hvtype>(*classes_ptr, K, D, max_idx);
        *update_hv_ptr = __hetero_hdc_sub<D, hvtype>(*update_hv_ptr, *encoded_hv_ptr); // May need an instrinsic for this.
        __hetero_hdc_set_matrix_row<K, D, hvtype>(*classes_ptr, *update_hv_ptr, max_idx); // How do we normalize?
    }
    }
    __hetero_task_end(task3);

    __hetero_section_end(fpga_section);

     __hetero_task_end(root_task);

    __hetero_section_end(root_section);
}

template<int D, int N_FEATURES>
void FlatInitialEncodingDFG(
        /* Input Buffers: 2*/
        __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
        __hypervector__<N_FEATURES, hvtype>* input_datapoint_ptr, size_t input_datapoint_size, // __hypervector__<N_FEATURES, int> 
        /* Output Buffers: 1*/
        __hypervector__<D, hvtype>* output_hv_ptr, size_t output_hv_size) { // __hypervector__<D, binary>
    
    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(
        /* Input Buffers: 2*/ 3, rp_matrix_ptr, rp_matrix_size, input_datapoint_ptr, input_datapoint_size, output_hv_ptr, output_hv_size,
        /* Parameters: 0*/
        /* Output Buffers: 1*/ 1, output_hv_ptr, output_hv_size,
        "initial_encoding_wrapper"
    );
    
    __hypervector__<D, hvtype> encoded_hv = __hetero_hdc_create_hypervector<D, hvtype>(0, (void*) zero_hv<hvtype>);
    *output_hv_ptr = encoded_hv;

    encoded_hv = __hetero_hdc_matmul<D, N_FEATURES, hvtype>(*input_datapoint_ptr, *rp_matrix_ptr); 
    *output_hv_ptr = encoded_hv;

    __hetero_task_end(task); 

    __hetero_section_end(section);
    return;
}
