
#include <stdlib.h>
#include "api_hdnn_reram.hpp"

sim_hdnn_reram *ins_hdnn_reram = nullptr;

//////////////////////////////////////////////////
//////////////// LLVM interface //////////////////
//////////////////////////////////////////////////

extern "C" {
void initialize_encoder(void* cfg_void) {
  //printf("** Initialize Encoder ** \n");

  config* cfg = (config*) cfg_void;
  if (ins_hdnn_reram == nullptr) {
    printf("** Initialize Encoder ** \n");
    ins_hdnn_reram = new sim_hdnn_reram(cfg->num_features, cfg->hypervector_dim,
                                        cfg->num_classes);
  }
}


void initialize_device(void* cfg_void) {
 //printf("** Initialize Device ** \n");
  config* cfg = (config*) cfg_void;
  if (ins_hdnn_reram == nullptr) {
    ins_hdnn_reram = new sim_hdnn_reram(cfg->num_features, cfg->hypervector_dim,
                                        cfg->num_classes);
  }
}

void encode_hypervector(int16_t *dst_pointer, int16_t *input_features,
                        int input_dimension, int encoded_dimension) {

  //printf("** Encode Hypervector ** \n");
  assert(input_dimension == ins_hdnn_reram->dim_feature);
  assert(encoded_dimension == ins_hdnn_reram->dim_hv);

  ins_hdnn_reram->enc_kronecker(dst_pointer, input_features);
}


void execute_encode(void* dst_pointer_void, void *input_features_void,
                        int input_dimension, int encoded_dimension) {

  int16_t* dst_pointer = (int16_t*) dst_pointer_void; 
  int16_t* input_features = (int16_t*) input_features_void; 

  assert(input_dimension == ins_hdnn_reram->dim_feature);
  assert(encoded_dimension == ins_hdnn_reram->dim_hv);

  ins_hdnn_reram->enc_kronecker(dst_pointer, input_features);
}

void hamming_distance(void *result, void *encoded_query
                      ) {
  bool reram_comp = false;
  ins_hdnn_reram->hamming_distance((int16_t*) result,(int16_t*) encoded_query, reram_comp);
}

void allocate_base_mem(void *BasePtr, size_t NumBytes) {
  // Do nothing: use internal base matrix for encoding
}

void allocate_feature_mem(void *FeatureMem, size_t NumBytes) {
  ins_hdnn_reram->allocate_feature_mem((int16_t *)FeatureMem, NumBytes);
}

void allocate_class_mem(void *ClassMem, size_t NumBytes) {
  ins_hdnn_reram->allocate_class_mem((int16_t *)ClassMem, NumBytes);
}

void read_class_mem(void *ClassMem, size_t NumBytes) {
  ins_hdnn_reram->read_class_mem((int16_t *)ClassMem, NumBytes);
}

void read_score_mem(void *ScoreMem, size_t NumBytes) {
  ins_hdnn_reram->read_score_mem((int16_t *)ScoreMem, NumBytes);
}

void execute_train(int label) {
  printf("** Execute retrain \n");

  uint32_t dim_hv = ins_hdnn_reram->dim_hv;
  uint32_t dim_feature = ins_hdnn_reram->dim_feature;
  uint32_t num_class = ins_hdnn_reram->num_class;

  // Compute aggregate class HVs
  int16_t *ptr_dst = new int16_t[dim_hv];
  encode_hypervector(ptr_dst, ins_hdnn_reram->feature_mem, dim_feature, dim_hv);

  for (int j = 0; j < dim_hv; j++) {
    ins_hdnn_reram->class_mem[label * dim_hv + j] += ptr_dst[j];
  }

  // Program class HVs into reram array
  for (int i = 0; i < dim_hv; i++) {
    ins_hdnn_reram->program_reram_bit(
        ins_hdnn_reram->class_mem[label * dim_hv + i] >= 0, label, i);
  }

  delete[] ptr_dst;
}

int execute_inference() {
  uint32_t dim_hv = ins_hdnn_reram->dim_hv;
  uint32_t dim_feature = ins_hdnn_reram->dim_feature;
  uint32_t num_class = ins_hdnn_reram->num_class;

  uint32_t correct = 0;
  // Compute encoded query
  int16_t *ptr_dst = new int16_t[dim_hv];
  encode_hypervector(ptr_dst, ins_hdnn_reram->feature_mem, dim_feature, dim_hv);

  // Hamming distance
  hamming_distance(ins_hdnn_reram->score_mem, ptr_dst);
  delete[] ptr_dst;

  int16_t *ptr_hamming_score = ins_hdnn_reram->score_mem;
  int16_t *ptr_min_hamming =
      std::min_element(ptr_hamming_score, ptr_hamming_score + num_class);
  int pred = std::distance(ptr_hamming_score, ptr_min_hamming);
  return pred;
}

void execute_retrain(int label) {
  uint32_t dim_hv = ins_hdnn_reram->dim_hv;
  uint32_t dim_feature = ins_hdnn_reram->dim_feature;
  uint32_t num_class = ins_hdnn_reram->num_class;

  // Encode features
  int16_t *ptr_dst = new int16_t[dim_hv];
  encode_hypervector(ptr_dst, ins_hdnn_reram->feature_mem, dim_feature, dim_hv);

  // Hamming distance
  hamming_distance(ins_hdnn_reram->score_mem, ptr_dst);

  int pred =
      std::distance(ins_hdnn_reram->score_mem,
                    std::min_element(ins_hdnn_reram->score_mem,
                                     ins_hdnn_reram->score_mem + num_class));

  if (pred != label) {
    for (int j = 0; j < dim_hv; j++) {
      ins_hdnn_reram->class_mem[label * dim_hv + j] += ptr_dst[j];
      ins_hdnn_reram->class_mem[pred * dim_hv + j] -= ptr_dst[j];
    }
  }

  // Program class HVs into reram array
  for (int i = 0; i < dim_hv; i++) {
    ins_hdnn_reram->program_reram_bit(
        ins_hdnn_reram->class_mem[label * dim_hv + i] >= 0, label, i);
  }

  delete[] ptr_dst;
}
}
