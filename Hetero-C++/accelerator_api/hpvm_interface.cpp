
#include "api_hdnn_reram.hpp"
#include <stdlib.h>
sim_hdnn_reram *ins_hdnn_reram = nullptr;

//////////////////////////////////////////////////
//////////////// LLVM interface //////////////////
//////////////////////////////////////////////////
extern "C" {
void initialize_encoder(void* cfg_void) {
  config cfg = *(config*) cfg_void;
  if (ins_hdnn_reram == nullptr) {
    ins_hdnn_reram = new sim_hdnn_reram(cfg.num_features, cfg.hypervector_dim,
                                        cfg.num_classes,cfg.mlc_level,true);
  }
}

void initialize_device(void* cfg_void) {
  config cfg = *(config*) cfg_void;
  // Do nothing.
    if (ins_hdnn_reram == nullptr) {
    ins_hdnn_reram = new sim_hdnn_reram(cfg.num_features, cfg.hypervector_dim,
                                        cfg.num_classes,cfg.mlc_level, true);
  }
  
#ifdef TRACE
  printf("initialize_device\n");
#endif
}

void encode_hypervector(int16_t *dst_pointer, int16_t *input_features,
                        int input_dimension, int encoded_dimension) {

  assert(input_dimension == ins_hdnn_reram->dim_feature);
  assert(encoded_dimension == ins_hdnn_reram->dim_hv);

  ins_hdnn_reram->enc_kronecker(dst_pointer, input_features);
}

void hamming_distance(int16_t *result, int16_t *encoded_query,
                      bool reram_comp = false) {
  ins_hdnn_reram->hamming_distance_pcm(result, encoded_query, reram_comp);
}

void dimension_packing(int16_t* input_hv, int16_t* packed_hv)
{
  ins_hdnn_reram->mlc_packing(input_hv,packed_hv);
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

void write_encoded_vector(void *encoded_ref, int64_t label_index, int32_t write_verify_cycles)
{
  ins_hdnn_reram->write_class_vector((int16_t*) encoded_ref,label_index,write_verify_cycles);
}

void execute_train(int label,int write_verify_cycle) {

  uint32_t dim_hv = ins_hdnn_reram->dim_hv;
  uint32_t dim_feature = ins_hdnn_reram->dim_feature;
  uint32_t num_class = ins_hdnn_reram->num_class;
  uint32_t dim_hv_packed=dim_hv/ins_hdnn_reram->pcm_mlc_level;
  // Compute aggregate class HVs
  int16_t *ptr_dst = new int16_t[dim_hv];
  int16_t *ptr_dst_packet=new int16_t[dim_hv_packed];
  encode_hypervector(ptr_dst, ins_hdnn_reram->feature_mem, dim_feature, dim_hv);

//dimension packing
  if (ins_hdnn_reram->pcm_mlc_level>1)
  {
    dimension_packing(ptr_dst,ptr_dst_packet);
  }
  else
  {
    std::memcpy(ptr_dst_packet,ptr_dst,dim_hv*sizeof(int16_t));
  }

  for (int j = 0; j < dim_hv_packed; j++) {
    ins_hdnn_reram->class_mem[label * dim_hv_packed + j] += ptr_dst_packet[j];
  }



  // Program class HVs into reram array
  for (int i = 0; i < dim_hv_packed; i++) {
    ins_hdnn_reram->program_reram_bit_mlc(
        ins_hdnn_reram->class_mem[label * dim_hv_packed + i], label, i,write_verify_cycle);
  }

  delete[] ptr_dst;
  delete[] ptr_dst_packet;
}

int execute_inference() {
  uint32_t dim_hv = ins_hdnn_reram->dim_hv;
  uint32_t dim_feature = ins_hdnn_reram->dim_feature;
  uint32_t num_class = ins_hdnn_reram->num_class;
  uint32_t dim_hv_packed=dim_hv/ins_hdnn_reram->pcm_mlc_level;

  uint32_t correct = 0;
  // Compute encoded query
  int16_t *ptr_dst = new int16_t[dim_hv];
  int16_t *packed_dst=new int16_t[dim_hv_packed];
  encode_hypervector(ptr_dst, ins_hdnn_reram->feature_mem, dim_feature, dim_hv);

  if (ins_hdnn_reram->pcm_mlc_level>1)
  {
    dimension_packing(ptr_dst,packed_dst);
  }
  else
  {
    std::memcpy(packed_dst,ptr_dst,dim_hv*sizeof(int16_t));
  }
  

  // Hamming distance
  hamming_distance(ins_hdnn_reram->score_mem, packed_dst, false);
  delete[] ptr_dst;
  delete[] packed_dst;


  int16_t *ptr_hamming_score = ins_hdnn_reram->score_mem;
  int16_t *ptr_min_hamming =
      std::min_element(ptr_hamming_score, ptr_hamming_score + num_class);
  int pred = std::distance(ptr_hamming_score, ptr_min_hamming);
  return pred;
}

void execute_retrain(int label,int write_verify_cycle) {
  uint32_t dim_hv = ins_hdnn_reram->dim_hv;
  uint32_t dim_feature = ins_hdnn_reram->dim_feature;
  uint32_t num_class = ins_hdnn_reram->num_class;
  uint32_t dim_hv_packed=dim_hv/ins_hdnn_reram->pcm_mlc_level;

  // Encode features
  int16_t *ptr_dst = new int16_t[dim_hv];
  int16_t *packed_dst = new int16_t[dim_hv_packed];

  encode_hypervector(ptr_dst, ins_hdnn_reram->feature_mem, dim_feature, dim_hv);

  if (ins_hdnn_reram->pcm_mlc_level>1)
  {
    dimension_packing(ptr_dst,packed_dst);
  }
  else
  {
    std::memcpy(packed_dst,ptr_dst,dim_hv*sizeof(int16_t));
  }

  // Hamming distance
  hamming_distance(ins_hdnn_reram->score_mem, packed_dst, false);

  int pred =
      std::distance(ins_hdnn_reram->score_mem,
                    std::min_element(ins_hdnn_reram->score_mem,
                                     ins_hdnn_reram->score_mem + num_class));

  if (pred != label) {
    for (int j = 0; j < dim_hv_packed; j++) {
      ins_hdnn_reram->class_mem[label * dim_hv_packed + j] += packed_dst[j];
      ins_hdnn_reram->class_mem[pred * dim_hv_packed + j] -= packed_dst[j];
    }
  }

  // Program class HVs into reram array
  for (int i = 0; i < dim_hv_packed; i++) {
    ins_hdnn_reram->program_reram_bit_mlc(
        ins_hdnn_reram->class_mem[label * dim_hv_packed + i], label, i,write_verify_cycle);
  }

  delete[] ptr_dst;
  delete[] packed_dst;

}
}