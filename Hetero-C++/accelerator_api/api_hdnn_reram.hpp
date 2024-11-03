// This file is compiled as C++ and linked with stdlib
// into the simulation executable.
//#define DEBUG 1

#include <cstring>
#include <random>
#include <algorithm>
#include <cassert>

struct config {
  int num_features;
  int hypervector_dim;
  int num_classes;
  int mlc_level;// >0 and <4, make sure in range,use mlc_level=1 for default case,ie SLC mode
  // int width_classes;
  // int retrain_label;
};



class sim_hdnn_reram {
private:
  float *reram_array = nullptr;
  float *pcm_array=nullptr;

  //pcm related 

  uint16_t BER_sb2te [10]={0,35,22,18,13,1,8,5};//type sb2te3/gst467
  uint16_t BER_tite2 [4][8]={{0, 0, 0, 0, 0, 0, 0, 0},
                             {0,10, 6, 3, 3, 2, 1, 0},
                             {0,20,11, 7, 6, 5, 4, 3},
                             {0,30,16,10, 9, 7, 6, 5}};//type tite2/gst467, write_very=0 no erro, reference mode
  //assume linear scailing of bit error rate from MLC1-MLC3
  //0,30,16,10,9,7,6,5 MLC3
  //0,20,11, 7,6,5,4,3 MLC2
  //0,10,6,  3,3,2,1,0 MLC1 SLC

  float mlc_resis_LUT_step[4]={0.0,1.0,0.5,0.333};//support SLC,MLC2,MLC3, MLC=1 reference mode,do not use MLC=0

  // Manual seed
  std::mt19937 rnd_gen;
  std::lognormal_distribution<float> dist_hrs;
  std::lognormal_distribution<float> dist_lrs;

  float *enc_mat_1; // Encoding weight matrix with shape (f1, d1)
  float *enc_mat_2; // Encoding weight matrix with shape (f2, d2)
  uint32_t f1, d1, f2, d2;//?? encoding matrix related

  void dim_prime_fact();

  void init_buffer();
  void init_reram_array();
  int16_t adc_sensing_mlc(float inp);
  inline int adc_sensing(float inp);

public:
  uint32_t dim_feature;
  uint32_t dim_hv;
  uint32_t num_class;

  uint32_t pcm_mlc_capability=4;
  uint32_t pcm_mlc_level;

  bool reference_mode;

  int16_t *feature_mem = nullptr;
  int16_t *class_mem = nullptr;
  int16_t *score_mem = nullptr;

  sim_hdnn_reram(uint32_t dim_f, uint32_t dim_d, uint32_t num_c,uint32_t mlc_level,bool reference_m);
  ~sim_hdnn_reram();

  void allocate_base_mem(int16_t *BasePtr, size_t NumBytes);
  void allocate_feature_mem(int16_t *FeatureMem, size_t NumBytes);
  void allocate_class_mem(int16_t *ClassMem, size_t NumBytes);

  void read_class_mem(int16_t *ClassMem, size_t NumBytes);
  void read_score_mem(int16_t *ScoreMem, size_t NumBytes);

  void init_enc_mat();
  void encode_hypervector();
  void enc_kronecker(int16_t *dst_ptr, int16_t *inp_feature);

  void hamming_distance_slc(int16_t *result, int16_t *encoded_query,
                        bool reram_comp);
  void hamming_distance_pcm(int16_t *result, int16_t *encoded_query,
                        bool reram_comp);
  inline int readout_reram_bit(uint32_t i, uint32_t j);
  inline void program_reram_bit(bool inp_bit, uint32_t i, uint32_t j,int write_verify_cycles);
  

  void program_reram_bit_mlc(int16_t inp_bit, uint32_t i, uint32_t j,int write_verify_cycles);
  int16_t readout_reram_bit_mlc(uint32_t i, uint32_t j);
  void write_class_vector(int16_t *encoded_ref,int16_t label_index,int write_verify_cycles);

  void mlc_packing(int16_t*input_hv, int16_t* packed_hv);
  
};

sim_hdnn_reram::sim_hdnn_reram(uint32_t dim_f, uint32_t dim_d, uint32_t num_c,uint32_t mlc_level,bool reference_m) {
  // Initialize HD params
  dim_feature = dim_f;
  dim_hv = dim_d;
  num_class = num_c;
  pcm_mlc_level=mlc_level;
  reference_mode=reference_m;
  // Initialize Kronecker encoder
  srand(0);
  dim_prime_fact();
  init_enc_mat();
  // Initialize reram array and buffers
  init_buffer();
  init_reram_array();

  printf("Initialized SpecPCM simulator with %d features, %d HD dims, and %d "
         "classes.\n",
         dim_feature, dim_hv, num_class);

#ifdef DEBUG
  printf("f1=%d\tf2=%d\td1=%d\td2=%d\n", f1, f2, d1, d2);
#endif
}

sim_hdnn_reram::~sim_hdnn_reram() {

  if (enc_mat_1)
    delete[] enc_mat_1;

  if (enc_mat_2)
    delete[] enc_mat_2;

  if (reram_array)
    delete[] reram_array;

  if (feature_mem)
    delete[] feature_mem;

  if (class_mem)
    delete[] class_mem;

  if (score_mem)
    delete[] score_mem;
}

void sim_hdnn_reram::allocate_base_mem(int16_t *BasePtr, size_t NumBytes) {
  // Do nothing: use internal base matrix for encoding
}

void sim_hdnn_reram::allocate_feature_mem(int16_t *FeatureMem,
                                          size_t NumBytes) {
  std::memcpy(feature_mem, FeatureMem, NumBytes);
}

void sim_hdnn_reram::allocate_class_mem(int16_t *ClassMem, size_t NumBytes) {
  std::memcpy(class_mem, ClassMem, NumBytes);
}

void sim_hdnn_reram::read_class_mem(int16_t *ClassMem, size_t NumBytes) {
  std::memcpy(ClassMem, class_mem, NumBytes);
}

void sim_hdnn_reram::read_score_mem(int16_t *ScoreMem, size_t NumBytes) {
  std::memcpy(ScoreMem, this->score_mem, NumBytes);
}

void sim_hdnn_reram::dim_prime_fact() {
  uint32_t n = sqrt(dim_feature);

  // Decompose feature dim.
  while (1) {
    if (dim_feature % n == 0) {
      f1 = n;
      f2 = dim_feature / f1;
      break;
    }
    n--;
  }

  // Decompose HV dim.
  n = sqrt(dim_hv);
  while (1) {
    if (dim_hv % n == 0) {
      d1 = n;
      d2 = dim_hv / d1;
      break;
    }
    n--;
  }
}

void sim_hdnn_reram::init_enc_mat() {
  enc_mat_1 = new float[f1 * d1];
  for (int i = 0; i < f1; i++) {
    for (int j = 0; j < d1; j++) {
      int rnd = rand() % 2;
      enc_mat_1[i * d1 + j] = rnd * 2.0 - 1;
    }
  }

  enc_mat_2 = new float[f2 * d2];
  for (int i = 0; i < f2; i++) {
    for (int j = 0; j < d2; j++) {
      int rnd = rand() % 2;
      enc_mat_2[i * d2 + j] = rnd * 2.0 - 1;
    }
  }
}

void sim_hdnn_reram::init_buffer() {
  feature_mem = new int16_t[dim_feature];
  class_mem = new int16_t[dim_hv * num_class];
  score_mem = new int16_t[num_class];
}

void sim_hdnn_reram::init_reram_array() {
  this->rnd_gen = std::mt19937(0);
  this->dist_lrs = std::lognormal_distribution<float>(1.6, 0.25);
  this->dist_hrs = std::lognormal_distribution<float>(1.6, 0.25);

  this->reram_array = new float[this->dim_hv * this->num_class];

  memset(this->reram_array, 0.0,
         this->dim_hv * this->num_class * sizeof(float));
}

void sim_hdnn_reram::enc_kronecker(int16_t *dst_ptr, int16_t *inp_feature) {
  float *tmp = new float[f2 * d1];

  // Kronecker 1 encoding (f2,f1) x (f1,d1) -> (f2,d1)
  for (int i = 0; i < f2; i++) {
    for (int j = 0; j < d1; j++) {
      tmp[i * d1 + j] = 0.0;
      for (int k = 0; k < f1; k++) {
        tmp[i * d1 + j] += inp_feature[i * f1 + k] * enc_mat_1[k * d1 + j];
        // printf("tmp @ %d, %d, %d = %f x %f = %f\n", i, j, k, inp_feature[i *
        // f1 + k], enc_mat_1[k * d1 + j], tmp[i * d1 + j]);
      }
      // printf("tmp @ %d, %d = %f\n", i, j, tmp[i * d1 + j]);
    }
  }

  // Kronecker 2 encoding: (f2,d1) x (f2,d2) -> (d1,d2)
  for (int i = 0; i < d1; i++) {
    for (int j = 0; j < d2; j++) {
      float t = 0.0;
      for (int k = 0; k < f2; k++) {
        t += tmp[k * d1 + i] * enc_mat_2[k * d2 + j];
      }
      dst_ptr[i * d2 + j] = t >= 0 ? 1 : -1; // Binarized encoded results
    }
  }

  if (tmp)
    delete[] tmp;
}


//default implementation
inline int sim_hdnn_reram::adc_sensing(float inp) {
  // return inp < 2000 ? 1 : 0;
  return inp >= 0 ? 1 : -1;
}

inline int sim_hdnn_reram::readout_reram_bit(uint32_t i, uint32_t j) {
  return this->adc_sensing(this->reram_array[i * dim_hv + j]);
}

inline void sim_hdnn_reram::program_reram_bit(bool inp_bit, uint32_t i,
                                              uint32_t j,int write_verify_cycles) {
  // this->reram_array[i * this->dim_hv + j] =
  // inp_bit ? this->dist_lrs(this->rnd_gen) : this->dist_hrs(this->rnd_gen);
  this->reram_array[i * this->dim_hv + j] = inp_bit ? 1 : -1;
}

                           

void sim_hdnn_reram::hamming_distance_slc(int16_t *result, int16_t *encoded_query,
                                      bool reram_comp = false) {
  if (reram_comp) {
    *result = 0;
  } else {
    for (uint32_t c = 0; c < this->num_class; c++) {
      result[c] = 0;
      for (uint32_t j = 0; j < this->dim_hv; j++) {
        result[c] += this->readout_reram_bit_mlc(c, j) == encoded_query[j] ? 0 : 1;
      }
    }
  }
}


//pcm implementation
void sim_hdnn_reram::mlc_packing(int16_t*input_hv, int16_t* packed_hv)
{
  for (uint32_t i = 0; i < dim_hv/pcm_mlc_level; i++)
  {
    int16_t temp_pack=0;
    for (uint32_t j = 0; j < pcm_mlc_level; j++)
    {
      temp_pack+=input_hv[i*pcm_mlc_level+j];
    }
    packed_hv[i]=temp_pack;
  } 
}


int16_t sim_hdnn_reram:: adc_sensing_mlc(float inp)
{

  if (pcm_mlc_level>1&&pcm_mlc_level<4)
  {
    float step=mlc_resis_LUT_step[pcm_mlc_level];
    return static_cast<int16_t> (inp/step);
  }
  else
  {
    return inp >= 0 ? 1 : -1;
  }

}
 

int16_t sim_hdnn_reram::readout_reram_bit_mlc(uint32_t i, uint32_t j)
{
  int16_t temp= this->adc_sensing_mlc(this->reram_array[i*dim_hv/pcm_mlc_level+j]);
 
 
   
  return temp;
}


void sim_hdnn_reram::hamming_distance_pcm(int16_t *result, int16_t *encoded_query,
                                      bool reram_comp = false) {
  if (reram_comp) {
    *result = 0;
  } else {
    for (uint32_t c = 0; c < this->num_class; c++) {
      result[c] = 0;
      for (uint32_t j = 0; j < this->dim_hv/this->pcm_mlc_level; j++) {
        result[c] += this->readout_reram_bit_mlc(c, j)*encoded_query[j];//dot product 
      }
      result[c]=-result[c];//select min for best,so flip
    }
  }
}


void sim_hdnn_reram::program_reram_bit_mlc(int16_t inp_bit, uint32_t i, uint32_t j,int write_verify_cycles)
{
   // this->reram_array[i * this->dim_hv + j] =
  // inp_bit ? this->dist_lrs(this->rnd_gen) : this->dist_hrs(this->rnd_gen);

  int16_t write_value=inp_bit;
  uint16_t threashold=(BER_tite2[pcm_mlc_level][write_verify_cycles]*pcm_mlc_level);
  bool truefalse= (rand() % 100) <threashold;
  bool drift_dir=(rand() % 100) <50;

  //printf("threashold %d ,true false %d \n",threashold,truefalse);
  if (truefalse)
  {
    if(pcm_mlc_level==1)
    {
      write_value=-write_value;
    }
    else
    {
        if (drift_dir)
      {
        write_value++;
      }
      else
      {
        write_value--;
      } 
    }
  }

  if (write_value>(int16_t)pcm_mlc_level)
  {
    write_value=(int16_t)pcm_mlc_level;
  }

  if (write_value<-((int16_t)pcm_mlc_level))
  {
    write_value=-(int16_t)pcm_mlc_level;
  }
  float step=mlc_resis_LUT_step[pcm_mlc_level];

  this->reram_array[i * this->dim_hv/pcm_mlc_level + j] = (float)write_value*step;
  //printf("write_value value %d\n",write_value);

}



void sim_hdnn_reram::write_class_vector(int16_t *encoded_ref,int16_t label_index,int write_verify_cycles)
{
  for (uint32_t i = 0; i < this->dim_hv/this->pcm_mlc_level; i++)
  {
    //printf("label_index: %d i: %d\n", label_index, i);
    this->program_reram_bit_mlc(encoded_ref[i], label_index,i,write_verify_cycles);
  }
  
}
