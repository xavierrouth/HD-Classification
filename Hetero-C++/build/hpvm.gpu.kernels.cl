

/* Support for floating point constants */
typedef ulong ConstantDoubleTy;
typedef uint ConstantFloatTy;
typedef struct { ulong f1; ushort f2; ushort pad[3]; } ConstantFP80Ty;
typedef struct { ulong f1; ulong f2; } ConstantFP128Ty;


/* OpenCL Pragmas */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable


/* Global Declarations */
/* Helper union for bitcasts */
typedef union {
  uint Int32;
  ulong Int64;
  float Float;
  double Double;
} llvmBitCastUnion;

/* Types Declarations */

/* Types Definitions */
struct l_array_1263616_float {
  float array[1263616];
};
struct l_array_2048_float {
  float array[2048];
};
struct l_array_617_float {
  float array[617];
};
struct l_vector_617_float {
  float vector[617];
} __attribute__((aligned(4096)));
struct l_vector_1263616_float {
  float vector[1263616];
} __attribute__((aligned(8388608)));
struct l_vector_2048_float {
  float vector[2048];
} __attribute__((aligned(8192)));

/* Function definitions */

/* Function Declarations */
__kernel 
void/* Processing Function: _Z21rp_encoding_node_copyILi2048ELi617EEvPu11matrix_typeIXT_EXT0_EfEmPu11matrix_typeILi1EXT0_EfEmPu11matrix_typeILi1EXT_EfEm_loop_header_reorder_c_opencl_c_c_c: 0*/
 _Z21rp_encoding_node_copyILi2048ELi617EEvPu11matrix_typeIXT_EXT0_EfEmPu11matrix_typeILi1EXT0_EfEmPu11matrix_typeILi1EXT_EfEm_loop_header_reorder_c_opencl_c_c_c(__global struct l_array_617_float*, ulong, __global struct l_array_1263616_float*, ulong, __global struct l_array_2048_float*, ulong);
ulong/* Processing Function: get_global_id: 0*/
 get_global_id(uint);
ulong/* Processing Function: get_global_size: 0*/
 get_global_size(uint);
__kernel 
void/* Processing Function: _Z26encoding_and_training_nodeILi2048ELi26ELi6238ELi617EEvPu11matrix_typeIXT_EXT2_EfEmPu11matrix_typeILi1EXT2_EfEmPu11matrix_typeIXT0_EXT_EfEmiPu11matrix_typeILi1EXT_EfEmPu11matrix_typeILi1EXT0_EfEmS9_mS7_mPim_loop_header_reorder_c_opencl_c_c_c: 0*/
 _Z26encoding_and_training_nodeILi2048ELi26ELi6238ELi617EEvPu11matrix_typeIXT_EXT2_EfEmPu11matrix_typeILi1EXT2_EfEmPu11matrix_typeIXT0_EXT_EfEmiPu11matrix_typeILi1EXT_EfEmPu11matrix_typeILi1EXT0_EfEmS9_mS7_mPim_loop_header_reorder_c_opencl_c_c_c(__global struct l_array_617_float*, ulong, __global struct l_array_1263616_float*, ulong, __global struct l_array_2048_float*, ulong);
__kernel 
void/* Processing Function: _Z27encoding_and_inference_nodeILi2048ELi26ELi1559ELi617EEvPu11matrix_typeIXT_EXT2_EfEmPu11matrix_typeILi1EXT2_EfEmPu11matrix_typeIXT0_EXT_EfEmPimPu11matrix_typeILi1EXT_EfEmPu11matrix_typeILi1EXT0_EfEmSA_mi_loop_header_reorder_c_opencl_c_c_c: 0*/
 _Z27encoding_and_inference_nodeILi2048ELi26ELi1559ELi617EEvPu11matrix_typeIXT_EXT2_EfEmPu11matrix_typeILi1EXT2_EfEmPu11matrix_typeIXT0_EXT_EfEmPimPu11matrix_typeILi1EXT_EfEmPu11matrix_typeILi1EXT0_EfEmSA_mi_loop_header_reorder_c_opencl_c_c_c(__global struct l_array_617_float*, ulong, __global struct l_array_1263616_float*, ulong, __global struct l_array_2048_float*, ulong);


/* LLVM Intrinsic Builtin Function Bodies */
static __forceinline int llvm_fcmp_ord(double X, double Y) { return X == X && Y == Y; }
static __forceinline int llvm_fcmp_uno(double X, double Y) { return X != X || Y != Y; }
static __forceinline int llvm_fcmp_ueq(double X, double Y) { return X == Y || llvm_fcmp_uno(X, Y); }
static __forceinline int llvm_fcmp_une(double X, double Y) { return X != Y; }
static __forceinline int llvm_fcmp_ult(double X, double Y) { return X <  Y || llvm_fcmp_uno(X, Y); }
static __forceinline int llvm_fcmp_ugt(double X, double Y) { return X >  Y || llvm_fcmp_uno(X, Y); }
static __forceinline int llvm_fcmp_ule(double X, double Y) { return X <= Y || llvm_fcmp_uno(X, Y); }
static __forceinline int llvm_fcmp_uge(double X, double Y) { return X >= Y || llvm_fcmp_uno(X, Y); }
static __forceinline int llvm_fcmp_oeq(double X, double Y) { return X == Y ; }
static __forceinline int llvm_fcmp_one(double X, double Y) { return X != Y && llvm_fcmp_ord(X, Y); }
static __forceinline int llvm_fcmp_olt(double X, double Y) { return X <  Y ; }
static __forceinline int llvm_fcmp_ogt(double X, double Y) { return X >  Y ; }
static __forceinline int llvm_fcmp_ole(double X, double Y) { return X <= Y ; }
static __forceinline int llvm_fcmp_oge(double X, double Y) { return X >= Y ; }
static __forceinline int llvm_fcmp_0(double X, double Y) { return 0; }
static __forceinline int llvm_fcmp_1(double X, double Y) { return 1; }


/* Function Bodies */
__kernel 
void/* Processing Function: _Z21rp_encoding_node_copyILi2048ELi617EEvPu11matrix_typeIXT_EXT0_EfEmPu11matrix_typeILi1EXT0_EfEmPu11matrix_typeILi1EXT_EfEm_loop_header_reorder_c_opencl_c_c_c: 0*/
 _Z21rp_encoding_node_copyILi2048ELi617EEvPu11matrix_typeIXT_EXT0_EfEmPu11matrix_typeILi1EXT0_EfEmPu11matrix_typeILi1EXT_EfEm_loop_header_reorder_c_opencl_c_c_c(__global struct l_array_617_float* input_datapoint_ptr, ulong input_datapoint_size, __global struct l_array_1263616_float* rp_matrix_ptr, ulong rp_matrix_size, __global struct l_array_2048_float* output_hv_ptr, ulong output_hv_size) {

  ulong tmp__1;
  ulong tmp__2;
  __global struct l_vector_617_float* _clone2_cloned;
  __global struct l_vector_1263616_float* _clone1_cloned;
  __global struct l_vector_2048_float* _clone_cloned;
  ulong loop_iv9_cloned;
  ulong loop_iv9_cloned__PHI_TEMPORARY;
  float load_buffer_cloned;
  float load_buffer12_cloned;
  float load_buffer13_cloned;
  ulong loop_step10_cloned;


/* Printing: newFuncRoot_cloned */


/* Processing Basic Block: newFuncRoot_cloned */
/* newFuncRoot_cloned: */
  tmp__1 = get_global_id((uint) 0);
  tmp__2 = get_global_size((uint) 0);
  _clone2_cloned = ((__global struct l_vector_617_float*)input_datapoint_ptr->array);
  _clone1_cloned = ((__global struct l_vector_1263616_float*)rp_matrix_ptr->array);
  _clone_cloned = ((__global struct l_vector_2048_float*)output_hv_ptr->array);
  *((&(((__global float*)_clone_cloned))[tmp__1])) = (float) 0.000000e+00;

/* Branch:   br label %loop.header7_cloned */
/* Printing PHIs for newFuncRoot_cloned->loop.header7_cloned */
/* Printing phi node:   %loop.iv9_cloned = phi i64 [ 0, %newFuncRoot_cloned ], [ %loop.step10_cloned, %loop.header7_cloned ] */
    loop_iv9_cloned__PHI_TEMPORARY = (ulong) 0;   /* for PHI node */

/* Printing: loop.header7_cloned */


/* Processing Loop Block: loop.header7_cloned */

  #pragma unroll 1
  for ( loop_iv9_cloned = 0; loop_iv9_cloned != 617; loop_iv9_cloned = loop_iv9_cloned + 1) {


/* Processing Basic Block: loop.header7_cloned */
/* loop_header7_cloned: */
/* PHINode of induction variable was here */
  load_buffer_cloned = *((&(((__global float*)_clone1_cloned))[(loop_iv9_cloned + ((ulong) 617 * tmp__1))]));
  load_buffer12_cloned = *((&(((__global float*)_clone2_cloned))[loop_iv9_cloned]));
  load_buffer13_cloned = *((&(((__global float*)_clone_cloned))[tmp__1]));
  *((&(((__global float*)_clone_cloned))[tmp__1])) = (load_buffer13_cloned + (((float)(load_buffer_cloned * load_buffer12_cloned))));
  loop_step10_cloned = loop_iv9_cloned + (ulong) 1;

/* Branch:   br i1 %loop.step11_cloned, label %loop.header7_cloned, label %loop.latch3_cloned, !llvm.loop !42 */
/* This is a loop branch! */
/* Branching back to header: loop.header7_cloned */
/* Closing loop! */
/* Printing PHIs for loop.header7_cloned->loop.header7_cloned */
/* Skipping (indvar) phi node:   %loop.iv9_cloned = phi i64 [ 0, %newFuncRoot_cloned ], [ %loop.step10_cloned, %loop.header7_cloned ] */
  }
/* Printing PHIs for loop.header7_cloned->loop.latch3_cloned */

/* Printing: loop.latch3_cloned */


/* Processing Basic Block: loop.latch3_cloned */
/* loop_latch3_cloned: */
  return;


}

__kernel 
void/* Processing Function: _Z26encoding_and_training_nodeILi2048ELi26ELi6238ELi617EEvPu11matrix_typeIXT_EXT2_EfEmPu11matrix_typeILi1EXT2_EfEmPu11matrix_typeIXT0_EXT_EfEmiPu11matrix_typeILi1EXT_EfEmPu11matrix_typeILi1EXT0_EfEmS9_mS7_mPim_loop_header_reorder_c_opencl_c_c_c: 0*/
 _Z26encoding_and_training_nodeILi2048ELi26ELi6238ELi617EEvPu11matrix_typeIXT_EXT2_EfEmPu11matrix_typeILi1EXT2_EfEmPu11matrix_typeIXT0_EXT_EfEmiPu11matrix_typeILi1EXT_EfEmPu11matrix_typeILi1EXT0_EfEmS9_mS7_mPim_loop_header_reorder_c_opencl_c_c_c(__global struct l_array_617_float* datapoint_vec_ptr, ulong datapoint_vec_size, __global struct l_array_1263616_float* rp_matrix_ptr, ulong rp_matrix_size, __global struct l_array_2048_float* encoded_hv_ptr, ulong encoded_hv_size) {

  ulong tmp__3;
  ulong tmp__4;
  __global struct l_vector_617_float* _clone2_cloned;
  __global struct l_vector_1263616_float* _clone1_cloned;
  __global struct l_vector_2048_float* _clone_cloned;
  ulong loop_iv9_cloned;
  ulong loop_iv9_cloned__PHI_TEMPORARY;
  float load_buffer_cloned;
  float load_buffer12_cloned;
  float load_buffer13_cloned;
  ulong loop_step10_cloned;


/* Printing: newFuncRoot_cloned */


/* Processing Basic Block: newFuncRoot_cloned */
/* newFuncRoot_cloned: */
  tmp__3 = get_global_id((uint) 0);
  tmp__4 = get_global_size((uint) 0);
  _clone2_cloned = ((__global struct l_vector_617_float*)datapoint_vec_ptr->array);
  _clone1_cloned = ((__global struct l_vector_1263616_float*)rp_matrix_ptr->array);
  _clone_cloned = ((__global struct l_vector_2048_float*)encoded_hv_ptr->array);
  *((&(((__global float*)_clone_cloned))[tmp__3])) = (float) 0.000000e+00;

/* Branch:   br label %loop.header7_cloned */
/* Printing PHIs for newFuncRoot_cloned->loop.header7_cloned */
/* Printing phi node:   %loop.iv9_cloned = phi i64 [ 0, %newFuncRoot_cloned ], [ %loop.step10_cloned, %loop.header7_cloned ] */
    loop_iv9_cloned__PHI_TEMPORARY = (ulong) 0;   /* for PHI node */

/* Printing: loop.header7_cloned */


/* Processing Loop Block: loop.header7_cloned */

  #pragma unroll 1
  for ( loop_iv9_cloned = 0; loop_iv9_cloned != 617; loop_iv9_cloned = loop_iv9_cloned + 1) {


/* Processing Basic Block: loop.header7_cloned */
/* loop_header7_cloned: */
/* PHINode of induction variable was here */
  load_buffer_cloned = *((&(((__global float*)_clone1_cloned))[(loop_iv9_cloned + ((ulong) 617 * tmp__3))]));
  load_buffer12_cloned = *((&(((__global float*)_clone2_cloned))[loop_iv9_cloned]));
  load_buffer13_cloned = *((&(((__global float*)_clone_cloned))[tmp__3]));
  *((&(((__global float*)_clone_cloned))[tmp__3])) = (load_buffer13_cloned + (((float)(load_buffer_cloned * load_buffer12_cloned))));
  loop_step10_cloned = loop_iv9_cloned + (ulong) 1;

/* Branch:   br i1 %loop.step11_cloned, label %loop.header7_cloned, label %loop.latch3_cloned, !llvm.loop !42 */
/* This is a loop branch! */
/* Branching back to header: loop.header7_cloned */
/* Closing loop! */
/* Printing PHIs for loop.header7_cloned->loop.header7_cloned */
/* Skipping (indvar) phi node:   %loop.iv9_cloned = phi i64 [ 0, %newFuncRoot_cloned ], [ %loop.step10_cloned, %loop.header7_cloned ] */
  }
/* Printing PHIs for loop.header7_cloned->loop.latch3_cloned */

/* Printing: loop.latch3_cloned */


/* Processing Basic Block: loop.latch3_cloned */
/* loop_latch3_cloned: */
  return;


}

__kernel 
void/* Processing Function: _Z27encoding_and_inference_nodeILi2048ELi26ELi1559ELi617EEvPu11matrix_typeIXT_EXT2_EfEmPu11matrix_typeILi1EXT2_EfEmPu11matrix_typeIXT0_EXT_EfEmPimPu11matrix_typeILi1EXT_EfEmPu11matrix_typeILi1EXT0_EfEmSA_mi_loop_header_reorder_c_opencl_c_c_c: 0*/
 _Z27encoding_and_inference_nodeILi2048ELi26ELi1559ELi617EEvPu11matrix_typeIXT_EXT2_EfEmPu11matrix_typeILi1EXT2_EfEmPu11matrix_typeIXT0_EXT_EfEmPimPu11matrix_typeILi1EXT_EfEmPu11matrix_typeILi1EXT0_EfEmSA_mi_loop_header_reorder_c_opencl_c_c_c(__global struct l_array_617_float* datapoint_vec_ptr, ulong datapoint_vec_size, __global struct l_array_1263616_float* rp_matrix_ptr, ulong rp_matrix_size, __global struct l_array_2048_float* encoded_hv_ptr, ulong encoded_hv_size) {

  ulong tmp__5;
  ulong tmp__6;
  __global struct l_vector_617_float* _clone2_cloned;
  __global struct l_vector_1263616_float* _clone1_cloned;
  __global struct l_vector_2048_float* _clone_cloned;
  ulong loop_iv9_cloned;
  ulong loop_iv9_cloned__PHI_TEMPORARY;
  float load_buffer_cloned;
  float load_buffer12_cloned;
  float load_buffer13_cloned;
  ulong loop_step10_cloned;


/* Printing: newFuncRoot_cloned */


/* Processing Basic Block: newFuncRoot_cloned */
/* newFuncRoot_cloned: */
  tmp__5 = get_global_id((uint) 0);
  tmp__6 = get_global_size((uint) 0);
  _clone2_cloned = ((__global struct l_vector_617_float*)datapoint_vec_ptr->array);
  _clone1_cloned = ((__global struct l_vector_1263616_float*)rp_matrix_ptr->array);
  _clone_cloned = ((__global struct l_vector_2048_float*)encoded_hv_ptr->array);
  *((&(((__global float*)_clone_cloned))[tmp__5])) = (float) 0.000000e+00;

/* Branch:   br label %loop.header7_cloned */
/* Printing PHIs for newFuncRoot_cloned->loop.header7_cloned */
/* Printing phi node:   %loop.iv9_cloned = phi i64 [ 0, %newFuncRoot_cloned ], [ %loop.step10_cloned, %loop.header7_cloned ] */
    loop_iv9_cloned__PHI_TEMPORARY = (ulong) 0;   /* for PHI node */

/* Printing: loop.header7_cloned */


/* Processing Loop Block: loop.header7_cloned */

  #pragma unroll 1
  for ( loop_iv9_cloned = 0; loop_iv9_cloned != 617; loop_iv9_cloned = loop_iv9_cloned + 1) {


/* Processing Basic Block: loop.header7_cloned */
/* loop_header7_cloned: */
/* PHINode of induction variable was here */
  load_buffer_cloned = *((&(((__global float*)_clone1_cloned))[(loop_iv9_cloned + ((ulong) 617 * tmp__5))]));
  load_buffer12_cloned = *((&(((__global float*)_clone2_cloned))[loop_iv9_cloned]));
  load_buffer13_cloned = *((&(((__global float*)_clone_cloned))[tmp__5]));
  *((&(((__global float*)_clone_cloned))[tmp__5])) = (load_buffer13_cloned + (((float)(load_buffer_cloned * load_buffer12_cloned))));
  loop_step10_cloned = loop_iv9_cloned + (ulong) 1;

/* Branch:   br i1 %loop.step11_cloned, label %loop.header7_cloned, label %loop.latch3_cloned, !llvm.loop !42 */
/* This is a loop branch! */
/* Branching back to header: loop.header7_cloned */
/* Closing loop! */
/* Printing PHIs for loop.header7_cloned->loop.header7_cloned */
/* Skipping (indvar) phi node:   %loop.iv9_cloned = phi i64 [ 0, %newFuncRoot_cloned ], [ %loop.step10_cloned, %loop.header7_cloned ] */
  }
/* Printing PHIs for loop.header7_cloned->loop.latch3_cloned */

/* Printing: loop.latch3_cloned */


/* Processing Basic Block: loop.latch3_cloned */
/* loop_latch3_cloned: */
  return;


}

