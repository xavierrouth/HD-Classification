HPVM_INSTALL_DIR ?= /home/arnoor2/hpvm_hdc/hpvm/hpvm/
HPVM_BUILD_DIR := $(HPVM_INSTALL_DIR)/build/
INCLUDES := -I$(HPVM_INSTALL_DIR)/include -I$(HPVM_INSTALL_DIR)/benchmarks/include -I$(HPVM_INSTALL_DIR)/projects/hpvm-hdc -I$(HPVM_INSTALL_DIR)/projects/hetero-c++/test/include -I./src

HPVM_LIB := $(HPVM_INSTALL_DIR)/build/lib
HPVM_BIN := $(HPVM_INSTALL_DIR)/build/bin

HPVM_DECLS_FILE := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hetero-c++/lib/HPVMCFunctionDeclarations/HPVMCFunctionDeclarations.bc

CUDA_PATH:=/software/cuda-9.1
OPENCL_PATH:=$(CUDA_PATH)
OPENCL_LIB_PATH:=$(OPENCL_PATH/lib64)

CFLAGS := -fenable-matrix -fno-discard-value-names -fno-exceptions -lm -lrt -pthread -lOpenCL

HDC_LIBS := -load $(HPVM_LIB)/LLVMHPVMHDC.so -load $(HPVM_LIB)/LLVMHDCBinProp.so  -load $(HPVM_LIB)/LLVMBinarize.so  

NVCC ?= "/usr/local/cuda"/bin/nvcc

O1_FLAGS = -targetlibinfo -tti -tbaa -scoped-noalias-aa -assumption-cache-tracker -profile-summary-info -forceattrs -inferattrs -ipsccp -called-value-propagation -globalopt -domtree -mem2reg -deadargelim -basic-aa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -simplifycfg -basiccg -globals-aa -prune-eh -always-inline -function-attrs  -memoryssa -early-cse-memssa -speculative-execution -lazy-value-info -jump-threading -correlated-propagation -libcalls-shrinkwrap -branch-prob -block-freq -pgo-memop-opt -tailcallelim -reassociate -loop-simplify -lcssa-verification -lcssa -scalar-evolution -loop-rotate -licm -loop-unswitch -indvars -loop-idiom -loop-deletion -loop-unroll -memdep -memcpyopt -sccp -demanded-bits -bdce -dse -postdomtree -adce -barrier -rpo-function-attrs -globaldce -float2int -loop-accesses -loop-distribute -loop-vectorize -loop-load-elim -alignment-from-assumptions -strip-dead-prototypes -loop-sink -instsimplify -div-rem-pairs -verify -ee-instrument -early-cse -lower-expect

# PARAMETERS := -DHPVM -DDEVICE=hpvm::CPU_TARGET
#
#
#
### ---------- SIMULATOR ---------------

# Generate Executable
host-sim: build/hdc.lowered.sim.ll accelerator_api/hpvm_interface.cpp accelerator_api/api_hdnn_reram.hpp
	$(HPVM_INSTALL_DIR)/build/bin/clang++ -lm -lrt -g -Ofast -march=native $< accelerator_api/hpvm_interface.cpp   -o $@ 

# Clang Frontend to IR
build/host.sim.ll: src/host.cpp
	$(HPVM_BIN)/clang++ -DNODFG -DACCEL $(CFLAGS) $(INCLUDES) -S -emit-llvm $^ -o $@ 

# GenHPVM for HDC only and mem2reg
build/hpvm.genhdc.sim.ll: build/host.sim.ll
	$(HPVM_BIN)/opt -mem2reg -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@


# Lower HDC Intrinsics
build/hdc.lowered.sim.ll: build/hpvm.genhdc.sim.ll #hpvm. # -enable-hcc 
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/LLVMHPVMHDC.so  -vector-block-dim=1 -lower-hdc -compile-sim -dce -simplifycfg $< -S -o $@ 

### ---------- ACCEL ---------------

# Generate Executable
host-accel: build/hdc.lowered.accel.ll accelerator_api/hpvm_interface.cpp accelerator_api/api_hdnn_reram.hpp
	$(HPVM_INSTALL_DIR)/build/bin/clang++ -lm -lrt -g -Ofast -march=native $< accelerator_api/hpvm_interface.cpp   -o $@ 

# Clang Frontend to IR
build/host.accel.ll: src/host.cpp
	$(HPVM_BIN)/clang++ -DNODFG -DACCEL $(CFLAGS) $(INCLUDES) -S -emit-llvm $^ -o $@ 

# GenHPVM for HDC only and mem2reg
build/hpvm.genhdc.accel.ll: build/host.accel.ll
	$(HPVM_BIN)/opt -mem2reg -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@


# Lower HDC Intrinsics
build/hdc.lowered.accel.ll: build/hpvm.genhdc.accel.ll #hpvm. # -enable-hcc 
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/LLVMHPVMHDC.so -vector-block-dim=1 -lower-hdc -compile-accel -dce -simplifycfg $< -S -o $@ 

### ---------- CPU ---------------

# Generate Executable
host-cpu: build/output.ll
	$(HPVM_INSTALL_DIR)/build/bin/clang++ -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) -lm -lrt -pthread -lOpenCL -O3 $< -o $@ 
#

# Clang Frontend to IR
build/host.cpu.ll: src/host.cpp src/DFG.hpp src/host.h
	$(HPVM_BIN)/clang++ -O0 -Xclang -disable-O0-optnone -DHPVM -DDEVICE=hpvm::CPU_TARGET $(CFLAGS) $(INCLUDES) -S -emit-llvm $< -o $@ 
#$HPVM_BIN/clang++ -O1 -I$OPENCL_PATH/include/CL -L$OPENCL_LIB_PATH -I$HPVM_ROOT/projects/hetero-c++/test/include/ -lm -lrt -pthread -lOpenCL  $INCLUDE_FLAG -fenable-matrix    $PREFIX.cpp -S -emit-llvm -o $PREFIX.ll

# GenHPVM for HDC only and mem2reg
build/hpvm.genhdc.cpu.ll: build/host.cpu.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@
# $OPT -mem2reg -enable-new-pm=0 -load $HPVM_LIB/HPVMGenHPVM.so -genhpvm  -hpvm-hdc-only $PREFIX.ll -S -o $PREFIX.genhdc.hpvm.ll

# Lower HDC Intrinsics
build/hdc.lowered.cpu.ll: build/hpvm.genhdc.cpu.ll #hpvm. # -enable-hcc 
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) -hdc-bin-prop -hdc-binarize -vector-block-dim=1 -lower-hdc  -dce -simplifycfg $< -S -o $@ 
	
# HCC -> HPVM
build/hpvmc.cpu.ll: build/hdc.lowered.cpu.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) -sanitise-funcs --dot-dfg -skip-verify  -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

# GenHPVM for the HPVM IR
build/hpvm.cpu.ll: build/hpvmc.cpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -dce -mem2reg -genhpvm $< -S -o $@

# Other HPVM Passes
build/hpvmcompiled.cpu.ll: build/hpvm.cpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMBuildDFG.so -load $(HPVM_LIB)/HPVMDFG2LLVM_CPU.so -load $(HPVM_LIB)/HPVMClearDFG.so -load $(HPVM_LIB)/HPVMDFGTransformPasses.so -dfg2llvm-cpu -clearDFG -hpvm-timers-cpu $< -S -o $@

HPVM_RT := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc

# LINK HPVM RT and DCE
build/output.ll: build/hpvmcompiled.cpu.ll # build/cpu_test.hpvmcompiled.ll
	$(HPVM_INSTALL_DIR)/build/bin/llvm-link $< $(HPVM_RT) -S -o $@ 
	$(HPVM_INSTALL_DIR)/build/bin/opt -dce -globaldce $< -S -o $<

### ---------- GPU ---------------

# Generate Executable
host-gpu: build/output.gpu.ll
	$(HPVM_INSTALL_DIR)/build/bin/clang++ -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) -lm -lrt -pthread -lOpenCL -O2 $< -o $@

# Clang Frontend to IR
build/host.gpu.ll: src/host.cpp
	$(HPVM_BIN)/clang++ -DHPVM -DDEVICE=hpvm::GPU_TARGET $(CFLAGS) $(INCLUDES) -S -emit-llvm $^ -o $@ 
#$HPVM_BIN/clang++ -O1 -I$OPENCL_PATH/include/CL -L$OPENCL_LIB_PATH -I$HPVM_ROOT/projects/hetero-c++/test/include/ -lm -lrt -pthread -lOpenCL  $INCLUDE_FLAG -fenable-matrix    $PREFIX.cpp -S -emit-llvm -o $PREFIX.ll

# GenHPVM for HDC only and mem2reg
build/hpvm.genhdc.gpu.ll: build/host.gpu.ll
	$(HPVM_BIN)/opt -mem2reg -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@
# $OPT -mem2reg -enable-new-pm=0 -load $HPVM_LIB/HPVMGenHPVM.so -genhpvm  -hpvm-hdc-only $PREFIX.ll -S -o $PREFIX.genhdc.hpvm.ll

# Lower HDC Intrinsics
build/hdc.lowered.gpu.ll: build/hpvm.genhdc.gpu.ll #hpvm.
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/LLVMHPVMHDC.so -vector-block-dim=1 -lower-hdc -enable-hcc -dce -simplifycfg $< -S -o $@

# HCC -> HPVM
build/hpvmc.gpu.ll: build/hdc.lowered.gpu.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) -sanitise-funcs --dot-dfg --skip-verify -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

# GenHPVM for the HPVM IR
build/hpvm.gpu.ll: build/hpvmc.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -dce -mem2reg -genhpvm $< -S -o $@

build/hpvmcompiled.gpu.ll: build/hpvm.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMBuildDFG.so -load $(HPVM_LIB)/HPVMLocalMem.so -load $(HPVM_LIB)/HPVMDFG2LLVM_GPU_OCL.so -load $(HPVM_LIB)/HPVMDFG2LLVM_CPU.so -load $(HPVM_LIB)/HPVMClearDFG.so -load $(HPVM_LIB)/HPVMDFGTransformPasses.so -localmem -dfg2llvm-gpu-ocl -dfg2llvm-cpu -clearDFG -hpvm-timers-cpu -hpvm-timers-ptx $< -S -o $@
	#$(HPVM_BIN)/llvm-ocl build/hpvm.gpu.kernels.ll

HPVM_RT := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc

# LINK HPVM RT and DCE
build/output.gpu.ll: build/hpvmcompiled.gpu.ll # build/gpu_test.hpvmcompiled.ll
	$(HPVM_INSTALL_DIR)/build/bin/llvm-link $< $(HPVM_RT) -S -o $@ 
	$(HPVM_INSTALL_DIR)/build/bin/opt -dce -globaldce $< -S -o $<

### ---------- FPGA ---------------

# Generate Executable
host-fpga: build/output.fpga.ll
	$(HPVM_INSTALL_DIR)/build/bin/clang++ -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) -lm -lrt -pthread -lOpenCL -O2 $< -o $@

# Clang Frontend to IR
build/host.fpga.ll: src/host.cpp
	$(HPVM_BIN)/clang++ -DHPVM -DFPGA -DDEVICE=hpvm::FPGA_TARGET $(CFLAGS) $(INCLUDES) -S -emit-llvm $^ -o $@ 
#$HPVM_BIN/clang++ -O1 -I$OPENCL_PATH/include/CL -L$OPENCL_LIB_PATH -I$HPVM_ROOT/projects/hetero-c++/test/include/ -lm -lrt -pthread -lOpenCL  $INCLUDE_FLAG -fenable-matrix    $PREFIX.cpp -S -emit-llvm -o $PREFIX.ll

# GenHPVM for HDC only and mem2reg
build/hpvm.genhdc.fpga.ll: build/host.fpga.ll
	$(HPVM_BIN)/opt -mem2reg -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@
# $OPT -mem2reg -enable-new-pm=0 -load $HPVM_LIB/HPVMGenHPVM.so -genhpvm  -hpvm-hdc-only $PREFIX.ll -S -o $PREFIX.genhdc.hpvm.ll

# Lower HDC Intrinsics
build/hdc.lowered.fpga.ll: build/hpvm.genhdc.fpga.ll #hpvm.
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/LLVMHPVMHDC.so -vector-block-dim=1 -lower-hdc  -dce -simplifycfg $< -S -o $@

# HCC -> HPVM
build/hpvmc.fpga.ll: build/hdc.lowered.fpga.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) -sanitise-funcs --dot-dfg  -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

build/hpvm.host.fpga.ll build/hpvm.kernels.fpga.ll build/hpvm.kernels.fpga.cl build/hpvm.kernels.fpga.aocx: build/hpvmc.fpga.ll
	$(HPVM_BIN)/hpvm2fpga -hm=build/hpvm.host.fpga.ll -km=build/hpvm.kernels.fpga.ll -clm=build/hpvm.kernels.fpga.cl -cxm=build/hpvm.kernels.fpga.aocx -run-aoc $^


HPVM_RT := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc

# LINK HPVM RT and DCE
build/output.fpga.ll: build/hpvm.host.fpga.ll
	$(HPVM_INSTALL_DIR)/build/bin/llvm-link $< $(HPVM_RT) -S -o $@ 
	$(HPVM_INSTALL_DIR)/build/bin/opt -dce -globaldce $< -S -o $<

### ---------- CUBLAS ---------------

# Generate Executable
host-cublas: build/host.cublas.o build/hdc_cuda_rt.o build/hdc_kernels.o build/hdc_cuda_rt.o build/hdc_kernels.o
	$(NVCC) $^ -o $@ -lcudadevrt -lcudart -lcublas

# Clang Frontend to IR
build/host.cublas.ll: src/host.cpp
	mkdir -p build
	$(HPVM_BIN)/clang++ -DNODFG -O0 -Xclang -disable-O0-optnone $(CFLAGS) $(INCLUDES) -S -emit-llvm -fPIC $^ -o $@ 

# GenHPVM for HDC only and mem2reg
build/hpvm.genhdc.cublas.ll: build/host.cublas.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@


# Lower HDC Intrinsics
build/hdc.lowered.cublas.ll: build/hpvm.genhdc.cublas.ll #hpvm. # -enable-hcc 
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/LLVMHPVMHDC.so  -vector-block-dim=1 -lower-hdc -compile-cublas -dce -simplifycfg $(O1_FLAGS) $< -S -o $@ 
build/host.cublas.o: build/hdc.lowered.cublas.ll
	$(HPVM_BIN)/clang -fPIC -c -Ofast -march=native $< -o $@

build/hdc_cuda_rt.o: $(HPVM_INSTALL_DIR)/projects/hpvm-hdc/cuda-impl/hdc_cuda_rt.cu
	$(NVCC) -g -c -O2 $^ -o $@ 
build/hdc_kernels.o: $(HPVM_INSTALL_DIR)/projects/hpvm-hdc/cuda-impl/hdc_kernels.cu
	$(NVCC) -g -c -O2 $^ -o $@ 

.PHONY: clean

clean:
	rm -rf *.dot build/*.ll host-gpu host-cpu host-accel host-sim host-fpga host-cublas
