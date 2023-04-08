INCLUDES += -I$(HPVM_INSTALL_DIR)/include -I$(HPVM_INSTALL_DIR)/benchmarks/include

CFLAGS := -O1 -ffast-math -fno-lax-vector-conversions -fno-vectorize -fno-slp-vectorize -finline -finline-functions
LFLAGS :=

host-normal: src/host.cpp src/hd.cpp src/host.h src/hd.h src/hd.h
	clang++ $(CFLAGS) -o $@ src/host.cpp src/hd.cpp $(LFLAGS)

build/host.cpu.ll: src/host.cpp src/host.h src/hd.h
	clang++ -DHPVM -DDEVICE=hpvm::CPU_TARGET $(CFLAGS) $(INCLUDES) -emit-llvm -S -o $@ $<
build/hd.cpu.ll: src/hd.cpp src/hd.h
	clang++ -DHPVM -DDEVICE=hpvm::CPU_TARGET $(CFLAGS) $(INCLUDES) -emit-llvm -S -o $@ $<
build/linked.hetero.cpu.ll: build/host.cpu.ll build/hd.cpu.ll
	llvm-link -S -o $@ $^
build/hpvmc.cpu.ll: build/linked.hetero.cpu.ll
	hcc $< -declsfile $(HPVM_BUILD_DIR)/tools/hpvm/projects/hetero-c++/lib/HPVMCFunctionDeclarations/HPVMCFunctionDeclarations.bc -sanitise-funcs -S -o $@
build/hpvm.host.cpu.ll: build/hpvmc.cpu.ll
	opt -enable-new-pm=0 -inline -load HPVMGenHPVM.so -genhpvm -globaldce -hpvm-timers-gen -load HPVMBuildDFG.so -load HPVMDFG2LLVM_CPU.so -load HPVMClearDFG.so -load HPVMDFGTransformPasses.so  -sequentializeflatten -dfg2llvm-cpu -clearDFG -hpvm-timers-cpu -S $< -o $@
build/host-cpu.ll: build/hpvm.host.cpu.ll
	llvm-link -S -o $@ $^ $(HPVM_BUILD_DIR)/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc
host-cpu: build/host-cpu.ll
	clang++ -O1 -lm -lrt -lpthread -lOpenCL $^ -o $@

build/host.fpga.ll: src/host.cpp src/host.h src/hd.h
	clang++ -DHPVM -DDEVICE=hpvm::FPGA_TARGET $(CFLAGS) $(INCLUDES) -emit-llvm -S -o $@ $<
build/hd.fpga.ll: src/hd.cpp src/hd.h
	clang++ -DHPVM -DDEVICE=hpvm::FPGA_TARGET $(CFLAGS) $(INCLUDES) -emit-llvm -S -o $@ $<
build/linked.hetero.fpga.ll: build/host.fpga.ll build/hd.fpga.ll
	llvm-link -S -o $@ $^
build/hpvmc.fpga.ll: build/linked.hetero.fpga.ll
	hcc $< -declsfile $(HPVM_BUILD_DIR)/tools/hpvm/projects/hetero-c++/lib/HPVMCFunctionDeclarations/HPVMCFunctionDeclarations.bc -sanitise-funcs -S -o $@
build/hpvm.host.fpga.ll build/hpvm.kernels.fpga.ll build/hpvm.kernels.fpga.cl hpvm.kernels.fpga.aocx: build/hpvmc.fpga.ll
	hpvm2fpga -hm=build/hpvm.host.fpga.ll -km=build/hpvm.kernels.fpga.ll -clm=build/hpvm.kernels.fpga.cl -cxm=build/hpvm.kernels.fpga.aocx -emu -run-aoc $^
build/host-fpga.ll: build/hpvm.host.fpga.ll
	llvm-link -S -o $@ $^ $(HPVM_BUILD_DIR)/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc
host-fpga: build/host-fpga.ll
	clang++ -O1 -lm -lrt -lpthread -lOpenCL $^ -o $@

.PHONY: clean

clean:
	rm -rf host-normal host-cpu host-fpga build/*.o build/*.ll build/*.tmp build/*.cl *.temp *.aoco build/*.aocx build/*.aocr
