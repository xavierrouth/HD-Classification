INCLUDES += -I$(HPVM_INSTALL_DIR)/include -I$(HPVM_INSTALL_DIR)/benchmarks/include

CFLAGS := -O1 -ffast-math -fno-lax-vector-conversions -fno-vectorize -fno-slp-vectorize -finline -finline-functions
LFLAGS :=

host-normal: src/host.cpp src/hd.cpp src/host.h src/hd.h src/hd.h
	clang++ $(CFLAGS) -o $@ src/host.cpp src/hd.cpp $(LFLAGS)

build/host.ll: src/host.cpp src/host.h src/hd.h
	clang++ -DHPVM -DDEVICE=hpvm::FPGA_TARGET $(CFLAGS) $(INCLUDES) -emit-llvm -S -o $@ $<
build/hd.ll: src/hd.cpp src/hd.h
	clang++ -DHPVM -DDEVICE=hpvm::FPGA_TARGET $(CFLAGS) $(INCLUDES) -emit-llvm -S -o $@ $<
build/linked.hetero.ll: build/host.ll build/hd.ll
	llvm-link -S -o $@ $^
build/hpvmc.ll: build/linked.hetero.ll
	hcc $< -declsfile $(HPVM_BUILD_DIR)/tools/hpvm/projects/hetero-c++/lib/HPVMCFunctionDeclarations/HPVMCFunctionDeclarations.bc -sanitise-funcs -S -o $@
build/hpvm.host.ll build/hpvm.kernels.ll build/hpvm.kernels.cl hpvm.kernels.aocx: build/hpvmc.ll
	hpvm2fpga -hm=build/hpvm.host.ll -km=build/hpvm.kernels.ll -clm=build/hpvm.kernels.cl -cxm=build/hpvm.kernels.aocx -emu -run-aoc $^
build/host-hetero.ll: build/hpvm.host.ll
	llvm-link -S -o $@ $^ $(HPVM_BUILD_DIR)/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc
host-hetero: build/host-hetero.ll
	clang++ -O1 -lm -lrt -lpthread -lOpenCL $^ -o $@

.PHONY: clean

clean:
	rm -rf host-normal host-hetero build/*.o build/*.ll build/*.tmp build/*.cl *.temp *.aoco
