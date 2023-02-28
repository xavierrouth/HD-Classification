INCLUDES += -I$(HPVM_INSTALL_DIR)/include -I$(HPVM_INSTALL_DIR)/benchmarks/include

CFLAGS :=
LFLAGS :=

host-normal: src/host.cpp src/hd.cpp src/host.h src/hd.h src/hd.h
	clang++ $(CFLAGS) -o $@ src/host.cpp src/hd.cpp $(LFLAGS)
host-hetero: src/host.cpp src/hd.cpp src/host.h src/hd.h src/hd.h
	hpvm-clang -v -DHPVM -DDEVICE=hpvm::CPU_TARGET --hetero-cc --hpvm-target cpu -d build $(INCLUDES) src/hd.cpp src/host.cpp -o host-hetero

.PHONY: clean

clean:
	rm -f host-normal host-hetero build/*.o
