# targets
HOST_EXE := host

CFLAGS := -g -std=c++11
LFLAGS :=

# primary build targets
.PHONY: all clean

all: $(HOST_EXE)

clean:
	rm -f $(HOST_EXE) build/*.o

# $(HOST_EXE): build/host.o build/hd.o
# 	g++ $(CFLAGS) -o $@ $^ $(LFLAGS)
# build/host.o: src/host.cpp src/host.h src/hd.h
# 	g++ $(CFLAGS) -c -o $@ $<
# build/hd.o: src/hd.cpp src/hd.h
# 	g++ $(CFLAGS) -c -o $@ $<
$(HOST_EXE): src/host.cpp src/hd.cpp src/host.h src/hd.h
	hpvm-clang -DDEVICE=CPU_TARGET --hetero-cc --hpvm-target cpu -d build src/hd.cpp src/host.cpp -o host
