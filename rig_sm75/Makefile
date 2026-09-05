NVCC   ?= nvcc
ARCH   ?= sm_75
CCBIN  ?=
FLAGS   = -O3 -std=c++17 -arch=$(ARCH) $(if $(CCBIN),-ccbin "$(CCBIN)",)

all: rig

rig: rig.cu fp4_sim.cuh
	$(NVCC) $(FLAGS) -o $@ $<

run: rig
	./rig

clean:
	rm -f rig rig.exe *.exp *.lib *.obj

.PHONY: all run clean
