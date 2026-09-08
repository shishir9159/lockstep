NVCC   ?= nvcc
ARCH   ?= sm_75
CCBIN  ?=
FLAGS   = -O3 -std=c++17 -arch=$(ARCH) $(if $(CCBIN),-ccbin "$(CCBIN)",)

all: rig rig_q

rig: rig.cu fp4_sim.cuh
	$(NVCC) $(FLAGS) -o $@ $<

rig_q: rig_q.cu fp4_sim.cuh
	$(NVCC) $(FLAGS) -o $@ $<

run: rig rig_q
	./rig
	./rig_q

clean:
	rm -f rig rig_q rig.exe rig_q.exe *.exp *.lib *.obj

.PHONY: all run clean
