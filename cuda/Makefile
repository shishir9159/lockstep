NVCC   ?= nvcc
ARCH   ?= sm_90a
CCBIN  ?=
FLAGS   = -O3 -std=c++17 -arch=$(ARCH) $(if $(CCBIN),-ccbin "$(CCBIN)",)

all: mxfp4

mxfp4: mxfp4_mma_gemm.cu fp4_unpack.cuh
	$(NVCC) $(FLAGS) -o $@ $<

# Prove which tensor-core datapath ptxas picked. On sm_90 an FP8 mma.sync
# lowers to F2FP.F16.E4M3 + HMMA.16816 -- i.e. the FP16 units.
sass: mxfp4_mma_gemm.cu fp4_unpack.cuh
	$(NVCC) $(FLAGS) --resource-usage -cubin -o mxfp4.cubin $<
	cuobjdump -sass mxfp4.cubin | grep -oE "HMMA[^ ]*|QGMMA[^ ]*|HGMMA[^ ]*|F2FP[^ ]*|LDGSTS[^ ]*" | sort | uniq -c

# Host-only exhaustive test of the FP4 -> FP8 table (no GPU needed).
test_unpack: test_unpack.c fp4_unpack.cuh
	$(CC) -O2 -I. -o $@ $<
	./$@

clean:
	rm -f mxfp4 mxfp4.cubin mxfp4.obj test_unpack *.exe

.PHONY: all sass clean
