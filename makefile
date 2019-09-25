SRC = $(wildcard *.cu)
OBJ = $(patsubst %.cu,%.o,$(SRC))
TAR = $(SRC:%.cu=%)

CVCC = nvcc
CFLAG = -std=c++14
INC = -I /usr/local/cuda-9.0/include/

main: $(TAR)

global_contrast_kernel: global_contrast_kernel.cu
	$(CVCC) $(CFLAG) $(INC) -o $@ $^


.PHONY = clean all

all:
	@for i in $(TAR); do \
		echo compiling $$i.cu...;\
		$(CVCC) $(CFLAG) $(INC) -o $$i $$i.cu;\
	done

clean:
	@echo $(OBJ)
	$(RM) $(TAR) $(OBJ)
