################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/glutFrontend.cu 

OBJS += \
./src/glutFrontend.o 

CU_DEPS += \
./src/glutFrontend.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -D_FORCE_INLINES -G -g -O3 --use_fast_math -keep -Xcompiler -Wall -Xcompiler -Wextra -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -D_FORCE_INLINES -G -g -O3 --use_fast_math -keep -Xcompiler -Wall -Xcompiler -Wextra -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


