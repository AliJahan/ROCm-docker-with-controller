#include "resource_controller.hpp"

namespace Control{
    CUMaskingSharedMemory* cumasking_shm_init(uint32_t num_gpus) {
        const char *name = "/z_cu_shm";
        CUMaskingSharedMemory* cumasking_shm = new CUMaskingSharedMemory();
        
        errno = 0;

        
        // Open existing shared memory object, or create one.
        cumasking_shm->shm_fd = shm_open(name, O_RDWR|O_CREAT, 0660);
        // create cu masking space per gpu (each uint32_t holds the cu masking value set by controller)
        cumasking_shm->shm_size = cumasking_shm->base_shm_size + num_gpus * sizeof(uint32_t); 
        if (cumasking_shm->shm_fd == -1) {
            // perror("shm_open failed \n");
            return nullptr;
        }
        if(ftruncate(cumasking_shm->shm_fd, cumasking_shm->shm_size) != 0 ) {
            // perror("ftruncate");
            return nullptr;
        }
        // Map into the shared memory.
        void *base_addr = mmap(
            nullptr,
            sizeof(cumasking_shm->shm_size),
            PROT_READ|PROT_WRITE,
            MAP_SHARED,
            cumasking_shm->shm_fd,
            0
        );

        if (base_addr == MAP_FAILED) {
            // perror("mmap failed\n");
            return nullptr;
        }

        cumasking_shm->base_addr = base_addr;
        uint32_t* num_gpus_ptr = (uint32_t*)base_addr;              // number of GPUs space
        uint32_t* gpus_mask_ptr = (uint32_t*)(num_gpus_ptr + 1U);   // CU mask space per each GPUs
        *(num_gpus_ptr) = num_gpus;
        for(uint32_t i = 0U; i < num_gpus; i++){                    // Initialize cu mask for all GPUs to 60 (= default = max number of cus )
            *(gpus_mask_ptr + i) = 60U;                             // TODO: get it from ROCM API to support all gpus.
        }
        // Initialize interprocess cumasking_shm in SHM
        pthread_mutex_t *mutex_ptr = (pthread_mutex_t *)(gpus_mask_ptr + num_gpus);
        pthread_mutexattr_t attr;
        if (pthread_mutexattr_init(&attr)) {
            perror("pthread_mutexattr_init failed in ROCR/src/core/runtime/shared_mutex.cpp\n");
            //TODO: cleanup before return
            return nullptr;
        }
        if (pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED)) {
            perror("pthread_mutexattr_setpshared failed in ROCR/src/core/runtime/shared_mutex.cpp\n");
            //TODO: cleanup before return
            return nullptr;
        }
        if (pthread_mutex_init(mutex_ptr, &attr)) {
            perror("pthread_mutex_init failed in ROCR/src/core/runtime/shared_mutex.cpp\n");
            //TODO: cleanup before return
            return nullptr;
        }
        // Set up CUMaskingSharedMemory
        cumasking_shm->mutex_ptr = mutex_ptr;
        cumasking_shm->num_gpus_ptr = num_gpus_ptr;
        cumasking_shm->gpus_num_cu_ptr = gpus_mask_ptr;
        cumasking_shm->created = 1;
        cumasking_shm->name = (char *)malloc(NAME_MAX+1);
        strcpy(cumasking_shm->name, name);
        
        return cumasking_shm;
    }
    // Closes the SHM first, and then frees all allocated resources (make sure the mutex is unlocked, this DOES NOT UNLOCK the mutex)
    int cumasking_shm_destroy(CUMaskingSharedMemory*& cumasking_shm) {
        if(cumasking_shm == nullptr || cumasking_shm->created == 0) // this process did not create shm, do not destroy it
            return -1;
        if ((errno = pthread_mutex_destroy(cumasking_shm->mutex_ptr))) {
            perror("pthread_mutex_destroy failed in ROCR/src/core/runtime/shared_mutex.cpp");
            return -1;
        }
        if (munmap(cumasking_shm->base_addr, cumasking_shm->shm_size)) {
            perror("munmap failed in ROCR/src/core/runtime/shared_mutex.cpp");
            return -1;
        }
        if (close(cumasking_shm->shm_fd)) {
            perror("close failed in ROCR/src/core/runtime/shared_mutex.cpp");
            return -1;
        }
        cumasking_shm->shm_fd = 0;
        if (shm_unlink(cumasking_shm->name)) {
            perror("shm_unlink failed in ROCR/src/core/runtime/shared_mutex.cpp");
            return -1;
        }
        free(cumasking_shm->name);
        free(cumasking_shm);
        cumasking_shm->mutex_ptr = nullptr;
        cumasking_shm->base_addr = nullptr;
        cumasking_shm->gpus_num_cu_ptr = nullptr;
        cumasking_shm->num_gpus_ptr = nullptr;
        cumasking_shm->created = 0;
        cumasking_shm = nullptr;
        return 0;
    }
}