#include <iostream>
#include <atomic>
#include <thread>
#include <cassert>
#include <zmq.h>
#include <zmq_addon.hpp>


#include "resource_controller.hpp"
#include "rocm_smi/rocm_smi.h"

namespace Control{

class Controller{
public:
    enum command_t{
        NOOP = 0,
        SET_FREQ = 1,
        RESET_FREQ = 2,
        SET_CUMASK = 3,
        RESET_CUMASK = 4,
        TOT_COMMANDS
    };

    struct control_msg_t{
        command_t command_type;
        uint32_t gpu_ind;
        int32_t value; 
        control_msg_t(): 
            command_type(command_t::NOOP),
            gpu_ind(0U),
            value(-1) {}

        control_msg_t(
            const command_t& command,
            uint32_t gpu_ind,
            int32_t value
        ):
            command_type(command),
            gpu_ind(gpu_ind),
            value(value) {}
        
        // copy const.
        control_msg_t(const struct control_msg_t& other):
            command_type(other.command_type),
            gpu_ind(other.gpu_ind),
            value(other.value) {}

        // assign opt.
        control_msg_t& operator=(const struct control_msg_t& other){
            command_type = other.command_type;
            gpu_ind = other.gpu_ind;
            value = other.value;
            return *this;
        }
    };

    friend std::ostream& operator<<(std::ostream& os, enum Control::Controller::command_t const& c);

    friend std::ostream& operator<<(std::ostream& os, struct Control::Controller::control_msg_t const& m);

    Controller(std::string port_str): 
        ctx_(1),
        socket_ptr_(nullptr),
        control_socket_address_(
            "tcp://*:"+port_str
        ),
        smi_initted_(false),
        num_gpus_(0U),
        shm_(nullptr),
        shm_initted_(false),
        zmq_initted_(false),
        running_(false),
        thread_(nullptr) {};

    ~Controller() { stop(); }
    bool start(bool detached);
    void stop();
private:
    void run();
    void set_cus(uint32_t gpu_ind, uint32_t num_cus);
    void set_freq(uint32_t gpu_ind, uint32_t freq);
    bool init_rocm_smi();
    void deinit_rocm_smi();
    bool init_controller();
    void deinit_controller();
    bool init_shm();
    void deinit_shm();
    bool init_zmq();
    void deinit_zmq();
    zmq::message_t get_command();
    void apply_command(zmq::message_t& msg);
    Controller() {}; // no default ctor.
    std::string get_cur_time_str();
    uint32_t num_gpus_;
    uint64_t gpu_min_power_;
    uint64_t gpu_max_power_;
    Control::CUMaskingSharedMemory* shm_;
    std::atomic_bool shm_initted_;
    zmq::context_t ctx_;
    zmq::socket_t* socket_ptr_;
    std::atomic_bool zmq_initted_;
    std::atomic_bool smi_initted_;
    std::string control_socket_address_;
    std::atomic_bool running_;
    std::thread* thread_;
};
}