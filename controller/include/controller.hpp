#include <iostream>
#include <atomic>
#include <thread>
#include <cassert>
#include <string>
#include <sstream>
#include <vector>
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
        // 2 values for cumask
        uint32_t value1; 
        uint32_t value2; 
        control_msg_t(): 
            command_type(command_t::NOOP),
            gpu_ind(0U),
            value1(0xffffffff),
            value2(0xfffffff) {} // default cu mask (max cus)

        control_msg_t(
            const command_t& command,
            uint32_t gpu_ind,
            uint32_t value1,
            uint32_t value2
        ):
            command_type(command),
            gpu_ind(gpu_ind),
            value1(value1),
            value2(value2) {}
        
        // copy const.
        control_msg_t(const struct control_msg_t& other):
            command_type(other.command_type),
            gpu_ind(other.gpu_ind),
            value1(other.value1),
            value2(other.value2) {}

        // assign opt.
        control_msg_t& operator=(const struct control_msg_t& other){
            command_type = other.command_type;
            gpu_ind = other.gpu_ind;
            value1 = other.value1;
            value2 = other.value2;
            return *this;
        }
    };

    friend std::ostream& operator<<(std::ostream& os, enum Control::Controller::command_t const& c);

    friend std::ostream& operator<<(std::ostream& os, struct Control::Controller::control_msg_t const& m);

    Controller(std::string contoller_remote_ip, std::string contoller_remote_port, std::string app_name): 
        ctx_(1),
        socket_ptr_(nullptr),
        control_address(
            "tcp://"+contoller_remote_ip+":"+contoller_remote_port
        ),
        app_name(app_name),
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
    void set_cus(uint32_t gpu_ind, const uint32_t mask0, const uint32_t mask1);
    void set_freq(uint32_t gpu_ind, uint32_t freq);
    bool init_rocm_smi();
    void deinit_rocm_smi();
    bool init_controller();
    void deinit_controller();
    bool init_shm();
    void deinit_shm();
    bool init_zmq();
    void deinit_zmq();
    void split_string(const std::string &s, char delim, std::vector<std::string>& splitted);
    bool hexstr2uint32(std::string mask_str, uint32_t& mask_uint32);
    inline uint32_t count_set_bits(uint32_t n); // utility
    std::string uint2hexstr(uint32_t num); //utility
    bool get_command(zmq::message_t& msg);
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
    std::atomic_bool running_;
    std::thread* thread_;
    std::string control_address;
    std::string app_name;
};
}