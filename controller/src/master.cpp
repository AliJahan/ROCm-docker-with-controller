#include <zmq.h>
#include <zmq_addon.hpp>
#include <stdio.h>
#include <sstream> // for uint32_t to hex str conversion
#include <iostream>

#include "controller.hpp"

inline uint32_t count_set_bits(uint32_t n){
    uint32_t count = 0;
    while (n) {
        n &= (n - 1);
        count++;
    }
    return count;
}
bool hexstr2uint32(std::string mask_str, uint32_t& mask_uint32){
    if(mask_str.size() != 8){
        std::cout << "error: mask has to be 8-charachter hex. Example: ffffffff.\nusage: \n" << "SET CU MASK:\n\t./master cu <gpu_ind> <8-charachter hex mask0> <8-charachter hex mask1>\n";
        return false;
    }
    try {
        std::stringstream ss;
        ss << std::hex << mask_str;
        ss >> mask_uint32;
        return true;
    }
    catch (...) {
        std::cout << "error: could not convert " << mask_str << " to hex\n.\nusage: \n" << "SET CU MASK:\n\t./master cu <gpu_ind> <8-charachter hex mask0> <8-charachter hex mask1>\n";
        return false;
    }
}
int main(int argc, char *argv[]){
    
    if(argc < 3){
        std::cout << "usage: \n" << "SET CU MASK:\n\t./master cu <gpu_ind> <8-charachter hex mask0> <8-charachter hex mask1>\n" << "SET GPU FREQ:\n\t./master freq <gpu_ind> <freq>\n";
        return -1;
    }
    std::string cmd(argv[1]);
    if(cmd == "cu" && argc < 4){
        std::cout << "usage: \n" << "SET CU MASK:\n\t./master cu <gpu_ind> <8-charachter hex mask0> <8-charachter hex mask1>\n";
        return -1;
    }
    // get gpu index
    uint32_t gpu = std::stoi(argv[2]);
    if(gpu < 0U || gpu >7U){
        std::cout << "gpu index provided is out of range, use [0,7] values\n";
        return -1;
    }
    uint32_t value1 = 0U;
    uint32_t value2 = 0U;
    if(cmd == "power"){
        value1 = (uint32_t)std::stoi(argv[3]);
        if(value1 < 1U || value1 > 225U){
            std::cout << "power value has to be in range [1,225]\n";
            return -1;
        }
    }
    else if(cmd == "cu"){
        if(hexstr2uint32(std::string(argv[3]), value1)== false){
            std::cout << "error parsing hex value: " << std::string(argv[3]) << "\n";
            return -1;
        }
        if(hexstr2uint32(std::string(argv[4]), value2) == false){
            std::cout << "error parsing hex value: " << std::string(argv[4]) << "\n";
            return -1;
        }
        uint32_t requests_cus = count_set_bits(value1) + count_set_bits(value2);
        std::cout << "masks: 0x" << std::string(argv[3]) << " 0x" << std::string(argv[4]) << " num cus: " << requests_cus << std::endl << std::flush;
        if(requests_cus < 1U || requests_cus > 60U){
            std::cout << "cu masks provided exceed 60 CU or less than 1 CU " << requests_cus << "\n";
            return -1;
        }
    }
    
    zmq::context_t ctx_(1);
    zmq::socket_t* socket_ptr_;
    socket_ptr_ = new zmq::socket_t(ctx_, zmq::socket_type::dealer);
    uint32_t id_int = 16; //TODO
    std::string id(std::to_string(id_int));
    // std::string id("power");
    zmq_setsockopt(*socket_ptr_, ZMQ_SUBSCRIBE, id.c_str(), id.size());
    // uint32_t msg_to_keep = 2;
    // zmq_setsockopt(*socket_ptr_, ZMQ_CONFLATE, &msg_to_keep, sizeof(uint32_t));
    socket_ptr_->connect("tcp://localhost:9090");
 
    Control::Controller::control_msg_t msg_cmd;
    if(cmd == "cu"){
        msg_cmd.command_type = Control::Controller::command_t::SET_CUMASK;
    }
    if(cmd == "power"){
        msg_cmd.command_type = Control::Controller::command_t::SET_FREQ;
    }

    msg_cmd.gpu_ind = gpu;
    msg_cmd.value1 = value1;
    msg_cmd.value2 = value2;

    zmq::message_t msg_s(&msg_cmd, sizeof(msg_cmd));
    socket_ptr_->send(msg_s, zmq::send_flags::dontwait);
    zmq::message_t msg_r;
    socket_ptr_->recv(msg_r);
    std::cout << std::string(msg_r.data<char>(), msg_r.size()) << std::endl << std::flush;

    socket_ptr_->disconnect("tcp://localhost:9090");
    delete socket_ptr_;
    ctx_.shutdown();
    ctx_.close();


    return 0;
}
