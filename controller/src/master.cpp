#include <zmq.h>
#include <zmq_addon.hpp>

#include "controller.hpp"

int main(int argc, char *argv[]){
    
    if(argc < 2){
        return 1;
    }
    std::string cmd(argv[1]);
    
    uint32_t cap = std::stoi(argv[2]);
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
        msg_cmd.gpu_ind = 0U;
        msg_cmd.value=cap;

        zmq::message_t msg_s(&msg_cmd, sizeof(msg_cmd));
        socket_ptr_->send(msg_s, zmq::send_flags::dontwait);
        zmq::message_t msg_r;
        socket_ptr_->recv(msg_r);
        std::cout << std::string(msg_r.data<char>(), msg_r.size()) << std::endl << std::flush;

    socket_ptr_->disconnect("tcp://localhost:9090");
    ctx_.shutdown();
    ctx_.close();


    return 0;        
}
