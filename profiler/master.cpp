#include <zmq.h>
#include <zmq_addon.hpp>

namespace Control
{
    
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

    std::ostream& operator<<(std::ostream& os, enum Control::command_t const& c);

    std::ostream& operator<<(std::ostream& os, struct Control::control_msg_t const& m);

} // namespace Control

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

        
    Control::control_msg_t msg_cmd;
    if(cmd == "cu"){
        msg_cmd.command_type = Control::command_t::SET_CUMASK;
    }
    if(cmd == "power"){
        msg_cmd.command_type = Control::command_t::SET_FREQ;
    }
    msg_cmd.gpu_ind = 0U;
    msg_cmd.value=cap;

    zmq::message_t msg_s(&msg_cmd, sizeof(msg_cmd));
    socket_ptr_->send(msg_s, zmq::send_flags::dontwait);
    zmq::message_t msg_r;
    socket_ptr_->recv(msg_r);
    std::string reply(msg_r.data<char>(), msg_r.size());

    socket_ptr_->disconnect("tcp://localhost:9090");
    delete socket_ptr_;
    socket_ptr_ = nullptr;
    ctx_.shutdown();
    ctx_.close();

    return 0;        
}
