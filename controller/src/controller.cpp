#include "controller.hpp"

#if DEBUG_MODE
#define PRINT(prefix, msg) std::cout << prefix << ":"<< std::string(msg) << std::endl << std::flush 
#endif


namespace Control{
    std::ostream& operator<<(std::ostream& os, enum Control::Controller::command_t const& c){ 
        std::string cmd = "";
        switch (c){
        case Control::Controller::command_t::NOOP :
            /* code */
            cmd = "NOOP";
            break;
        case Control::Controller::command_t::SET_FREQ :
            /* code */
            cmd = "SET_FREQ";
            break;
        case Control::Controller::command_t::RESET_FREQ :
            /* code */
            cmd = "RESET_FREQ";
            break;
        case Control::Controller::command_t::SET_CUMASK :
            /* code */
            cmd = "SET_CUMASK";
            break;
        case Control::Controller::command_t::RESET_CUMASK :
            /* code */
            cmd = "RESET_CUMASK";
            break;
        default:
            cmd = "UNSUPPORTED";
            break;
        }
        return os << cmd;
    }
    std::ostream& operator<<(std::ostream& os, struct Control::Controller::control_msg_t const& m) { 
        return os << "Control Message{"
                << "cmd:"             << m.command_type   << " "
                << "GPU:"             << m.gpu_ind        << " "
                << "Value:"           << m.value          << " }";
    }
    // initializes the controller (order important)
    bool Controller::init_controller(){
        if(init_rocm_smi() == false){ // read number of gpus from rocm_smi
            return false;
        }
        if(init_shm() == false){ // init shm for ROCR IPC
            return false;
        }
        if(init_zmq() == false){ // init controller frontend to receive control commands
            return false;
        }

        return true;
    }
    bool Controller::init_rocm_smi(){
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::init_rocm_smi", "initting rocm_smi");
        #endif
        if(smi_initted_ == true){
            return true;
        }
        // init rocm_smi
        rsmi_status_t st = rsmi_init(0);
        if(st != rsmi_status_t::RSMI_STATUS_SUCCESS){
            std::cerr << "Initting rsmi failed\n" << std::flush;
            return false;
        }
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::init_rocm_smi", "initting rocm_smi DONE");
        #endif
        // Read number of gpus 
        st = rsmi_num_monitor_devices(&num_gpus_);
        if(st != rsmi_status_t::RSMI_STATUS_SUCCESS){
            std::cerr << "reading number of GPUs from rsmi failed\n" << std::flush;
            return false;
        }
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::init_rocm_smi", "Read number of gpus: "+std::to_string(num_gpus_));
        #endif
        st = rsmi_dev_power_cap_range_get(0, 0, &gpu_max_power_, &gpu_min_power_);
        if(st != rsmi_status_t::RSMI_STATUS_SUCCESS){
            std::cerr << "reading min/max power of GPUs from rsmi failed\n" << std::flush;
            return false;
        }
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::init_rocm_smi", "Read power range: min("+std::to_string(gpu_min_power_)+") max("+std::to_string(gpu_max_power_)+")");
        #endif
        smi_initted_ = true;
        return true;
    }
    // deinitializes shm if it is initialized already
    bool Controller::init_shm(){
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::init_shm", "initting shm");
        #endif
        if(shm_initted_ == true){
            assert(shm_ != nullptr && "FATAL: shm is initialized but shm_ pointer is null\n");
            return true;
        }
        shm_ = cumasking_shm_init(num_gpus_);
        if(shm_ == nullptr){
            std::cout << "Controller failed to init SHM\n" << std::flush;
            return false;
        }
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::init_shm", "shm initted with "+std::to_string(num_gpus_)+" gpu slots");
        #endif
        shm_initted_ = true;
        return true;
    }
    // deinitializes shm if it is initialized already
    bool Controller::init_zmq(){
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::init_zmq", "initting zmq");
        #endif
        if(zmq_initted_){
            assert(socket_ptr_ != nullptr && "FATAL: zmq is initialized but socket_ptr_ pointer is null\n");
            return true;
        }

        try{
  	        socket_ptr_ = new zmq::socket_t(ctx_, zmq::socket_type::router);
            uint32_t id_int = 112; //TODO
            std::string id(std::to_string(id_int));
            // std::string id("power");
            zmq_setsockopt(*socket_ptr_, ZMQ_SUBSCRIBE, id.c_str(), id.size());
            uint32_t msg_to_keep = 2;
            zmq_setsockopt(*socket_ptr_, ZMQ_CONFLATE, &msg_to_keep, sizeof(uint32_t));
            socket_ptr_->bind(control_socket_address_);
        }
        catch(...){
            std::cout << "Controller failed to init zmq\n" << std::flush;
        }
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::init_zmq", "initting zmq DONE");
        #endif
        zmq_initted_ = true;
        return true;
    }
    // deinitializes shm(for ROCR<->Controller IPC), rsmi, and controller socket
    void Controller::deinit_controller(){
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::deinit_controller", "deinit_controller");
        #endif
        deinit_zmq();
        deinit_shm();
        deinit_rocm_smi();
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::deinit_controller", "deinit_controller DONE");
        #endif
    }
    void Controller::deinit_shm(){
        if(shm_initted_ == false)
            return;
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::deinit_shm", "deinit_shm");
        #endif
        if(cumasking_shm_destroy(shm_) < 0){
            std::cout << "Controller failed to deinit SHM\n" << std::flush;
        }
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::deinit_shm", "deinit_shm DONE");
        #endif
        shm_initted_ = false;
        return;
    }
    void Controller::deinit_zmq(){
        if(zmq_initted_ == false)
            return;
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::deinit_zmq", "deinit_zmq");
        #endif
        assert(socket_ptr_ != nullptr && "FATAL: zmq is initialized but socket_ptr_ pointer is null\n");
        socket_ptr_->close();
        ctx_.shutdown();
        ctx_.close();
        delete socket_ptr_;
        socket_ptr_ = nullptr;
        zmq_initted_ = false;
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::deinit_zmq", "deinit_zmq DONE");
        #endif
        return;
    } 
    void Controller::deinit_rocm_smi(){
        if(smi_initted_ == false){
            return;
        }
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::deinit_rocm_smi", "deinit_rocm_smi");
        #endif
        rsmi_shut_down(); // shutdown rocm-smi
        smi_initted_ = false;
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::deinit_rocm_smi", "deinit_rocm_smi DONE");
        #endif
    }
    // initializes and runs the controller thread
    bool Controller::start(bool detached){

        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::start", "startting");
        #endif
        if(init_controller() == false){ // init failed
            return false;
        }
        
        running_ = true;
        if(detached)
            thread_ = new std::thread(&Control::Controller::run, this);
        else{
            run();
        }
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::start", "startting DONE");
        #endif
        return true;
    }
    // Stops the controller thread 
    void Controller::stop() {
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::stop", "stopping");
        #endif

        if(!running_) // not running
            return;
        running_ = false;

        if(thread_ != nullptr){
            if(thread_->joinable()){
                thread_->join();
            }
            delete thread_;
            thread_ = nullptr;
        }

        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::stop", "stopping DONE");
        #endif
    }
    // Control thread
    void Controller::run(){
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::run", "run");
        #endif
        std::cout << "Controller running\n" << std::flush;
        if(thread_ != nullptr)
            pthread_setname_np(thread_->native_handle(), "Controller");
        zmq::pollitem_t items[] = {
            { *socket_ptr_, 0, ZMQ_POLLIN, 0 }
        };  
        const std::chrono::milliseconds timeout{50U};
        while(running_){
            zmq::poll(&items[0], 1, timeout);
            // event on workers (reply received)?
            if (items[0].revents & ZMQ_POLLIN) {
                zmq::message_t msg = get_command();
                apply_command(msg);
            }
        }
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::run", "stopping@run");
        #endif
        // when we get here, this thread has been already stopped
        deinit_controller();
        std::cout << "Controller stopped\n" << std::flush;
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::run", "stopped@run");
        #endif
    }
    // Reads the command(zmq_msg) data from socket
    zmq::message_t Controller::get_command(){
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::get_command", "getting command");
        #endif
        std::string rep_str("success");
        zmq::message_t reply(rep_str.c_str(), rep_str.size());
        zmq::message_t id;
        zmq::message_t msg;
        zmq::recv_result_t rcv_result = socket_ptr_->recv(id);
        assert(rcv_result && "Controller: getting id from socket rcv failed\n");
        rcv_result = socket_ptr_->recv(msg);
        assert(rcv_result && "Controller: getting msg from socket rcv failed\n");
        socket_ptr_->send(id, zmq::send_flags::sndmore);
        socket_ptr_->send(reply, zmq::send_flags::dontwait);
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::get_command", "GOT command");
        #endif
        return std::move(msg);
    }
    // Excution of any recived controll msg happens here
    void Controller::apply_command(zmq::message_t& msg){
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::apply_command", "applying command");
        #endif
        Control::Controller::control_msg_t* control_msg_ptr = msg.data<Control::Controller::control_msg_t>();
        std::cout << get_cur_time_str() << " <CMD_RECVD> " << control_msg_ptr << std::endl << std::flush;
        switch (control_msg_ptr->command_type) {
            case Control::Controller::command_t::SET_FREQ :
                set_freq(control_msg_ptr->gpu_ind, control_msg_ptr->value);
                break;
            case Control::Controller::command_t::RESET_FREQ :
                set_freq(control_msg_ptr->gpu_ind, 225U);
                break;
            case Control::Controller::command_t::SET_CUMASK :
                set_cus(control_msg_ptr->gpu_ind, control_msg_ptr->value);
                break;
            case Control::Controller::command_t::RESET_CUMASK :
                set_cus(control_msg_ptr->gpu_ind, 60U);
                break;
            default:
                break;
        }
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::apply_command", "applying command DONE");
        #endif
    }
    // Utility function for logging
    std::string Controller::get_cur_time_str(){
        // returns current time in string
        time_t now = time(0);
        struct tm  tstruct;
        char  buf[80];
        tstruct = *localtime(&now);
        strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
        return std::string(buf);
    }
    
    // Communicate with ROCR through SHM to set cu_mask
    void Controller::set_cus(uint32_t gpu_ind, uint32_t num_cus){
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::set_cus", "set_cus: gpu("+std::to_string(gpu_ind)+") cus("+std::to_string(num_cus)+")");
        #endif
        if(num_cus < 0U || num_cus > 60U){
            return;
        }
        
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::set_cus", "setting cus");
        #endif
        

        pthread_mutex_lock(shm_->mutex_ptr);
        *(shm_->gpus_num_cu_ptr + gpu_ind) = num_cus;
        pthread_mutex_unlock(shm_->mutex_ptr);
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::set_cus", "setting cus DONE");
        #endif
        return;
    }

    void Controller::set_freq(uint32_t gpu_ind, uint32_t freq){
        
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::set_freq", "set_freq: gpu("+std::to_string(gpu_ind)+") cap("+std::to_string(freq)+")");
        #endif

        uint64_t cap = freq * 1000000U;
        if(cap < gpu_min_power_ || cap > gpu_max_power_){
            return;
        }
        if(gpu_ind < 0 || gpu_ind >= num_gpus_){
            return;
        }
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::set_freq", "setting freq");
        #endif
        rsmi_status_t st = rsmi_dev_power_cap_set(gpu_ind, 0, cap);
        if(st != rsmi_status_t::RSMI_STATUS_SUCCESS){
            //todo log
            std::cerr << "Setting power cap with rsmi failed\n" << std::flush;
        }
        #if DEBUG_MODE
            PRINT(get_cur_time_str()+" @Controller::set_freq", "setting freq DONE");
        #endif
        return;
    }
}