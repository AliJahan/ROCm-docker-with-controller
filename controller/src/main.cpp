#include "controller.hpp"

#include <csignal>
#include <cstdlib>

Control::Controller* controller_ptr = nullptr;

void signalHandler( int signum ) {
#if DEBUG_MODE
    std::cout << "@signalHandler: Interrupt signal (" << signum << ") received.\n";
#endif
    // cleanup and close up stuff here  
    // terminate program  
    if(controller_ptr)
        controller_ptr->stop();
    delete controller_ptr;
    controller_ptr = nullptr;
    std::exit(signum);  
}

int main(int argc, char *argv[]){
    // CONTROL_PORT is set in Dockerfile args, then passed to cmake (see ../../Dockerfile)
    signal(SIGINT, signalHandler);  
    signal(SIGTERM, signalHandler);
    std::string control_ip(std::getenv("REMOTE_IP"));
    std::string control_port(std::getenv("RESOURCE_CONTROLLER_PORT"));
    std::string app_name(std::getenv("WORKLOAD"));
#if DEBUG_MODE
    std::cout << "@main: worload: " << app_name << " remote controller address:" << control_ip << ":" << control_port << std::endl << std::flush;
#endif
    
    Control::Controller controller(control_ip, control_port, app_name);
    if(controller.start(false) == false){
        std::cerr << "@main: Controller failed to start\n" << std::flush;
    }
    return 0;
}