#include "controller.hpp"

#include <csignal>

Control::Controller* controller_ptr = nullptr;

void signalHandler( int signum ) {
#if DEBUG_MODE
    std::cout << "@signalHandler: Interrupt signal (" << signum << ") received.\n";
#endif
    // cleanup and close up stuff here  
    // terminate program  
    if(controller_ptr )
        controller_ptr->stop();
    delete controller_ptr;
    controller_ptr = nullptr;
    std::exit(signum);  
}

int main(int argc, char *argv[]){
    std::string port("9090");
    if(argc > 1){
        port = std::string(argv[1]);
    }
    // CONTROL_PORT is set in Dockerfile args, then passed to cmake (see ../../Dockerfile)
    signal(SIGINT, signalHandler);  
    signal(SIGTERM, signalHandler);
#if DEBUG_MODE
    std::cout << "@main: port:" << port << std::endl << std::flush;
#endif
    Control::Controller controller(port);
    if(controller.start(false) == false){
        std::cerr << "@main: Controller failed to start\n" << std::flush;
    }
    return 0;
}