
g++ $(pkg-config --cflags jsoncpp eigen3 --libs opencv4) -std=c++11 ./*.cpp -o SFRout  -I /opt/homebrew/include/  /Users/thomasvo/eclipse-workspace/SFRcpp/kiss_fft130/kiss_fft.c
