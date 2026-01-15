BINARY = output
FILE_OBJECTS = main.o fileLoader.o
BOOST = -lboost_system -lboost_filesystem
GCC = g++ -std=c++17
FLAGS = -Wall -pedantic -Wextra

build: $(FILE_OBJECTS)
    $(GCC) $(FLAGS) $(FILE_OBJECTS) -o $(BINARY) $(BOOST)

main.o: main.cpp fileLoader.o
    $(GCC) $(FLAGS) -c main.cpp

fileLoader.o: fileLoader.cpp
    $(GCC) $(FLAGS) -c fileLoader.cpp

clean:
    rm -rf *.o $(BINARY)
