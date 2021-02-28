
# SOURCE = TestTF2Mnist.cc
SOURCE = TestTF2SSD.cc


OBJS = inference_test.o
TARGET = inference_test

OBJS_DEBUG = inference_test.o
TARGET_DEBUG = inference_test_debug

# compile and lib parameter
CC      = g++

LIBS    += -lmy_tf_inference -lstdc++ 
LIBS    += -lopencv_world 


    
STATIC_LIBS = 
LDFLAGS +=  -L./lib
INCLUDE += -I ./ -I ./include -I ./include/opencv
CFLAGS  += -Wall -O -fPIC  
CXXFLAGS += -std=gnu++11

all: $(TARGET) $(TARGET_DEBUG)

# link
$(TARGET):$(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS) $(STATIC_LIBS) $(CFLAGS) $(INCLUDE) $(CXXFLAGS)
	
$(TARGET_DEBUG):$(OBJS_DEBUG)
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS) $(STATIC_LIBS) $(CFLAGS) $(INCLUDE) $(CXXFLAGS) -g

#compile
$(OBJS):$(SOURCE)
	$(CC) $(CFLAGS) $(INCLUDE)  $(CXXFLAGS) -o $@ -c $^ 
	
$(OBJS_DEBUG):$(SOURCE)
	$(CC) $(CFLAGS) $(INCLUDE)  $(CXXFLAGS) -g -o $@ -c $^

# clean
clean:
	rm -fr *.o
	rm -fr $(TARGET) $(TARGET_DEBUG) 

