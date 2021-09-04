#ifndef DAISYKIT_UTILS_LOGGING_MJPEH_SERVER_H_
#define DAISYKIT_UTILS_LOGGING_MJPEH_SERVER_H_

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace daisykit {
namespace utils {
namespace logging {

typedef unsigned short Port;
typedef int Socket;
typedef struct hostent HostEnt;
typedef struct sockaddr SockAddr;
typedef struct sockaddr_in SockAddrIn;
typedef unsigned int* AddrPtr;
static const int kInvalidSocket = -1;
static const int kSocketError = -1;
static const int kTimeout = 200000;
static const int kNumConnections = 10;

struct clientFrame {
  uchar* outbuf;
  int outlen;
  int client;
};

struct clientPayload {
  void* context;
  clientFrame cf;
};

class MJPEGServer {
  Socket sock;
  fd_set master;
  int timeout;
  int quality;  // jpeg compression [1..100]
  std::vector<int> clients;
  pthread_t thread_listen, thread_write;
  pthread_mutex_t mutex_client = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t mutex_cout = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t mutex_writer = PTHREAD_MUTEX_INITIALIZER;
  cv::Mat lastFrame;
  int port;

  int _write(int sock, char* s, int len);
  int _read(int socket, char* buffer);
  static void* listen_Helper(void* context);
  static void* writer_Helper(void* context);
  static void* clientWrite_Helper(void* payload);

 public:
  MJPEGServer(int port = 0);
  ~MJPEGServer();
  bool release();
  bool open();
  bool isOpened();
  void start();
  void stop();
  void write(cv::Mat frame);

 private:
  void Listener();
  void Writer();
  void ClientWrite(clientFrame& cf);
};

}  // namespace logging
}  // namespace utils
}  // namespace daisykit

#endif