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
 public:
  MJPEGServer(int port = 0);
  ~MJPEGServer();
  bool Release();
  bool Open();
  bool IsOpened();
  void Start();
  void Stop();
  void WriteFrame(cv::Mat frame);

 private:
  Socket sock_;
  fd_set master_;
  int timeout_;
  int quality_;  // jpeg compression [1..100]
  std::vector<int> clients_;
  pthread_t thread_listen_, thread_write_;
  pthread_mutex_t mutex_client_ = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t mutex_cout_ = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t mutex_writer_ = PTHREAD_MUTEX_INITIALIZER;
  cv::Mat last_frame_;
  int port_;

  int SockWrite(int sock, char* s, int len);
  int SockRead(int socket, char* buffer);
  static void* ListenHelper(void* context);
  static void* WriteHelper(void* context);
  static void* ClientWriteHelper(void* payload);
  void Listener();
  void Writer();
  void ClientWrite(clientFrame& cf);
};

}  // namespace logging
}  // namespace utils
}  // namespace daisykit

#endif