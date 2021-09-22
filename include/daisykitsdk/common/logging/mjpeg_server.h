// Copyright 2021 The DaisyKit Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DAISYKIT_COMMON_LOGGING_MJPEH_SERVER_H_
#define DAISYKIT_COMMON_LOGGING_MJPEH_SERVER_H_

#include <netinet/in.h>
#include <pthread.h>
#include <sys/signal.h>
#include <sys/socket.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

namespace daisykit {
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
}  // namespace daisykit

#endif