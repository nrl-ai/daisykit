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

#include "daisykitsdk/common/logging/mjpeg_server.h"

#include <fstream>

namespace daisykit {
namespace logging {

int MJPEGServer::SockWrite(int sock, char* s, int len) {
  if (len < 1) {
    len = strlen(s);
  }
  int retval = ::send(sock, s, len, 0x4000);
  return retval;
}

int MJPEGServer::SockRead(int socket, char* buffer) {
  int result;
  result = recv(socket, buffer, 4096, MSG_PEEK);
  if (result < 0) {
    std::cerr << "An exception occurred. Exception Nr. " << result << std::endl;
    return result;
  }
  std::string s = buffer;
  buffer = (char*)s.substr(0, (int)result).c_str();
  return result;
}

void* MJPEGServer::ListenHelper(void* context) {
  ((MJPEGServer*)context)->Listener();
  return NULL;
}

void* MJPEGServer::WriteHelper(void* context) {
  ((MJPEGServer*)context)->Writer();
  return NULL;
}

void* MJPEGServer::ClientWriteHelper(void* payload) {
  void* ctx = ((clientPayload*)payload)->context;
  struct clientFrame cf = ((clientPayload*)payload)->cf;
  ((MJPEGServer*)ctx)->ClientWrite(cf);
  return NULL;
}

MJPEGServer::MJPEGServer(int port)
    : sock_(kInvalidSocket), timeout_(kTimeout), quality_(90), port_(port) {
  signal(SIGPIPE, SIG_IGN);
  FD_ZERO(&master_);
}

MJPEGServer::~MJPEGServer() { Release(); }

bool MJPEGServer::Release() {
  if (sock_ != kInvalidSocket) shutdown(sock_, 2);
  sock_ = (kInvalidSocket);
  return false;
}

bool MJPEGServer::Open() {
  sock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

  SockAddrIn address;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_family = AF_INET;
  address.sin_port = htons(port_);
  if (::bind(sock_, (SockAddr*)&address, sizeof(SockAddr)) == kSocketError) {
    std::cerr << "Error : couldn't bind sock " << sock_ << " to port " << port_
              << std::endl;
    return Release();
  }
  if (listen(sock_, kNumConnections) == kSocketError) {
    std::cerr << "Error : couldn't bind sock " << sock_ << " to port " << port_
              << std::endl;
    return Release();
  }
  FD_SET(sock_, &master_);
  return true;
}

bool MJPEGServer::IsOpened() { return sock_ != kInvalidSocket; }

void MJPEGServer::Start() {
  pthread_mutex_lock(&mutex_writer_);
  pthread_create(&thread_listen_, NULL, this->ListenHelper, this);
  pthread_create(&thread_write_, NULL, this->WriteHelper, this);
}

void MJPEGServer::Stop() {
  this->Release();
  pthread_join(thread_listen_, NULL);
  pthread_join(thread_write_, NULL);
}

void MJPEGServer::WriteFrame(cv::Mat frame) {
  pthread_mutex_lock(&mutex_writer_);
  if (!frame.empty()) {
    last_frame_.release();
    last_frame_ = frame.clone();
  }
  pthread_mutex_unlock(&mutex_writer_);
}

void MJPEGServer::Listener() {
  // send http header
  std::string header;
  header += "HTTP/1.0 200 OK\r\n";
  header += "Cache-Control: no-cache\r\n";
  header += "Pragma: no-cache\r\n";
  header += "Connection: close\r\n";
  header +=
      "Content-Type: multipart/x-mixed-replace; boundary=mjpegstream\r\n\r\n";
  const int header_size = header.size();
  char* header_data = (char*)header.data();
  fd_set rread;
  Socket maxfd;
  this->Open();
  pthread_mutex_unlock(&mutex_writer_);
  while (true) {
    rread = master_;

    struct timeval to = {0, timeout_};
    maxfd = sock_ + 1;
    if (sock_ == kInvalidSocket) {
      return;
    }
    int sel = select(maxfd, &rread, NULL, NULL, &to);
    if (sel > 0) {
      for (int s = 0; s < maxfd; s++) {
        if (FD_ISSET(s, &rread) && s == sock_) {
          int addrlen = sizeof(SockAddr);
          SockAddrIn address = {0};
          Socket client =
              accept(sock_, (SockAddr*)&address, (socklen_t*)&addrlen);
          if (client == kSocketError) {
            std::cout << "Error : couldn't accept connection on sock %d"
                      << std::endl;
            return;
          }
          maxfd = (maxfd > client ? maxfd : client);
          pthread_mutex_lock(&mutex_cout_);
          char headers[4096] = "\0";
          int readBytes = SockRead(client, headers);
          std::cout << headers;
          pthread_mutex_unlock(&mutex_cout_);
          pthread_mutex_lock(&mutex_client_);
          SockWrite(client, header_data, header_size);
          clients_.push_back(client);
          pthread_mutex_unlock(&mutex_client_);
        }
      }
    }
    usleep(1000);
  }
}

void MJPEGServer::Writer() {
  pthread_mutex_lock(&mutex_writer_);
  pthread_mutex_unlock(&mutex_writer_);
  const int milis2wait = 16666;
  while (this->IsOpened()) {
    pthread_mutex_lock(&mutex_client_);
    int num_connected_clients = clients_.size();
    pthread_mutex_unlock(&mutex_client_);
    if (!num_connected_clients) {
      usleep(milis2wait);
      continue;
    }
    pthread_t threads[kNumConnections];
    int count = 0;

    std::vector<uchar> outbuf;
    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(quality_);
    pthread_mutex_lock(&mutex_writer_);
    imencode(".jpg", last_frame_, outbuf, params);
    pthread_mutex_unlock(&mutex_writer_);
    int outlen = outbuf.size();

    pthread_mutex_lock(&mutex_client_);
    std::vector<int>::iterator begin = clients_.begin();
    std::vector<int>::iterator end = clients_.end();
    pthread_mutex_unlock(&mutex_client_);
    std::vector<clientPayload*> payloads;
    for (std::vector<int>::iterator it = begin; it != end; ++it, ++count) {
      if (count > kNumConnections) break;
      struct clientPayload* cp =
          new clientPayload({(MJPEGServer*)this, {outbuf.data(), outlen, *it}});
      payloads.push_back(cp);
      pthread_create(&threads[count], NULL, &MJPEGServer::ClientWriteHelper,
                     cp);
    }
    for (; count > 0; count--) {
      pthread_join(threads[count - 1], NULL);
      delete payloads.at(count - 1);
    }
    usleep(milis2wait);
  }
}

void MJPEGServer::ClientWrite(clientFrame& cf) {
  std::stringstream head;
  head << "--mjpegstream\r\nContent-Type: image/jpeg\r\nContent-Length: "
       << cf.outlen << "\r\n\r\n";
  std::string string_head = head.str();
  pthread_mutex_lock(&mutex_client_);
  SockWrite(cf.client, (char*)string_head.c_str(), string_head.size());
  int n = SockWrite(cf.client, (char*)(cf.outbuf), cf.outlen);
  if (n < cf.outlen) {
    std::vector<int>::iterator it;
    it = find(clients_.begin(), clients_.end(), cf.client);
    if (it != clients_.end()) {
      std::cout << "Kill client " << cf.client << std::endl;
      clients_.erase(std::remove(clients_.begin(), clients_.end(), cf.client));
      ::shutdown(cf.client, 2);
    }
  }
  pthread_mutex_unlock(&mutex_client_);
  pthread_exit(NULL);
}

}  // namespace logging
}  // namespace daisykit