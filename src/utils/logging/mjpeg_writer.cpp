#include <daisykitsdk/utils/logging/mjpeg_server.h>
#include <fstream>

using namespace std;
using namespace daisykit::utils::logging;

int MJPEGServer::_write(int sock, char* s, int len) {
  if (len < 1) {
    len = strlen(s);
  }
  int retval = ::send(sock, s, len, 0x4000);
  return retval;
}

int MJPEGServer::_read(int socket, char* buffer) {
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

void* MJPEGServer::listen_Helper(void* context) {
  ((MJPEGServer*)context)->Listener();
  return NULL;
}

void* MJPEGServer::writer_Helper(void* context) {
  ((MJPEGServer*)context)->Writer();
  return NULL;
}

void* MJPEGServer::clientWrite_Helper(void* payload) {
  void* ctx = ((clientPayload*)payload)->context;
  struct clientFrame cf = ((clientPayload*)payload)->cf;
  ((MJPEGServer*)ctx)->ClientWrite(cf);
  return NULL;
}

MJPEGServer::MJPEGServer(int port)
    : sock(kInvalidSocket), timeout(kTimeout), quality(90), port(port) {
  signal(SIGPIPE, SIG_IGN);
  FD_ZERO(&master);
}

MJPEGServer::~MJPEGServer() { release(); }

bool MJPEGServer::release() {
  if (sock != kInvalidSocket) shutdown(sock, 2);
  sock = (kInvalidSocket);
  return false;
}

bool MJPEGServer::open() {
  sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

  SockAddrIn address;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  if (::bind(sock, (SockAddr*)&address, sizeof(SockAddr)) == kSocketError) {
    std::cerr << "Error : couldn't bind sock " << sock << " to port " << port
              << std::endl;
    return release();
  }
  if (listen(sock, kNumConnections) == kSocketError) {
    std::cerr << "Error : couldn't bind sock " << sock << " to port " << port
              << std::endl;
    return release();
  }
  FD_SET(sock, &master);
  return true;
}

bool MJPEGServer::isOpened() { return sock != kInvalidSocket; }

void MJPEGServer::start() {
  pthread_mutex_lock(&mutex_writer);
  pthread_create(&thread_listen, NULL, this->listen_Helper, this);
  pthread_create(&thread_write, NULL, this->writer_Helper, this);
}

void MJPEGServer::stop() {
  this->release();
  pthread_join(thread_listen, NULL);
  pthread_join(thread_write, NULL);
}

void MJPEGServer::write(cv::Mat frame) {
  pthread_mutex_lock(&mutex_writer);
  if (!frame.empty()) {
    lastFrame.release();
    lastFrame = frame.clone();
  }
  pthread_mutex_unlock(&mutex_writer);
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
  this->open();
  pthread_mutex_unlock(&mutex_writer);
  while (true) {
    rread = master;

    struct timeval to = {0, timeout};
    maxfd = sock + 1;
    if (sock == kInvalidSocket) {
      return;
    }
    int sel = select(maxfd, &rread, NULL, NULL, &to);
    if (sel > 0) {
      for (int s = 0; s < maxfd; s++) {
        if (FD_ISSET(s, &rread) && s == sock) {
          int addrlen = sizeof(SockAddr);
          SockAddrIn address = {0};
          Socket client =
              accept(sock, (SockAddr*)&address, (socklen_t*)&addrlen);
          if (client == kSocketError) {
            std::cout << "Error : couldn't accept connection on sock %d"
                      << std::endl;
            return;
          }
          maxfd = (maxfd > client ? maxfd : client);
          pthread_mutex_lock(&mutex_cout);
          char headers[4096] = "\0";
          int readBytes = _read(client, headers);
          cout << headers;
          pthread_mutex_unlock(&mutex_cout);
          pthread_mutex_lock(&mutex_client);
          _write(client, header_data, header_size);
          clients.push_back(client);
          pthread_mutex_unlock(&mutex_client);
        }
      }
    }
    usleep(1000);
  }
}

void MJPEGServer::Writer() {
  pthread_mutex_lock(&mutex_writer);
  pthread_mutex_unlock(&mutex_writer);
  const int milis2wait = 16666;
  while (this->isOpened()) {
    pthread_mutex_lock(&mutex_client);
    int num_connected_clients = clients.size();
    pthread_mutex_unlock(&mutex_client);
    if (!num_connected_clients) {
      usleep(milis2wait);
      continue;
    }
    pthread_t threads[kNumConnections];
    int count = 0;

    std::vector<uchar> outbuf;
    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(quality);
    pthread_mutex_lock(&mutex_writer);
    imencode(".jpg", lastFrame, outbuf, params);
    pthread_mutex_unlock(&mutex_writer);
    int outlen = outbuf.size();

    pthread_mutex_lock(&mutex_client);
    std::vector<int>::iterator begin = clients.begin();
    std::vector<int>::iterator end = clients.end();
    pthread_mutex_unlock(&mutex_client);
    std::vector<clientPayload*> payloads;
    for (std::vector<int>::iterator it = begin; it != end; ++it, ++count) {
      if (count > kNumConnections) break;
      struct clientPayload* cp =
          new clientPayload({(MJPEGServer*)this, {outbuf.data(), outlen, *it}});
      payloads.push_back(cp);
      pthread_create(&threads[count], NULL, &MJPEGServer::clientWrite_Helper,
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
  stringstream head;
  head << "--mjpegstream\r\nContent-Type: image/jpeg\r\nContent-Length: "
       << cf.outlen << "\r\n\r\n";
  string string_head = head.str();
  pthread_mutex_lock(&mutex_client);
  _write(cf.client, (char*)string_head.c_str(), string_head.size());
  int n = _write(cf.client, (char*)(cf.outbuf), cf.outlen);
  if (n < cf.outlen) {
    std::vector<int>::iterator it;
    it = find(clients.begin(), clients.end(), cf.client);
    if (it != clients.end()) {
      std::cout << "Kill client " << cf.client << std::endl;
      clients.erase(std::remove(clients.begin(), clients.end(), cf.client));
      ::shutdown(cf.client, 2);
    }
  }
  pthread_mutex_unlock(&mutex_client);
  pthread_exit(NULL);
}
