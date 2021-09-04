#include <daisykitsdk/utils/logging/mjpeg_writer.h>
#include <fstream>

void MJPEGWriter::Listener() {
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
  SOCKET maxfd;
  this->open();
  pthread_mutex_unlock(&mutex_writer);
  while (true) {
    rread = master;

    struct timeval to = {0, timeout};
    maxfd = sock + 1;
    if (sock == INVALID_SOCKET) {
      return;
    }
    int sel = select(maxfd, &rread, NULL, NULL, &to);
    if (sel > 0) {
      for (int s = 0; s < maxfd; s++) {
        if (FD_ISSET(s, &rread) && s == sock) {
          int addrlen = sizeof(SOCKADDR);
          SOCKADDR_IN address = {0};
          SOCKET client =
              accept(sock, (SOCKADDR*)&address, (socklen_t*)&addrlen);
          if (client == SOCKET_ERROR) {
            std::cout << "Error : couldn't accept connection on sock %d" << std::endl;
            return;
          }
          maxfd = (maxfd > client ? maxfd : client);
          pthread_mutex_lock(&mutex_cout);
          cout << "new client " << client << endl;
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

void MJPEGWriter::Writer() {
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
    pthread_t threads[NUM_CONNECTIONS];
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
      if (count > NUM_CONNECTIONS) break;
      struct clientPayload* cp =
          new clientPayload({(MJPEGWriter*)this, {outbuf.data(), outlen, *it}});
      payloads.push_back(cp);
      pthread_create(&threads[count], NULL, &MJPEGWriter::clientWrite_Helper,
                     cp);
    }
    for (; count > 0; count--) {
      pthread_join(threads[count - 1], NULL);
      delete payloads.at(count - 1);
    }
    usleep(milis2wait);
  }
}

void MJPEGWriter::ClientWrite(clientFrame& cf) {
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
