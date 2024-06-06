#ifndef CLOCK__H
#define CLOCK__H

class Clock {
public:
  Clock() {
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
  }

  void start() { cudaEventRecord(event_start); }

  float stop() {
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time;
    cudaEventElapsedTime(&time, event_start, event_stop);
    return time;
  }

private:
  cudaEvent_t event_start, event_stop;
};

#endif
