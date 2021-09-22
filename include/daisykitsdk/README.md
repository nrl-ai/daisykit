# Coding structure - SDK

+ **daisykitsdk**
  + **common** common helpers
    + **io** (daisykit::io): read/write files, config, network
    + **logging** (daisykit::logging): logging / profilers
    + **types** (daisykit::types): data structures and types
    + **visualizers** (daisykit::visualizers): visualizers for object detection, segmentation, keypoints...
    + **utils** (daisykit::utils): common utilities
      + timer.h
    + profiler.h (daisykit::profilers): profilers to tracking system performance
  + **processors** (daisykit::processors) signal/image processors, basic operations to build computational system.
    + **image_processors**
    + **signal_processors**
    + **trackers**
  + **models** (daisykit::models): AI models for CV, NLP or other tasks
  + **graphs** (daisykit::graphs): graph API
    + **core**: core definitions of graph API
      + packet.h
      + node.h
      + graph.h
      ...
    + **nodes** (daisykit::graphs::nodes): computational or visualizer nodes.
      + **models**
      + **image_processors**
      + **signal_processors**
      + **trackers**
  + **flows** (daisykit::flows): completed flows built on graphs, or chain of processors
  + **thirdparties**: lightweight third-party libraries.
