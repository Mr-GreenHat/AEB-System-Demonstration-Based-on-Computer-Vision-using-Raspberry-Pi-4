# Perception Code Outline for Braking

This document describes the high-level structure and components of a perception module used to support a braking system. The actual implementation code will be added later; consider this a roadmap.

## Goals

1. Accept sensor inputs (camera, lidar, radar, etc.)
2. Detect relevant objects (vehicles, pedestrians, obstacles)
3. Estimate distances, velocities, and predict trajectories
4. Flag braking events when collision is imminent
5. Provide interfaces for braking controller

## Suggested Folder Structure

```
src/
  perception/
    sensors.py          # interfaces for sensor data ingestion
    detection.py        # object detection logic (e.g. neural network wrappers)
    tracking.py         # tracking and state estimation
    fusion.py           # sensor fusion algorithms
    decision.py         # determines when braking should be triggered
    utils.py            # helper functions, math, coordinate transforms
    tests/              # unit tests for perception components

  braking/
    controller.py       # braking control algorithms
    models.py           # braking dynamics models
    utils.py            # helper functions for braking
    tests/              # unit tests for braking components

  ipm/
    input_pipeline.py   # code to ingest and preprocess IPM data
    config.py           # configuration for IPM input
    tests/              # unit tests for IPM components
```

## Calibration Images

The `calibration_images` folder has been moved to `src/data/` to centralize shared resources. Refer to the `src/data/outline.md` for details.

## Workflow Outline

1. **Input IPM code**: Place the IPM data ingestion logic under `src/ipm/`.
2. **Sensor layer**: Create classes or functions to read raw sensor streams.
3. **Detection & Tracking**: Use ML models or classical algorithms to identify obstacles.
4. **Fusion & Decision**: Combine data streams and evaluate if braking is necessary.
5. **Braking interface**: Expose decision outputs to braking controller module.

> _Note_: The above is a starting point. You can expand modules as the project grows.
