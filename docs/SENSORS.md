# Observation audit: what the deployable policy may use

A deployable controller may use only the sensors the real Indego provides. This is what indego_sim's `ExoSensors`
(`indego_sim/exo/sensors.py`) gives a controller, checked against the list the deployable (student) policy may use.

## Rate and latency

| | value | source in indego_sim |
|---|---|---|
| control rate | 200 Hz | `ExoParams.control_rate_hz` [doc: Indego Explorer signals at 200 Hz] |
| sensor latency | 3 ms acquisition -> controller | `SensorParams.sensor_delay` [est] |
| command latency | 2 ms controller -> driver | `MotorParams.cmd_delay` [est] |
| total loop latency | ~10-11 ms incl. sample-and-hold | indego_sim docs/DEVELOPMENT.md |

## Signals

| signal (obs key) | size | on the stock Indego | allowed for the deployable policy |
|---|---|---|---|
| joint angle `q` (hip_l, knee_l, hip_r, knee_r), 14-bit encoders | 4 | yes | joint encoders |
| joint velocity `qd`, filtered encoder difference (firmware) | 4 | yes | firmware velocity |
| motor current `current` and the current-based torque `tau_est` = K i N | 4 + 4 | yes | motor current |
| thigh IMUs `acc_thigh_l/r`, `gyr_thigh_l/r` | 12 | yes | thigh IMUs |
| hip-unit IMU `acc_hip`, `gyr_hip`, orientation `hip_quat` | 10 | yes | hip IMU |
| trunk IMU `acc_trunk`, `gyr_trunk`, `trunk_quat` | 10 | no - add-on (`SensorParams.trunk_imu`) | optional trunk IMU |
| foot switches `foot_contact` | 2 | no - add-on (`SensorParams.foot_switches`) | not in the list: not used |
| crutch force | - | no | optional crutch force: **not modelled as a sensor yet** (the sim knows the crutch loads; add an instrumented-crutch sensor for the AIS A branch) |

`ExoSensors.flatten` gives the 34 stock values (q, qd, tau_est, the three IMUs, hip_quat) + 10 with the trunk IMU;
`current` is in the observation dict (tau_est is derived from it). Noise, bias, quantisation and latency are applied
in the sensor model and are randomised in training.

## Privileged (teacher-only) state

Available from the sim, never to the student: per-muscle fatigue (`indego_aan.deficits.Fatigue3CCr.MF`), the
volitional drive (`Deficits.drive_now`), the human's joint torques from muscle forces (`indego_aan.rollout.Recorder`),
gait phase (the walking controller's swing/stance state), ground reaction forces, COM.
