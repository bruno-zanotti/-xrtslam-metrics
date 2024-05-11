---
layout: post
title:  "Basalt Ablation Study"
date:   2024-05-11 16:25:28 -0300
---

## Table of Contents

1. [Overview](#overview)
2. [Basalt new features](#improvements)
3. [Results](#results)
    - [Euroc](#euroc)
    - [Monado Valve Index](#monado-valve-index)
4. [Conclusion](#conclusions)
5. [Appendix](#appendix)
    - [All datasets](#all-the-datasets)

## Overview

The main objective of this post is to measure the impact of main improvement that have been made to Basalt. Taking [Basalt from Usenko’s GitLab repository][1] as a starting point and comparing it with the main branch of [mateosss/basalt][2] (At the moment of writing this post, this is the last commit [673cc5c6][3]), these are the main features that have been implemented in chronological order.

## Improvements

| Version | Name                             | Description                                                                                                                                                                                                                                            | PR                                                                        | commit  | How to enable/disable?                                     |
| ------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- | ------- | ---------------------------------------------------------- |
| Basalt  | Basalt original                  | Original Basalt from Usenko.                                                                                                                                                                                                                           |                                                                           |         | \-                                                         |
| MGT     | Matching guess types             | Before the same pixel was being used (now called SAME_PIXEL), now we assume features in the left camera have a depth that is equal to the average of all currently known depths, and reproject that 3D point into the right camera (REPROJ_AVG_DEPTH). | [#10](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/10) | a0990d9 | *optical_flow_matching_guess_type*                           |
| OVL     | Detection non-overlapping area   | Use camera's non-overlapping area: this allows HMDs that have canted cameras to track features on the area of the right camera that does not overlap with the left (main) camera.                                                                      | [#20](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/20) | 46ef397 | *optical_flow_detection_nonoverlap*                          |
| TMC     | Multicamera                      | Triangulate landmarks using all the available cameras.                                                                                                                                                                                                 | [#20](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/20) | 46ef397 |                                                            |
| IMU     | IMU optical flow prediction      | Front-end predicts the pose using IMU information.                                                                                                                                                                                                     | [#23](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/23) | 6c46c2c |                                                            |
| ISR     | Image safe radius                | Use safe_radius property to ignore black image corners.                                                                                                                                                                                                | [#35](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/35) | 4b14885 | *optical_flow_image_safe_radius*                             |
| REC     | Recall landmarks                 | Recall previously seen landmarks with optical flow.                                                                                                                                                                                                    | [#35](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/35) | 4b14885 | *optical_flow_recall_enable*<br>*optical_flow_recall_all_cams*<br>*vio_marg_lost_landmarks* |

The other feature that will be taken into account in the study is the possibility of making co-visibility queries to a map database. This is implemented in the PR [#52](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/52), the main idea is to create a map database where to store all the landmarks and keyframes of the sequence and use that information to incorporate old co-visible keyframes to the VIO window.

In the same PR two new settings have been added to be able to enable or disable the features:

- **Multicamera:** By default True, the setting vio_triangulate_with_all_cams allow to choose if Basalt will triangulate features from all cameras.
- **IMU optical flow prediction:** By default True, the setting optical_flow_predict_with_imu allows to choose if the optical flow uses the information of the IMU to predict the new pose.
NOTE: the main objective of this new configuration is to be able to measure the impact of these features, but in a real scenario it is expected that it will always be enabled.

If we now take this into account the table of new features looks like:

| Version | Name                             | Description                                                                                                                                                                                                                                            | PR                                                                        | commit  | How to enable/disable?                                     |
| ------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- | ------- | ---------------------------------------------------------- |
| Basalt  | Basalt original                  | Original Basalt from Usenko.                                                                                                                                                                                                                           |                                                                           |         | \-                                                         |
| Current | Current Basalt w/o improvements. | Our implementation without the improvements.                                                                                                                                                                                                           |                                                                           |         | \-                                                         |
| MGT     | Matching guess types             | Before the same pixel was being used (now called SAME_PIXEL), now we assume features in the left camera have a depth that is equal to the average of all currently known depths, and reproject that 3D point into the right camera (REPROJ_AVG_DEPTH). | [#10](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/10) | a0990d9 | *optical_flow_matching_guess_type*                           |
| OVL     | Detection non-overlapping area   | Use camera's non-overlapping area: this allows HMDs that have canted cameras to track features on the area of the right camera that does not overlap with the left (main) camera.                                                                      | [#20](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/20) | 46ef397 | *optical_flow_detection_nonoverlap*                          |
| TMC     | Multicamera                      | Triangulate landmarks using all the available cameras.                                                                                                                                                                                                 | [#20](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/20) | 46ef397 | *vio_triangulate_with_all_cams*                              |
| IMU     | IMU optical flow prediction      | Front-end predicts the pose using IMU information.                                                                                                                                                                                                     | [#23](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/23) | 6c46c2c | *optical_flow_predict_with_imu*                              |
| ISR     | Image safe radius                | Use safe_radius property to ignore black image corners.                                                                                                                                                                                                | [#35](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/35) | 4b14885 | *optical_flow_image_safe_radius*                             |
| REC     | Recall landmarks                 | Recall previously seen landmarks with optical flow.                                                                                                                                                                                                    | [#35](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/35) | 4b14885 | *optical_flow_recall_enable*<br>*optical_flow_recall_all_cams*<br>*vio_marg_lost_landmarks* |
| COV     | Map covisibility queries         | Implement covisibility queries to recall lost landmarks.                                                                                                                                                                                               | [#52](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/52) |         | *vio_covisibility_query_frequency*                           |

With these new features clearly distinguished, an ablation study was made to measure how each one improves (or not) Basalt results.

## Datasets

The datasets used in this study are:

- EMH*: [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) machine hall datasets.
- EV*: [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) Vicon Room datasets.
- MIO*: [MSD (Monado SLAM Datasets)](https://huggingface.co/datasets/collabora/monado-slam-datasets) Valve index other datasets.

(*) More datasets will be added in the future.

## Basalt runs

- **basalt**: Usenko's Original Basalt.
- **curr**: Current Basalt implementation but disabling all the improvements.
- **curr_mgt**: Same as curr but enabling feature *Matching guess types* (optical_flow_matching_guess_type=REPROJ_AVG_DEPTH).
- **curr_mgt_ovl**: Same as curr_mgt but enabling feature *Detection non-overlapping area* (optical_flow_detection_nonoverlap=True).
- **curr_mgt_ovl_tmc**: Same as curr_mgt_ovl but enabling feature *Multicamera* (vio_triangulate_with_all_cams=True).
- **curr_mgt_ovl_tmc_imu**: Same as curr_mgt_ovl_tmc but enabling feature *IMU optical flow prediction* (vio_triangulate_with_all_cams=True).
- **curr_mgt_ovl_tmc_imu_isr**: Same as curr_mgt_ovl_tmc_imu but enabling feature *Image safe radius* (vio_triangulate_with_all_cams=True).
- **curr_mgt_ovl_tmc_imu_isr_rec**: Same as curr_mgt_ovl_tmc_imu_isr but enabling feature *Recall landmarks* (optical_flow_recall_enable=True, optical_flow_recall_all_cams=True, vio_marg_lost_landmarks=False).
- **curr_mgt_ovl_tmc_imu_isr_rec_cov**: Same as curr_mgt_ovl_tmc_imu_isr_rec but enabling feature *Map covisibility queries* (vio_covisibility_query_frequency=1).

(*) Except from Usenko’s basalt, all the other runs were done standing on the branch vio+mapper from the PR [#52](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/52) plus a commit to make Basalt deterministic without changing the behaviour of the system. This was to get always the same result and isolate it from hardware limitations.

## Results

All the metrics shown below were obtained with [xrtslam-metrics](https://gitlab.freedesktop.org/mateosss/xrtslam-metrics/).

### Euroc

#### Average total features

|       | basalt   | curr            | curr_mgt        | curr_mgt_ovl    | curr_mgt_ovl_tmc   | curr_mgt_ovl_tmc_imu   | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:---------|:----------------|:----------------|:----------------|:-------------------|:-----------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| EMH01 | —        | 387.92 ± 113.40 | 381.60 ± 120.89 | 385.23 ± 116.67 | 384.89 ± 116.32    | 388.21 ± 113.16        | 388.21 ± 113.16            | 399.16 ± 113.86                | **399.74 ± 112.15**                |
| EMH02 | —        | 393.32 ± 145.28 | 389.35 ± 144.13 | 390.85 ± 142.40 | 390.87 ± 142.87    | 391.77 ± 140.81        | 391.77 ± 140.81            | 401.49 ± 135.05                | **403.40 ± 136.37**                |
| EMH03 | —        | 365.46 ± 95.40  | 370.84 ± 96.33  | 370.84 ± 96.33  | 370.90 ± 96.43     | 377.47 ± 95.81         | 377.47 ± 95.81             | 382.34 ± 96.55                 | **383.72 ± 97.38**                 |
| EMH05 | —        | 340.42 ± 96.42  | 347.08 ± 96.96  | 347.08 ± 96.96  | 347.09 ± 97.00     | 347.18 ± 95.79         | 347.18 ± 95.79             | **352.31 ± 99.10**             | 351.54 ± 97.72                     |
| EV101 | —        | 381.97 ± 85.88  | 392.97 ± 88.80  | 392.99 ± 88.77  | 392.87 ± 88.91     | 395.99 ± 89.56         | 395.99 ± 89.56             | 408.42 ± 86.41                 | **410.26 ± 85.98**                 |
| EV102 | —        | 282.28 ± 64.55  | 286.87 ± 66.93  | 286.87 ± 66.93  | 286.86 ± 66.94     | 307.14 ± 69.97         | 307.14 ± 69.97             | 320.68 ± 72.43                 | **322.69 ± 72.69**                 |
| EV103 | —        | 208.51 ± 73.20  | 211.73 ± 74.37  | 211.78 ± 74.33  | 211.80 ± 74.33     | 221.81 ± 74.07         | 221.81 ± 74.07             | 238.93 ± 68.65                 | **240.21 ± 70.87**                 |
| EV201 | —        | 314.10 ± 92.93  | 315.35 ± 92.10  | 315.35 ± 92.10  | 315.25 ± 92.13     | 315.96 ± 89.64         | 315.96 ± 89.64             | 329.68 ± 90.59                 | **330.25 ± 90.99**                 |
| EV202 | —        | 248.34 ± 67.25  | 250.26 ± 67.88  | 250.26 ± 67.88  | 250.22 ± 67.92     | 259.01 ± 66.77         | 259.01 ± 66.77             | 273.30 ± 68.67                 | **274.23 ± 68.63**                 |
| [AVG] | —        | 324.70 ± 92.70  | 327.34 ± 94.27  | 327.92 ± 93.60  | 327.86 ± 93.65     | 333.84 ± 92.84         | 333.84 ± 92.84             | 345.15 ± 92.37                 | **346.23 ± 92.53**                 |

#### Average features recalled

|       | basalt   | curr        | curr_mgt    | curr_mgt_ovl   | curr_mgt_ovl_tmc   | curr_mgt_ovl_tmc_imu   | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:---------|:------------|:------------|:---------------|:-------------------|:-----------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| EMH01 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 3.01 ± 4.22                    | **3.05 ± 4.24**                    |
| EMH02 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 3.69 ± 5.17                    | **3.80 ± 5.61**                    |
| EMH03 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 4.48 ± 5.41                    | **4.63 ± 5.56**                    |
| EMH05 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 4.48 ± 3.99                    | **4.82 ± 4.16**                    |
| EV101 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 10.02 ± 6.24                   | **10.84 ± 6.80**                   |
| EV102 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 15.10 ± 10.50                  | **17.74 ± 10.96**                  |
| EV103 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | **22.49 ± 15.44**              | 22.26 ± 14.99                      |
| EV201 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 7.80 ± 4.57                    | **8.44 ± 4.92**                    |
| EV202 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 15.23 ± 13.23                  | **15.98 ± 14.19**                  |
| [AVG] | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 9.59 ± 7.64                    | **10.17 ± 7.94**                   |

#### Percentage of the features that were recalled

|       | basalt   | curr          | curr_mgt      | curr_mgt_ovl   | curr_mgt_ovl_tmc   | curr_mgt_ovl_tmc_imu   | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:---------|:--------------|:--------------|:---------------|:-------------------|:-----------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| EMH01 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 0.89% ± 1.58%                  | **0.91% ± 1.57%**                  |
| EMH02 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 1.17% ± 2.01%                  | **1.22% ± 2.25%**                  |
| EMH03 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 1.34% ± 1.85%                  | **1.38% ± 1.88%**                  |
| EMH05 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 1.38% ± 1.42%                  | **1.48% ± 1.50%**                  |
| EV101 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 2.65% ± 1.93%                  | **2.84% ± 2.05%**                  |
| EV102 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 4.80% ± 3.56%                  | **5.59% ± 3.70%**                  |
| EV103 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | **9.94% ± 6.80%**              | 9.83% ± 6.67%                      |
| EV201 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 2.61% ± 1.90%                  | **2.83% ± 2.00%**                  |
| EV202 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 6.31% ± 6.64%                  | **6.52% ± 6.83%**                  |
| [AVG] | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 3.45% ± 3.08%                  | **3.62% ± 3.16%**                  |

#### Average completion percentage [%]

|       | basalt   | curr   | curr_mgt   | curr_mgt_ovl   | curr_mgt_ovl_tmc   | curr_mgt_ovl_tmc_imu   | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:---------|:-------|:-----------|:---------------|:-------------------|:-----------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| EMH01 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| EMH02 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| EMH03 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| EMH05 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| EV101 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| EV102 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| EV103 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| EV201 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| EV202 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| [AVG] | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |

#### Absolute trajectory error (ATE) [m]

|       | basalt            | curr          | curr_mgt      | curr_mgt_ovl   | curr_mgt_ovl_tmc   | curr_mgt_ovl_tmc_imu   | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:------------------|:--------------|:--------------|:---------------|:-------------------|:-----------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| EMH01 | **0.066 ± 0.023** | 0.073 ± 0.021 | 0.069 ± 0.020 | 0.069 ± 0.020  | 0.074 ± 0.024      | 0.076 ± 0.021          | 0.076 ± 0.021              | 0.090 ± 0.026                  | 0.099 ± 0.030                      |
| EMH02 | 0.062 ± 0.031     | 0.057 ± 0.028 | 0.049 ± 0.025 | 0.049 ± 0.025  | **0.049 ± 0.027**  | 0.050 ± 0.030          | 0.050 ± 0.030              | 0.063 ± 0.023                  | 0.054 ± 0.022                      |
| EMH03 | 0.062 ± 0.019     | 0.062 ± 0.019 | 0.068 ± 0.021 | 0.068 ± 0.021  | 0.057 ± 0.019      | 0.060 ± 0.022          | 0.060 ± 0.022              | **0.056 ± 0.023**              | 0.074 ± 0.034                      |
| EMH05 | **0.145 ± 0.041** | 0.145 ± 0.041 | 0.179 ± 0.056 | 0.179 ± 0.056  | 0.164 ± 0.043      | 0.145 ± 0.043          | 0.145 ± 0.043              | 0.174 ± 0.053                  | 0.147 ± 0.040                      |
| EV101 | 0.043 ± 0.017     | 0.043 ± 0.017 | 0.043 ± 0.017 | 0.043 ± 0.017  | **0.043 ± 0.017**  | 0.043 ± 0.017          | 0.043 ± 0.017              | 0.050 ± 0.020                  | 0.050 ± 0.018                      |
| EV102 | 0.045 ± 0.013     | 0.045 ± 0.013 | 0.047 ± 0.014 | 0.047 ± 0.014  | 0.046 ± 0.013      | 0.054 ± 0.017          | 0.054 ± 0.017              | 0.048 ± 0.015                  | **0.042 ± 0.018**                  |
| EV103 | 0.053 ± 0.020     | 0.053 ± 0.020 | 0.051 ± 0.020 | 0.051 ± 0.020  | **0.050 ± 0.020**  | 0.056 ± 0.014          | 0.056 ± 0.014              | 0.082 ± 0.066                  | 0.087 ± 0.066                      |
| EV201 | 0.039 ± 0.015     | 0.039 ± 0.015 | 0.043 ± 0.017 | 0.043 ± 0.017  | 0.045 ± 0.020      | **0.036 ± 0.013**      | 0.036 ± 0.013              | 0.038 ± 0.013                  | 0.040 ± 0.015                      |
| EV202 | 0.049 ± 0.021     | 0.049 ± 0.021 | 0.046 ± 0.021 | 0.046 ± 0.021  | 0.051 ± 0.026      | **0.040 ± 0.014**      | 0.040 ± 0.014              | 0.041 ± 0.019                  | 0.042 ± 0.016                      |
| [AVG] | 0.063 ± 0.022     | 0.063 ± 0.022 | 0.066 ± 0.024 | 0.066 ± 0.024  | 0.064 ± 0.023      | **0.062 ± 0.021**      | 0.062 ± 0.021              | 0.071 ± 0.029                  | 0.071 ± 0.029                      |

#### Relative trajectory error (RTE) [m]

|       | basalt              | curr                | curr_mgt                | curr_mgt_ovl        | curr_mgt_ovl_tmc    | curr_mgt_ovl_tmc_imu    | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:--------------------|:--------------------|:------------------------|:--------------------|:--------------------|:------------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| EMH01 | 0.005266 ± 0.003094 | 0.005243 ± 0.003086 | 0.005214 ± 0.003065     | 0.005215 ± 0.003065 | 0.005138 ± 0.002977 | **0.005120 ± 0.002988** | 0.005120 ± 0.002988        | 0.005653 ± 0.003292            | 0.007173 ± 0.005223                |
| EMH02 | 0.004454 ± 0.002339 | 0.004489 ± 0.002334 | 0.004402 ± 0.002432     | 0.004401 ± 0.002433 | 0.004214 ± 0.002222 | **0.004161 ± 0.002211** | 0.004161 ± 0.002211        | 0.004820 ± 0.002642            | 0.005223 ± 0.003125                |
| EMH03 | 0.011959 ± 0.007684 | 0.011942 ± 0.007676 | 0.012003 ± 0.007755     | 0.012003 ± 0.007755 | 0.011862 ± 0.007594 | **0.011686 ± 0.007596** | 0.011686 ± 0.007596        | 0.012803 ± 0.008322            | 0.027000 ± 0.021251                |
| EMH05 | 0.010939 ± 0.006223 | 0.010974 ± 0.006224 | **0.010677 ± 0.006179** | 0.010677 ± 0.006179 | 0.010819 ± 0.006209 | 0.010726 ± 0.006291     | 0.010726 ± 0.006291        | 0.012333 ± 0.007172            | 0.016402 ± 0.011576                |
| EV101 | 0.013037 ± 0.006246 | 0.013037 ± 0.006249 | 0.013043 ± 0.006257     | 0.013043 ± 0.006257 | 0.013052 ± 0.006271 | **0.013026 ± 0.006258** | 0.013026 ± 0.006258        | 0.013232 ± 0.006344            | 0.022272 ± 0.014314                |
| EV102 | 0.011775 ± 0.005147 | 0.011768 ± 0.005138 | 0.011662 ± 0.005097     | 0.011662 ± 0.005097 | 0.011652 ± 0.005086 | **0.011124 ± 0.004775** | 0.011124 ± 0.004775        | 0.012039 ± 0.005234            | 0.031618 ± 0.020535                |
| EV103 | 0.013336 ± 0.006855 | 0.013359 ± 0.006871 | 0.013119 ± 0.006482     | 0.013115 ± 0.006465 | 0.013109 ± 0.006650 | **0.012279 ± 0.006104** | 0.012279 ± 0.006104        | 0.034668 ± 0.031187            | 0.048125 ± 0.039086                |
| EV201 | 0.003451 ± 0.001951 | 0.003461 ± 0.001967 | 0.003508 ± 0.002030     | 0.003508 ± 0.002030 | 0.003453 ± 0.001964 | **0.003428 ± 0.001972** | 0.003428 ± 0.001972        | 0.004036 ± 0.002219            | 0.005305 ± 0.003648                |
| EV202 | 0.009120 ± 0.005975 | 0.009126 ± 0.005975 | 0.009050 ± 0.005870     | 0.009050 ± 0.005870 | 0.008950 ± 0.005741 | **0.007066 ± 0.003858** | 0.007066 ± 0.003858        | 0.008517 ± 0.004789            | 0.030337 ± 0.022480                |
| [AVG] | 0.009260 ± 0.005057 | 0.009267 ± 0.005058 | 0.009187 ± 0.005019     | 0.009186 ± 0.005017 | 0.009139 ± 0.004968 | **0.008735 ± 0.004673** | 0.008735 ± 0.004673        | 0.012011 ± 0.007911            | 0.021495 ± 0.015693                |

### Monado Valve Index

#### Average total features

|       | basalt   | curr           | curr_mgt       | curr_mgt_ovl       | curr_mgt_ovl_tmc   | curr_mgt_ovl_tmc_imu   | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:---------|:---------------|:---------------|:-------------------|:-------------------|:-----------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| MIO01 | —        | 203.71 ± 38.36 | 235.04 ± 50.24 | 250.51 ± 51.63     | 250.46 ± 51.53     | **251.15 ± 51.51**     | 216.73 ± 52.50             | 235.40 ± 58.45                 | 235.52 ± 57.70                     |
| MIO02 | —        | 214.98 ± 36.91 | 242.36 ± 45.79 | 255.39 ± 46.25     | 255.41 ± 46.27     | **255.87 ± 45.53**     | 223.41 ± 47.19             | 240.61 ± 51.69                 | 241.48 ± 52.46                     |
| MIO03 | —        | 220.91 ± 36.81 | 254.15 ± 47.30 | 268.40 ± 48.41     | 268.45 ± 48.47     | **268.96 ± 48.03**     | 235.06 ± 49.37             | 256.40 ± 53.75                 | 258.10 ± 54.49                     |
| MIO04 | —        | 198.57 ± 37.69 | 222.36 ± 46.47 | 235.75 ± 47.57     | 235.80 ± 47.62     | **237.34 ± 47.53**     | 204.20 ± 49.03             | 216.02 ± 52.14                 | 216.27 ± 52.12                     |
| MIO05 | —        | 247.77 ± 49.23 | 290.83 ± 67.89 | **307.91 ± 72.13** | 307.63 ± 71.82     | 307.62 ± 71.38         | 274.76 ± 73.61             | 303.86 ± 80.44                 | 305.44 ± 80.84                     |
| MIO06 | —        | 227.14 ± 53.94 | 266.53 ± 76.01 | 281.51 ± 78.19     | 281.62 ± 78.27     | **283.90 ± 78.34**     | 250.61 ± 81.63             | 269.95 ± 88.77                 | 270.64 ± 89.05                     |
| MIO07 | —        | 227.15 ± 47.04 | 261.30 ± 65.42 | 276.92 ± 67.90     | 277.09 ± 67.84     | **277.18 ± 68.49**     | 244.82 ± 70.74             | 264.69 ± 76.92                 | 268.33 ± 78.07                     |
| MIO08 | —        | 222.81 ± 39.20 | 256.89 ± 56.23 | 269.68 ± 56.26     | 269.77 ± 56.35     | **273.36 ± 56.18**     | 241.60 ± 59.58             | 254.82 ± 61.80                 | 254.42 ± 60.18                     |
| MIO09 | —        | 254.61 ± 20.92 | 294.27 ± 25.09 | 306.35 ± 26.27     | 306.27 ± 26.24     | **306.58 ± 25.98**     | 277.35 ± 26.42             | 297.26 ± 31.74                 | 298.94 ± 33.20                     |
| MIO10 | —        | 215.71 ± 53.41 | 248.04 ± 66.84 | 259.71 ± 66.76     | 261.40 ± 69.39     | **263.42 ± 64.77**     | 229.28 ± 66.54             | 242.83 ± 70.98                 | 248.25 ± 74.66                     |
| MIO11 | —        | 208.02 ± 57.38 | 237.11 ± 70.57 | 250.04 ± 70.96     | 250.18 ± 71.15     | **254.60 ± 64.37**     | 222.23 ± 67.96             | 233.60 ± 72.57                 | 239.27 ± 76.49                     |
| [AVG] | —        | 221.94 ± 42.81 | 255.35 ± 56.17 | 269.29 ± 57.48     | 269.46 ± 57.72     | **270.91 ± 56.56**     | 238.19 ± 58.60             | 255.95 ± 63.57                 | 257.88 ± 64.48                     |

#### Average features recalled

|       | basalt   | curr        | curr_mgt    | curr_mgt_ovl   | curr_mgt_ovl_tmc   | curr_mgt_ovl_tmc_imu   | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:---------|:------------|:------------|:---------------|:-------------------|:-----------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| MIO01 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 13.01 ± 9.24                   | **13.38 ± 9.58**                   |
| MIO02 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 13.20 ± 11.42                  | **13.75 ± 11.47**                  |
| MIO03 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 12.47 ± 8.37                   | **13.25 ± 8.78**                   |
| MIO04 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 10.27 ± 8.81                   | **10.37 ± 8.70**                   |
| MIO05 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 17.92 ± 12.21                  | **19.58 ± 12.06**                  |
| MIO06 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 20.96 ± 16.39                  | **22.37 ± 16.68**                  |
| MIO07 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 13.42 ± 9.03                   | **17.60 ± 11.00**                  |
| MIO08 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 16.99 ± 11.35                  | **19.83 ± 11.54**                  |
| MIO09 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 20.68 ± 8.05                   | **23.97 ± 9.13**                   |
| MIO10 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 11.38 ± 5.71                   | **12.16 ± 6.34**                   |
| MIO11 | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 10.69 ± 6.47                   | **12.92 ± 8.11**                   |
| [AVG] | —        | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00    | 0.00 ± 0.00        | 0.00 ± 0.00            | 0.00 ± 0.00                | 14.64 ± 9.73                   | **16.29 ± 10.31**                  |

#### Percentage of the features that were recalled

|       | basalt   | curr          | curr_mgt      | curr_mgt_ovl   | curr_mgt_ovl_tmc   | curr_mgt_ovl_tmc_imu   | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:---------|:--------------|:--------------|:---------------|:-------------------|:-----------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| MIO01 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 5.57% ± 3.83%                  | **5.72% ± 3.96%**                  |
| MIO02 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 5.45% ± 4.55%                  | **5.64% ± 4.51%**                  |
| MIO03 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 4.82% ± 3.23%                  | **5.12% ± 3.35%**                  |
| MIO04 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 4.80% ± 4.30%                  | **4.86% ± 4.27%**                  |
| MIO05 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 5.59% ± 3.27%                  | **6.10% ± 3.18%**                  |
| MIO06 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 7.46% ± 5.63%                  | **7.93% ± 5.64%**                  |
| MIO07 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 4.79% ± 2.51%                  | **6.23% ± 3.08%**                  |
| MIO08 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 6.67% ± 4.31%                  | **7.69% ± 4.12%**                  |
| MIO09 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 6.97% ± 2.92%                  | **7.96% ± 3.06%**                  |
| MIO10 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 4.59% ± 2.10%                  | **4.75% ± 2.13%**                  |
| MIO11 | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 4.47% ± 2.66%                  | **5.29% ± 3.26%**                  |
| [AVG] | —        | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00%  | 0.00% ± 0.00%      | 0.00% ± 0.00%          | 0.00% ± 0.00%              | 5.56% ± 3.57%                  | **6.12% ± 3.69%**                  |

#### Average completion percentage [%]

|       | basalt   | curr   | curr_mgt   | curr_mgt_ovl   | curr_mgt_ovl_tmc   | curr_mgt_ovl_tmc_imu   | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:---------|:-------|:-----------|:---------------|:-------------------|:-----------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| MIO01 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| MIO02 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| MIO03 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| MIO04 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| MIO05 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| MIO06 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| MIO07 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| MIO08 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| MIO09 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| MIO10 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| MIO11 | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |
| [AVG] | ✓        | ✓      | ✓          | ✓              | ✓                  | ✓                      | ✓                          | ✓                              | ✓                                  |

#### Absolute trajectory error (ATE) [m]

|       | basalt            | curr              | curr_mgt      | curr_mgt_ovl   | curr_mgt_ovl_tmc   | curr_mgt_ovl_tmc_imu   | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:------------------|:------------------|:--------------|:---------------|:-------------------|:-----------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| MIO01 | 640.149 ± 421.131 | 636.739 ± 418.442 | 0.727 ± 0.494 | 0.726 ± 0.490  | 0.747 ± 0.521      | 0.626 ± 0.359          | 0.620 ± 0.354              | **0.547 ± 0.293**              | 0.550 ± 0.292                      |
| MIO02 | 70.642 ± 50.344   | 70.533 ± 50.266   | 1.342 ± 0.748 | 1.342 ± 0.748  | 1.331 ± 0.740      | 1.188 ± 0.629          | **1.182 ± 0.625**          | 1.252 ± 0.653                  | 1.226 ± 0.656                      |
| MIO03 | 0.141 ± 0.074     | 0.141 ± 0.074     | 0.085 ± 0.036 | 0.085 ± 0.036  | 0.076 ± 0.032      | 0.085 ± 0.032          | 0.093 ± 0.038              | **0.072 ± 0.028**              | 0.073 ± 0.029                      |
| MIO04 | 9.065 ± 4.503     | 9.006 ± 4.477     | 0.235 ± 0.130 | 0.234 ± 0.130  | 0.241 ± 0.130      | 0.207 ± 0.098          | 0.206 ± 0.100              | **0.146 ± 0.075**              | 0.150 ± 0.076                      |
| MIO05 | 0.076 ± 0.043     | 0.076 ± 0.043     | 0.039 ± 0.016 | 0.039 ± 0.016  | 0.038 ± 0.015      | 0.039 ± 0.016          | 0.034 ± 0.012              | 0.026 ± 0.011                  | **0.022 ± 0.010**                  |
| MIO06 | 0.137 ± 0.078     | 0.137 ± 0.078     | 0.069 ± 0.021 | 0.069 ± 0.021  | 0.073 ± 0.022      | 0.049 ± 0.019          | 0.049 ± 0.020              | **0.044 ± 0.021**              | 0.062 ± 0.027                      |
| MIO07 | 0.052 ± 0.036     | 0.051 ± 0.037     | 0.020 ± 0.008 | 0.021 ± 0.008  | 0.024 ± 0.011      | 0.020 ± 0.008          | 0.023 ± 0.011              | **0.018 ± 0.007**              | 0.024 ± 0.009                      |
| MIO08 | 0.093 ± 0.065     | 0.093 ± 0.065     | 0.066 ± 0.021 | 0.066 ± 0.021  | 0.062 ± 0.019      | 0.059 ± 0.020          | 0.057 ± 0.019              | 0.035 ± 0.012                  | **0.026 ± 0.013**                  |
| MIO09 | 0.018 ± 0.013     | 0.018 ± 0.013     | 0.006 ± 0.003 | 0.006 ± 0.003  | 0.006 ± 0.003      | 0.006 ± 0.003          | 0.006 ± 0.003              | 0.006 ± 0.003                  | **0.005 ± 0.003**                  |
| MIO10 | 0.046 ± 0.028     | 0.046 ± 0.028     | 0.026 ± 0.015 | 0.026 ± 0.015  | 0.026 ± 0.015      | 0.016 ± 0.009          | 0.015 ± 0.009              | **0.014 ± 0.007**              | 0.017 ± 0.007                      |
| MIO11 | 0.067 ± 0.034     | 0.067 ± 0.034     | 0.040 ± 0.015 | 0.039 ± 0.015  | 0.038 ± 0.015      | **0.024 ± 0.010**      | 0.024 ± 0.010              | 0.029 ± 0.011                  | 0.025 ± 0.011                      |
| [AVG] | 65.499 ± 43.304   | 65.173 ± 43.051   | 0.241 ± 0.137 | 0.241 ± 0.137  | 0.242 ± 0.138      | 0.211 ± 0.110          | 0.210 ± 0.109              | 0.199 ± 0.102                  | **0.198 ± 0.103**                  |

#### Relative trajectory error (RTE) [m]

|       | basalt              | curr                | curr_mgt            | curr_mgt_ovl        | curr_mgt_ovl_tmc    | curr_mgt_ovl_tmc_imu    | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|:------------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| MIO01 | 3.482786 ± 2.882257 | 3.455232 ± 2.857748 | 0.053218 ± 0.050661 | 0.053283 ± 0.050766 | 0.053065 ± 0.050284 | 0.037767 ± 0.035763     | 0.036980 ± 0.034926        | **0.032950 ± 0.030879**        | 0.053370 ± 0.046635                |
| MIO02 | 0.966807 ± 0.835925 | 0.965238 ± 0.834399 | 0.075515 ± 0.070433 | 0.075563 ± 0.070483 | 0.077877 ± 0.072748 | 0.064713 ± 0.060449     | **0.064532 ± 0.060300**    | 0.076237 ± 0.072524            | 0.079650 ± 0.072844                |
| MIO03 | 0.017892 ± 0.015679 | 0.017967 ± 0.015763 | 0.006246 ± 0.004214 | 0.006250 ± 0.004213 | 0.006952 ± 0.005151 | 0.006260 ± 0.004208     | **0.006075 ± 0.004005**    | 0.006790 ± 0.004552            | 0.020128 ± 0.016410                |
| MIO04 | 0.080239 ± 0.069821 | 0.079929 ± 0.069539 | 0.018737 ± 0.015821 | 0.018810 ± 0.015902 | 0.018252 ± 0.015375 | **0.013075 ± 0.010332** | 0.013089 ± 0.010349        | 0.015719 ± 0.012333            | 0.027176 ± 0.020810                |
| MIO05 | 0.009999 ± 0.009264 | 0.009902 ± 0.009163 | 0.003422 ± 0.002397 | 0.003429 ± 0.002408 | 0.003412 ± 0.002411 | 0.003217 ± 0.002214     | 0.003258 ± 0.002255        | **0.003022 ± 0.001855**        | 0.012838 ± 0.009763                |
| MIO06 | 0.022290 ± 0.019768 | 0.022516 ± 0.020016 | 0.011450 ± 0.009588 | 0.011439 ± 0.009566 | 0.011570 ± 0.009717 | 0.010502 ± 0.008615     | **0.010494 ± 0.008615**    | 0.011110 ± 0.008883            | 0.033663 ± 0.026510                |
| MIO07 | 0.016732 ± 0.016192 | 0.016500 ± 0.015946 | 0.002588 ± 0.001439 | 0.002596 ± 0.001454 | 0.002488 ± 0.001384 | 0.002474 ± 0.001369     | **0.002459 ± 0.001369**    | 0.003627 ± 0.002288            | 0.010598 ± 0.007637                |
| MIO08 | 0.016694 ± 0.013523 | 0.016423 ± 0.013252 | 0.007849 ± 0.005296 | 0.007877 ± 0.005319 | 0.007807 ± 0.005335 | **0.007152 ± 0.004637** | 0.007183 ± 0.004635        | 0.009127 ± 0.005315            | 0.018915 ± 0.013460                |
| MIO09 | 0.012798 ± 0.011447 | 0.012791 ± 0.011441 | 0.003042 ± 0.002139 | 0.003044 ± 0.002134 | 0.003086 ± 0.002174 | 0.002932 ± 0.002018     | **0.002875 ± 0.001961**    | 0.004647 ± 0.003280            | 0.004286 ± 0.002462                |
| MIO10 | 0.029617 ± 0.026311 | 0.029385 ± 0.026114 | 0.011100 ± 0.009450 | 0.011102 ± 0.009483 | 0.011129 ± 0.009400 | 0.004655 ± 0.003320     | **0.004602 ± 0.003275**    | 0.005494 ± 0.003604            | 0.007598 ± 0.004735                |
| MIO11 | 0.022319 ± 0.019281 | 0.022309 ± 0.019264 | 0.010562 ± 0.007706 | 0.010514 ± 0.007653 | 0.009817 ± 0.007059 | 0.007238 ± 0.004816     | **0.007048 ± 0.004739**    | 0.008820 ± 0.005085            | 0.016886 ± 0.012010                |
| [AVG] | 0.425288 ± 0.356315 | 0.422563 ± 0.353877 | 0.018521 ± 0.016286 | 0.018537 ± 0.016307 | 0.018678 ± 0.016458 | 0.014544 ± 0.012522     | **0.014418 ± 0.012403**    | 0.016140 ± 0.013691            | 0.025919 ± 0.021207                |

## Conclusions

Hopefully here will be some interesting conclusions...

## Appendix

### All the datasets

#### Absolute trajectory error (ATE) [m]

|       | basalt            | curr              | curr_mgt      | curr_mgt_ovl   | curr_mgt_ovl_tmc   | curr_mgt_ovl_tmc_imu   | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:------------------|:------------------|:--------------|:---------------|:-------------------|:-----------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| EMH01 | **0.066 ± 0.023** | 0.073 ± 0.021     | 0.069 ± 0.020 | 0.069 ± 0.020  | 0.074 ± 0.024      | 0.076 ± 0.021          | 0.076 ± 0.021              | 0.090 ± 0.026                  | 0.099 ± 0.030                      |
| EMH02 | 0.062 ± 0.031     | 0.057 ± 0.028     | 0.049 ± 0.025 | 0.049 ± 0.025  | **0.049 ± 0.027**  | 0.050 ± 0.030          | 0.050 ± 0.030              | 0.063 ± 0.023                  | 0.054 ± 0.022                      |
| EMH03 | 0.062 ± 0.019     | 0.062 ± 0.019     | 0.068 ± 0.021 | 0.068 ± 0.021  | 0.057 ± 0.019      | 0.060 ± 0.022          | 0.060 ± 0.022              | **0.056 ± 0.023**              | 0.074 ± 0.034                      |
| EMH05 | **0.145 ± 0.041** | 0.145 ± 0.041     | 0.179 ± 0.056 | 0.179 ± 0.056  | 0.164 ± 0.043      | 0.145 ± 0.043          | 0.145 ± 0.043              | 0.174 ± 0.053                  | 0.147 ± 0.040                      |
| EV101 | 0.043 ± 0.017     | 0.043 ± 0.017     | 0.043 ± 0.017 | 0.043 ± 0.017  | **0.043 ± 0.017**  | 0.043 ± 0.017          | 0.043 ± 0.017              | 0.050 ± 0.020                  | 0.050 ± 0.018                      |
| EV102 | 0.045 ± 0.013     | 0.045 ± 0.013     | 0.047 ± 0.014 | 0.047 ± 0.014  | 0.046 ± 0.013      | 0.054 ± 0.017          | 0.054 ± 0.017              | 0.048 ± 0.015                  | **0.042 ± 0.018**                  |
| EV103 | 0.053 ± 0.020     | 0.053 ± 0.020     | 0.051 ± 0.020 | 0.051 ± 0.020  | **0.050 ± 0.020**  | 0.056 ± 0.014          | 0.056 ± 0.014              | 0.082 ± 0.066                  | 0.087 ± 0.066                      |
| EV201 | 0.039 ± 0.015     | 0.039 ± 0.015     | 0.043 ± 0.017 | 0.043 ± 0.017  | 0.045 ± 0.020      | **0.036 ± 0.013**      | 0.036 ± 0.013              | 0.038 ± 0.013                  | 0.040 ± 0.015                      |
| EV202 | 0.049 ± 0.021     | 0.049 ± 0.021     | 0.046 ± 0.021 | 0.046 ± 0.021  | 0.051 ± 0.026      | **0.040 ± 0.014**      | 0.040 ± 0.014              | 0.041 ± 0.019                  | 0.042 ± 0.016                      |
| MIO01 | 640.149 ± 421.131 | 636.739 ± 418.442 | 0.727 ± 0.494 | 0.726 ± 0.490  | 0.747 ± 0.521      | 0.626 ± 0.359          | 0.620 ± 0.354              | **0.547 ± 0.293**              | 0.550 ± 0.292                      |
| MIO02 | 70.642 ± 50.344   | 70.533 ± 50.266   | 1.342 ± 0.748 | 1.342 ± 0.748  | 1.331 ± 0.740      | 1.188 ± 0.629          | **1.182 ± 0.625**          | 1.252 ± 0.653                  | 1.226 ± 0.656                      |
| MIO03 | 0.141 ± 0.074     | 0.141 ± 0.074     | 0.085 ± 0.036 | 0.085 ± 0.036  | 0.076 ± 0.032      | 0.085 ± 0.032          | 0.093 ± 0.038              | **0.072 ± 0.028**              | 0.073 ± 0.029                      |
| MIO04 | 9.065 ± 4.503     | 9.006 ± 4.477     | 0.235 ± 0.130 | 0.234 ± 0.130  | 0.241 ± 0.130      | 0.207 ± 0.098          | 0.206 ± 0.100              | **0.146 ± 0.075**              | 0.150 ± 0.076                      |
| MIO05 | 0.076 ± 0.043     | 0.076 ± 0.043     | 0.039 ± 0.016 | 0.039 ± 0.016  | 0.038 ± 0.015      | 0.039 ± 0.016          | 0.034 ± 0.012              | 0.026 ± 0.011                  | **0.022 ± 0.010**                  |
| MIO06 | 0.137 ± 0.078     | 0.137 ± 0.078     | 0.069 ± 0.021 | 0.069 ± 0.021  | 0.073 ± 0.022      | 0.049 ± 0.019          | 0.049 ± 0.020              | **0.044 ± 0.021**              | 0.062 ± 0.027                      |
| MIO07 | 0.052 ± 0.036     | 0.051 ± 0.037     | 0.020 ± 0.008 | 0.021 ± 0.008  | 0.024 ± 0.011      | 0.020 ± 0.008          | 0.023 ± 0.011              | **0.018 ± 0.007**              | 0.024 ± 0.009                      |
| MIO08 | 0.093 ± 0.065     | 0.093 ± 0.065     | 0.066 ± 0.021 | 0.066 ± 0.021  | 0.062 ± 0.019      | 0.059 ± 0.020          | 0.057 ± 0.019              | 0.035 ± 0.012                  | **0.026 ± 0.013**                  |
| MIO09 | 0.018 ± 0.013     | 0.018 ± 0.013     | 0.006 ± 0.003 | 0.006 ± 0.003  | 0.006 ± 0.003      | 0.006 ± 0.003          | 0.006 ± 0.003              | 0.006 ± 0.003                  | **0.005 ± 0.003**                  |
| MIO10 | 0.046 ± 0.028     | 0.046 ± 0.028     | 0.026 ± 0.015 | 0.026 ± 0.015  | 0.026 ± 0.015      | 0.016 ± 0.009          | 0.015 ± 0.009              | **0.014 ± 0.007**              | 0.017 ± 0.007                      |
| MIO11 | 0.067 ± 0.034     | 0.067 ± 0.034     | 0.040 ± 0.015 | 0.039 ± 0.015  | 0.038 ± 0.015      | **0.024 ± 0.010**      | 0.024 ± 0.010              | 0.029 ± 0.011                  | 0.025 ± 0.011                      |
| [AVG] | 36.052 ± 23.828   | 35.874 ± 23.688   | 0.163 ± 0.086 | 0.162 ± 0.086  | 0.162 ± 0.087      | 0.144 ± 0.070          | 0.143 ± 0.070              | 0.141 ± 0.069                  | **0.141 ± 0.070**                  |

#### Relative trajectory error (RTE) [m]

|       | basalt              | curr                | curr_mgt                | curr_mgt_ovl        | curr_mgt_ovl_tmc    | curr_mgt_ovl_tmc_imu    | curr_mgt_ovl_tmc_imu_isr   | curr_mgt_ovl_tmc_imu_isr_rec   | curr_mgt_ovl_tmc_imu_isr_rec_cov   |
|:------|:--------------------|:--------------------|:------------------------|:--------------------|:--------------------|:------------------------|:---------------------------|:-------------------------------|:-----------------------------------|
| EMH01 | 0.005266 ± 0.003094 | 0.005243 ± 0.003086 | 0.005214 ± 0.003065     | 0.005215 ± 0.003065 | 0.005138 ± 0.002977 | **0.005120 ± 0.002988** | 0.005120 ± 0.002988        | 0.005653 ± 0.003292            | 0.007173 ± 0.005223                |
| EMH02 | 0.004454 ± 0.002339 | 0.004489 ± 0.002334 | 0.004402 ± 0.002432     | 0.004401 ± 0.002433 | 0.004214 ± 0.002222 | **0.004161 ± 0.002211** | 0.004161 ± 0.002211        | 0.004820 ± 0.002642            | 0.005223 ± 0.003125                |
| EMH03 | 0.011959 ± 0.007684 | 0.011942 ± 0.007676 | 0.012003 ± 0.007755     | 0.012003 ± 0.007755 | 0.011862 ± 0.007594 | **0.011686 ± 0.007596** | 0.011686 ± 0.007596        | 0.012803 ± 0.008322            | 0.027000 ± 0.021251                |
| EMH05 | 0.010939 ± 0.006223 | 0.010974 ± 0.006224 | **0.010677 ± 0.006179** | 0.010677 ± 0.006179 | 0.010819 ± 0.006209 | 0.010726 ± 0.006291     | 0.010726 ± 0.006291        | 0.012333 ± 0.007172            | 0.016402 ± 0.011576                |
| EV101 | 0.013037 ± 0.006246 | 0.013037 ± 0.006249 | 0.013043 ± 0.006257     | 0.013043 ± 0.006257 | 0.013052 ± 0.006271 | **0.013026 ± 0.006258** | 0.013026 ± 0.006258        | 0.013232 ± 0.006344            | 0.022272 ± 0.014314                |
| EV102 | 0.011775 ± 0.005147 | 0.011768 ± 0.005138 | 0.011662 ± 0.005097     | 0.011662 ± 0.005097 | 0.011652 ± 0.005086 | **0.011124 ± 0.004775** | 0.011124 ± 0.004775        | 0.012039 ± 0.005234            | 0.031618 ± 0.020535                |
| EV103 | 0.013336 ± 0.006855 | 0.013359 ± 0.006871 | 0.013119 ± 0.006482     | 0.013115 ± 0.006465 | 0.013109 ± 0.006650 | **0.012279 ± 0.006104** | 0.012279 ± 0.006104        | 0.034668 ± 0.031187            | 0.048125 ± 0.039086                |
| EV201 | 0.003451 ± 0.001951 | 0.003461 ± 0.001967 | 0.003508 ± 0.002030     | 0.003508 ± 0.002030 | 0.003453 ± 0.001964 | **0.003428 ± 0.001972** | 0.003428 ± 0.001972        | 0.004036 ± 0.002219            | 0.005305 ± 0.003648                |
| EV202 | 0.009120 ± 0.005975 | 0.009126 ± 0.005975 | 0.009050 ± 0.005870     | 0.009050 ± 0.005870 | 0.008950 ± 0.005741 | **0.007066 ± 0.003858** | 0.007066 ± 0.003858        | 0.008517 ± 0.004789            | 0.030337 ± 0.022480                |
| MIO01 | 3.482786 ± 2.882257 | 3.455232 ± 2.857748 | 0.053218 ± 0.050661     | 0.053283 ± 0.050766 | 0.053065 ± 0.050284 | 0.037767 ± 0.035763     | 0.036980 ± 0.034926        | **0.032950 ± 0.030879**        | 0.053370 ± 0.046635                |
| MIO02 | 0.966807 ± 0.835925 | 0.965238 ± 0.834399 | 0.075515 ± 0.070433     | 0.075563 ± 0.070483 | 0.077877 ± 0.072748 | 0.064713 ± 0.060449     | **0.064532 ± 0.060300**    | 0.076237 ± 0.072524            | 0.079650 ± 0.072844                |
| MIO03 | 0.017892 ± 0.015679 | 0.017967 ± 0.015763 | 0.006246 ± 0.004214     | 0.006250 ± 0.004213 | 0.006952 ± 0.005151 | 0.006260 ± 0.004208     | **0.006075 ± 0.004005**    | 0.006790 ± 0.004552            | 0.020128 ± 0.016410                |
| MIO04 | 0.080239 ± 0.069821 | 0.079929 ± 0.069539 | 0.018737 ± 0.015821     | 0.018810 ± 0.015902 | 0.018252 ± 0.015375 | **0.013075 ± 0.010332** | 0.013089 ± 0.010349        | 0.015719 ± 0.012333            | 0.027176 ± 0.020810                |
| MIO05 | 0.009999 ± 0.009264 | 0.009902 ± 0.009163 | 0.003422 ± 0.002397     | 0.003429 ± 0.002408 | 0.003412 ± 0.002411 | 0.003217 ± 0.002214     | 0.003258 ± 0.002255        | **0.003022 ± 0.001855**        | 0.012838 ± 0.009763                |
| MIO06 | 0.022290 ± 0.019768 | 0.022516 ± 0.020016 | 0.011450 ± 0.009588     | 0.011439 ± 0.009566 | 0.011570 ± 0.009717 | 0.010502 ± 0.008615     | **0.010494 ± 0.008615**    | 0.011110 ± 0.008883            | 0.033663 ± 0.026510                |
| MIO07 | 0.016732 ± 0.016192 | 0.016500 ± 0.015946 | 0.002588 ± 0.001439     | 0.002596 ± 0.001454 | 0.002488 ± 0.001384 | 0.002474 ± 0.001369     | **0.002459 ± 0.001369**    | 0.003627 ± 0.002288            | 0.010598 ± 0.007637                |
| MIO08 | 0.016694 ± 0.013523 | 0.016423 ± 0.013252 | 0.007849 ± 0.005296     | 0.007877 ± 0.005319 | 0.007807 ± 0.005335 | **0.007152 ± 0.004637** | 0.007183 ± 0.004635        | 0.009127 ± 0.005315            | 0.018915 ± 0.013460                |
| MIO09 | 0.012798 ± 0.011447 | 0.012791 ± 0.011441 | 0.003042 ± 0.002139     | 0.003044 ± 0.002134 | 0.003086 ± 0.002174 | 0.002932 ± 0.002018     | **0.002875 ± 0.001961**    | 0.004647 ± 0.003280            | 0.004286 ± 0.002462                |
| MIO10 | 0.029617 ± 0.026311 | 0.029385 ± 0.026114 | 0.011100 ± 0.009450     | 0.011102 ± 0.009483 | 0.011129 ± 0.009400 | 0.004655 ± 0.003320     | **0.004602 ± 0.003275**    | 0.005494 ± 0.003604            | 0.007598 ± 0.004735                |
| MIO11 | 0.022319 ± 0.019281 | 0.022309 ± 0.019264 | 0.010562 ± 0.007706     | 0.010514 ± 0.007653 | 0.009817 ± 0.007059 | 0.007238 ± 0.004816     | **0.007048 ± 0.004739**    | 0.008820 ± 0.005085            | 0.016886 ± 0.012010                |
| [AVG] | 0.238075 ± 0.198249 | 0.236580 ± 0.196908 | 0.014320 ± 0.011216     | 0.014329 ± 0.011227 | 0.014385 ± 0.011288 | 0.011930 ± 0.008990     | **0.011860 ± 0.008924**    | 0.014282 ± 0.011090            | 0.023928 ± 0.018726                |

[1]: https://gitlab.com/VladyslavUsenko/basalt
[2]: https://gitlab.freedesktop.org/mateosss/basalt
[3]: https://gitlab.freedesktop.org/mateosss/basalt/-/commit/673cc5c68b1889f4d6567fa85fe2631672704351
