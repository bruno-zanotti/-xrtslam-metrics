# Basalt vs ORB-SLAM3

- Datasets used:
  - MSDMO: Monado SLAM Dataset Odyssey Plus.
- Configurations used:
  - ORB-SLAM3: ORB-SLAM master from [https://github.com/UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
  - basalt: Original Basalt from Usenko [https://gitlab.com/VladyslavUsenko/basalt](https://gitlab.com/VladyslavUsenko/basalt)
  - basalt_with_map: our implementation Basalt + persisten mapper.


Segment drift per meter error (SDM 0.01m) [m/m]

|       | ORB-SLAM3       | basalt              | basalt_with_map     |
|:------|:----------------|:--------------------|:--------------------|
| MOO01 | 1.5761 ± 0.5866 | **0.2291 ± 0.2371** | 0.2913 ± 0.2861     |
| MOO02 | 1.2754 ± 0.4671 | 0.3605 ± 0.3982     | **0.2536 ± 0.3176** |
| MOO03 | —               | **0.2986 ± 0.3134** | 0.4177 ± 0.4245     |
| MOO04 | —               | 0.4475 ± 0.7084     | **0.2857 ± 0.3254** |
| MOO05 | 1.6358 ± 0.4681 | 0.3018 ± 0.3668     | **0.2758 ± 0.3384** |
| MOO06 | 1.5403 ± 0.4549 | **0.2067 ± 0.2778** | 0.2804 ± 0.3301     |
| MOO07 | 1.5677 ± 0.4315 | **0.2326 ± 0.3227** | 0.2562 ± 0.3006     |
| MOO08 | 0.8304 ± 0.3293 | **0.2591 ± 0.2751** | 0.5647 ± 4.1388     |
| MOO09 | 0.5976 ± 0.2382 | 0.5134 ± 0.3960     | **0.0549 ± 0.0158** |
| MOO10 | 0.8368 ± 0.3569 | 0.3163 ± 0.3869     | **0.1551 ± 0.1814** |
| MOO11 | 0.4545 ± 0.2198 | 0.1847 ± 0.2816     | **0.1725 ± 0.1951** |
| MOO12 | —               | —                   | —                   |
| MOO13 | 1.1724 ± 0.3603 | 0.9176 ± 0.3010     | **0.2616 ± 0.2739** |
| MOO14 | 1.1930 ± 0.4109 | **0.2963 ± 0.3361** | 0.3304 ± 0.4230     |
| MOO15 | —               | —                   | **0.6475 ± 0.3918** |
| MOO16 | 0.6277 ± 0.4182 | —                   | **0.1173 ± 0.1841** |
| [AVG] | 1.1090 ± 0.3951 | 0.3511 ± 0.3539     | **0.2910 ± 0.5418** |

Average completion percentage [%]

|       | ORB-SLAM3   | basalt   | basalt_with_map   |
|:------|:------------|:---------|:------------------|
| MOO01 | ✓           | ✓        | ✓                 |
| MOO02 | ✓           | ✓        | ✓                 |
| MOO03 | —           | ✓        | ✓                 |
| MOO04 | —           | ✓        | ✓                 |
| MOO05 | ✓           | ✓        | ✓                 |
| MOO06 | ✓           | ✓        | ✓                 |
| MOO07 | ✓           | ✓        | ✓                 |
| MOO08 | 11.51%      | ✓        | ✓                 |
| MOO09 | 23.29%      | ✓        | ✓                 |
| MOO10 | 7.33%       | ✓        | ✓                 |
| MOO11 | 7.67%       | ✓        | ✓                 |
| MOO12 | —           | —        | —                 |
| MOO13 | 77.74%      | ✓        | ✓                 |
| MOO14 | ✓           | ✓        | ✓                 |
| MOO15 | —           | —        | ✓                 |
| MOO16 | 0.60%       | —        | ✓                 |
| [AVG] | 60.07%      | ✓        | ✓                 |

Absolute trajectory error (ATE) [m]

|       | ORB-SLAM3           | basalt              | basalt_with_map   |
|:------|:--------------------|:--------------------|:------------------|
| MOO01 | 0.889 ± 0.847       | 0.428 ± 0.226       | **0.145 ± 0.065** |
| MOO02 | 616.975 ± 512.712   | 1.474 ± 0.837       | **0.135 ± 0.054** |
| MOO03 | —                   | 0.292 ± 0.116       | **0.279 ± 0.131** |
| MOO04 | —                   | 4.050 ± 2.070       | **0.275 ± 0.139** |
| MOO05 | **0.025 ± 0.012**   | 0.067 ± 0.048       | 0.084 ± 0.036     |
| MOO06 | **0.016 ± 0.010**   | 0.090 ± 0.037       | 0.088 ± 0.038     |
| MOO07 | **0.016 ± 0.012**   | 0.027 ± 0.011       | 0.044 ± 0.021     |
| MOO08 | **0.008 ± 0.004**   | 0.449 ± 0.171       | 0.074 ± 0.023     |
| MOO09 | **0.006 ± 0.003**   | 0.014 ± 0.010       | 0.011 ± 0.007     |
| MOO10 | **0.006 ± 0.003**   | 0.016 ± 0.009       | 0.015 ± 0.006     |
| MOO11 | **0.008 ± 0.003**   | 0.035 ± 0.018       | 0.025 ± 0.008     |
| MOO12 | —                   | —                   | —                 |
| MOO13 | 2020.077 ± 1216.818 | 2067.699 ± 1230.732 | **0.114 ± 0.054** |
| MOO14 | 4332.685 ± 2958.369 | 0.359 ± 0.087       | **0.090 ± 0.052** |
| MOO15 | —                   | —                   | **0.123 ± 0.056** |
| MOO16 | **0.005 ± 0.002**   | —                   | 0.008 ± 0.005     |
| [AVG] | 580.893 ± 390.733   | 159.615 ± 94.952    | **0.101 ± 0.046** |

Relative trajectory error (RTE) [m]

|       | ORB-SLAM3               | basalt                  | basalt_with_map         |
|:------|:------------------------|:------------------------|:------------------------|
| MOO01 | 0.529578 ± 0.473832     | **0.020509 ± 0.014011** | 0.023172 ± 0.015210     |
| MOO02 | 128.149431 ± 127.647623 | 0.039301 ± 0.030429     | **0.023230 ± 0.014747** |
| MOO03 | —                       | **0.017824 ± 0.011549** | 0.024902 ± 0.016928     |
| MOO04 | —                       | 0.081972 ± 0.074747     | **0.026049 ± 0.017448** |
| MOO05 | 0.131091 ± 0.081405     | 0.013491 ± 0.012331     | **0.011298 ± 0.007312** |
| MOO06 | 0.233768 ± 0.148404     | **0.014265 ± 0.009916** | 0.019713 ± 0.013184     |
| MOO07 | 0.150159 ± 0.088548     | **0.010115 ± 0.008296** | 0.010594 ± 0.006525     |
| MOO08 | **0.016885 ± 0.008652** | 0.038105 ± 0.024501     | 0.029502 ± 0.020926     |
| MOO09 | 0.011389 ± 0.004213     | 0.010139 ± 0.008432     | **0.005872 ± 0.004493** |
| MOO10 | **0.005984 ± 0.002242** | 0.009341 ± 0.006785     | 0.009949 ± 0.007298     |
| MOO11 | 0.011854 ± 0.005767     | **0.009301 ± 0.006197** | 0.011533 ± 0.007257     |
| MOO12 | —                       | —                       | —                       |
| MOO13 | 23.230463 ± 18.192110   | 21.559450 ± 16.540984   | **0.029619 ± 0.018511** |
| MOO14 | 40.732104 ± 34.381158   | 0.020058 ± 0.015370     | **0.019713 ± 0.014359** |
| MOO15 | —                       | —                       | **0.008372 ± 0.006944** |
| MOO16 | 0.004649 ± 0.002354     | —                       | **0.001378 ± 0.001089** |
| [AVG] | 16.100613 ± 15.086359   | 1.680298 ± 1.289504     | **0.016993 ± 0.011482** |

Segment drift per meter error (SDM 0.01m) [m/m]

|       | ORB-SLAM3       | basalt              | basalt_with_map     |
|:------|:----------------|:--------------------|:--------------------|
| MOO01 | 1.5761 ± 0.5866 | **0.2291 ± 0.2371** | 0.2913 ± 0.2861     |
| MOO02 | 1.2754 ± 0.4671 | 0.3605 ± 0.3982     | **0.2536 ± 0.3176** |
| MOO03 | —               | **0.2986 ± 0.3134** | 0.4177 ± 0.4245     |
| MOO04 | —               | 0.4475 ± 0.7084     | **0.2857 ± 0.3254** |
| MOO05 | 1.6358 ± 0.4681 | 0.3018 ± 0.3668     | **0.2758 ± 0.3384** |
| MOO06 | 1.5403 ± 0.4549 | **0.2067 ± 0.2778** | 0.2804 ± 0.3301     |
| MOO07 | 1.5677 ± 0.4315 | **0.2326 ± 0.3227** | 0.2562 ± 0.3006     |
| MOO08 | 0.8304 ± 0.3293 | **0.2591 ± 0.2751** | 0.5647 ± 4.1388     |
| MOO09 | 0.5976 ± 0.2382 | 0.5134 ± 0.3960     | **0.0549 ± 0.0158** |
| MOO10 | 0.8368 ± 0.3569 | 0.3163 ± 0.3869     | **0.1551 ± 0.1814** |
| MOO11 | 0.4545 ± 0.2198 | 0.1847 ± 0.2816     | **0.1725 ± 0.1951** |
| MOO12 | —               | —                   | —                   |
| MOO13 | 1.1724 ± 0.3603 | 0.9176 ± 0.3010     | **0.2616 ± 0.2739** |
| MOO14 | 1.1930 ± 0.4109 | **0.2963 ± 0.3361** | 0.3304 ± 0.4230     |
| MOO15 | —               | —                   | **0.6475 ± 0.3918** |
| MOO16 | 0.6277 ± 0.4182 | —                   | **0.1173 ± 0.1841** |
| [AVG] | 1.1090 ± 0.3951 | 0.3511 ± 0.3539     | **0.2910 ± 0.5418** |