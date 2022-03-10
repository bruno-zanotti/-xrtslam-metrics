from collections import namedtuple
from subprocess import run
from os import environ

THESIS_DATASETS = "/home/mateo/Desktop/xrt/eurocs/thesis-datasets"
oxrinstall = environ["oxrinstall"]
# bsltdeps = environ["bsltdeps"]
# orb3deps = environ["orb3deps"]
kviodeps = environ["kviodeps"]

environ["OXR_DEBUG_GUI"] = "1"
environ["XR_RUNTIME_JSON"] = f"{oxrinstall}/share/openxr/1/openxr_monado.json"
environ["QWERTY_ENABLE"] = "true"
environ["QWERTY_COMBINE"] = "true"
environ["SLAM_SUBMIT_FROM_START"] = "true"
environ["SLAM_WRITE_CSVS"] = "true"
environ["EUROC_LOG"] = "debug"
environ["EUROC_HMD"] = "false"
environ["SLAM_LOG"] = "debug"


def sh(command):
    "Executes command in shell and returns its exit status"
    return run(command, shell=True).returncode


R = namedtuple("Run", ["system_name", "dataset_name", "config_path"])

"""
runs_bslt_posiccv21_float = [
    R("BNF", "d455-640-easy", f"{bsltdeps}/basalt/data/monado/d455-640x480.toml"),
    R("BNF", "d455-640-hard", f"{bsltdeps}/basalt/data/monado/d455-640x480.toml"),
    R("BNF", "d455-848-easy", f"{bsltdeps}/basalt/data/monado/d455-848x480.toml"),
    R("BNF", "d455-848-hard", f"{bsltdeps}/basalt/data/monado/d455-848x480.toml"),
    R("BNF", "ody-easy", f"{bsltdeps}/basalt/data/monado/odysseyplus_rt8.toml"),
    R("BNF", "ody-hard", f"{bsltdeps}/basalt/data/monado/odysseyplus_rt8.toml"),
    R("BNFK", "ody-easy", f"{bsltdeps}/basalt/data/monado/odysseyplus_kb4.toml"),
    R("BNFK", "ody-hard", f"{bsltdeps}/basalt/data/monado/odysseyplus_kb4.toml"),
    R("BNF", "euroc-V202", f"{bsltdeps}/basalt/data/monado/euroc.toml"),
    R("BNF", "euroc-MH04", f"{bsltdeps}/basalt/data/monado/euroc.toml"),
]

runs_bslt_posiccv21_double = [
    R("BND", "d455-640-easy", f"{bsltdeps}/basalt/data/monado/d455-640x480.toml"),
    R("BND", "d455-640-hard", f"{bsltdeps}/basalt/data/monado/d455-640x480.toml"),
    R("BND", "d455-848-easy", f"{bsltdeps}/basalt/data/monado/d455-848x480.toml"),
    R("BND", "d455-848-hard", f"{bsltdeps}/basalt/data/monado/d455-848x480.toml"),
    R("BND", "ody-easy", f"{bsltdeps}/basalt/data/monado/odysseyplus_rt8.toml"),
    R("BND", "ody-hard", f"{bsltdeps}/basalt/data/monado/odysseyplus_rt8.toml"),
    R("BNDK", "ody-easy", f"{bsltdeps}/basalt/data/monado/odysseyplus_kb4.toml"),
    R("BNDK", "ody-hard", f"{bsltdeps}/basalt/data/monado/odysseyplus_kb4.toml"),
    R("BND", "euroc-V202", f"{bsltdeps}/basalt/data/monado/euroc.toml"),
    R("BND", "euroc-MH04", f"{bsltdeps}/basalt/data/monado/euroc.toml"),
]

runs_bslt_preiccv21 = [
    R("BO", "d455-640-easy", f"{bsltdeps}/basalt/data/monado/d455-640x480.toml"),
    R("BO", "d455-640-hard", f"{bsltdeps}/basalt/data/monado/d455-640x480.toml"),
    R("BO", "d455-848-easy", f"{bsltdeps}/basalt/data/monado/d455-848x480.toml"),
    R("BO", "d455-848-hard", f"{bsltdeps}/basalt/data/monado/d455-848x480.toml"),
    R("BO", "ody-easy", f"{bsltdeps}/basalt/data/monado/odysseyplus_rt8.toml"),
    R("BO", "ody-hard", f"{bsltdeps}/basalt/data/monado/odysseyplus_rt8.toml"),
    R("BOK", "ody-easy", f"{bsltdeps}/basalt/data/monado/odysseyplus_kb4.toml"),
    R("BOK", "ody-hard", f"{bsltdeps}/basalt/data/monado/odysseyplus_kb4.toml"),
    R("BO", "euroc-V202", f"{bsltdeps}/basalt/data/monado/euroc.toml"),
    R("BO", "euroc-MH04", f"{bsltdeps}/basalt/data/monado/euroc.toml"),
]

runs_orb3_pos1 = [
    R("ON", "d455-640-easy", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/D455-640x480-30fps.yaml"),
    R("ON", "d455-640-hard", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/D455-640x480-30fps.yaml"),
    R("ON", "d455-848-easy", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/D455-848x480-60fps.yaml"),
    R("ON", "d455-848-hard", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/D455-848x480-60fps.yaml"),
    R("ON", "ody-easy", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/odysseyplus-stereo-inertial.yaml"),
    R("ON", "ody-hard", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/odysseyplus-stereo-inertial.yaml"),
    R("ON", "euroc-V202", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/EuRoC.yaml"),
    R("ON", "euroc-MH04", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/EuRoC.yaml"),
]

runs_orb3_pre1 = [
    R("OO", "d455-640-easy", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/D455-640x480-30fps.yaml"),
    R("OO", "d455-640-hard", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/D455-640x480-30fps.yaml"),
    R("OO", "d455-848-easy", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/D455-848x480-60fps.yaml"),
    R("OO", "d455-848-hard", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/D455-848x480-60fps.yaml"),
    R("OO", "ody-easy", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/odysseyplus-stereo-inertial.yaml"),
    R("OO", "ody-hard", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/odysseyplus-stereo-inertial.yaml"),
    R("OO", "euroc-V202", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/EuRoC.yaml"),
    R("OO", "euroc-MH04", f"{orb3deps}/ORB_SLAM3/Examples/Stereo-Inertial/EuRoC.yaml"),
]
"""

runs_kvio = [
    R("K", "d455-640-easy", f"{kviodeps}/Kimera-VIO/params/D455-640x480-30fps/flags/Monado.flags"),
    R("K", "d455-640-hard", f"{kviodeps}/Kimera-VIO/params/D455-640x480-30fps/flags/Monado.flags"),
    R("K", "d455-848-easy", f"{kviodeps}/Kimera-VIO/params/D455-848x480-60fps/flags/Monado.flags"),
    R("K", "d455-848-hard", f"{kviodeps}/Kimera-VIO/params/D455-848x480-60fps/flags/Monado.flags"),
    R("K", "ody-easy", f"{kviodeps}/Kimera-VIO/params/odysseyplus/flags/Monado.flags"),
    R("K", "ody-hard", f"{kviodeps}/Kimera-VIO/params/odysseyplus/flags/Monado.flags"),
    R("K", "euroc-V202", f"{kviodeps}/Kimera-VIO/params/Euroc/flags/Monado.flags"),
    R("K", "euroc-MH04", f"{kviodeps}/Kimera-VIO/params/Euroc/flags/Monado.flags"),
]

runs = runs_kvio

for system_name, dataset_name, config_path in runs:
    environ["SYSTEM_NAME"] = system_name
    environ["DATASET_NAME"] = dataset_name
    environ["SLAM_CONFIG"] = config_path
    environ["EUROC_PATH"] = f"{THESIS_DATASETS}/{dataset_name}"
    print("\n\n================ NEW RUN ================")
    print(f"{environ['SYSTEM_NAME']=}")
    print(f"{environ['DATASET_NAME']=}")
    print(f"{environ['SLAM_CONFIG']=}")
    print(f"{environ['EUROC_PATH']=}")
    print()
    sh("./hello_xr -g Vulkan")
