import argparse
import os
from pathlib import Path

import cv2
import rosbag

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def main():
    parser = argparse.ArgumentParser(description="Make EuRoC dataset from Hilti ROS bag")
    parser.add_argument("bag", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    bag_path = args.bag
    euroc_path = args.output_path

    mav0_path = euroc_path / "mav0"
    mav0_path.mkdir(parents=True, exist_ok=True)

    bag = rosbag.Bag(bag_path, "r")

    print("Exporting IMU data")
    imu_path = mav0_path / f"imu0"
    imu_path.mkdir()

    csv_path = imu_path / "data.csv"
    csv_file = open(csv_path, "w")
    csv_file.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\r\n")
    for topic, msg, t in bag.read_messages(topics=["/alphasense/imu"]):
        t = int(f"{msg.header.stamp.secs}{msg.header.stamp.nsecs}")
        w = msg.angular_velocity
        a = msg.linear_acceleration
        csv_file.write(f"{t},{w.x},{w.y},{w.z},{a.x},{a.y},{a.z}\r\n")
    csv_file.close()
    print("Done.")

    # Export camera data
    bridge = CvBridge()
    for i in range(5):
        print(f"Exporting cam{i} data")
        cam_path = mav0_path / f"cam{i}"
        cam_path.mkdir()

        data_path = cam_path / "data"
        data_path.mkdir()

        csv_path = cam_path / "data.csv"
        csv_file = open(csv_path, "w")
        csv_file.write("#timestamp [ns],filename\r\n")

        for c, (topic, msg, t) in enumerate(bag.read_messages(topics=[f"/alphasense/cam{i}/image_raw"])):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
            timestamp = int(f"{msg.header.stamp.secs}{msg.header.stamp.nsecs:09d}")
            cv2.imwrite(str(data_path / f"{timestamp}.png"), cv_img)
            csv_file.write(f"{timestamp},{timestamp}.png\r\n")
            print(f"Wrote image {c}\r", end="")

        csv_file.close()
        print(f"Done.")

    bag.close()

    return 0

if __name__ == '__main__':
    main()
