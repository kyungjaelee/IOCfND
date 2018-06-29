import numpy as np
import pickle

import rospy
import message_filters

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

from sensor_msgs.msg import Imu

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist
from dbw_mkz_msgs.msg import ThrottleCmd
from dbw_mkz_msgs.msg import SteeringCmd
from dbw_mkz_msgs.msg import BrakeCmd

bridge = CvBridge()

img_list = []
pc2_list = []
imu_list = []

cmd_vel_list = []
throttle_list = []
steering_list = []
brake_list = []

def callback(img, pc2, imu, cmd_vel, throttle, steering, brake):

    img_np = bridge.imgmsg_to_cv2(img, "bgr8")
    img_np = np.array(img_np, dtype=np.uint8)
    img_list.append(img_np)

    pc2_np = []
    for p in point_cloud2.read_points(pc2, skip_nans=True, field_names=("x", "y", "z")):
        pc2_np.append(np.asarray(p))
    pc2_np = np.asarray(pc2_np)
    pc2_list.append(pc2_np)

    imu_np = np.zeros((4 + 3 + 3,))
    imu_np[0] = imu.orientation.x
    imu_np[1] = imu.orientation.y
    imu_np[2] = imu.orientation.z
    imu_np[3] = imu.orientation.w

    imu_np[4] = imu.angular_velocity.x
    imu_np[5] = imu.angular_velocity.y
    imu_np[6] = imu.angular_velocity.z

    imu_np[7] = imu.linear_acceleration.x
    imu_np[8] = imu.linear_acceleration.y
    imu_np[9] = imu.linear_acceleration.z
    imu_list.append(imu_np)

    cmd_vel_np = np.zeros((2,))
    cmd_vel_np[0] = cmd_vel.angular.z
    cmd_vel_np[1] = cmd_vel.linear.x
    cmd_vel_list.append(cmd_vel_np)

    throttle_np = np.zeros((1,))
    throttle_np[0] = throttle.pedal_cmd
    throttle_list.append(throttle_np)

    steering_np = np.zeros((1,))
    steering_np[0] = steering.steering_wheel_angle_cmd
    steering_list.append(steering_np)

    brake_np = np.zeros((1,))
    brake_np[0] = brake.pedal_cmd
    brake_list.append(brake_np)

if __name__ == '__main__':
    img_sub = message_filters.Subscriber('/vehicle/front_camera/image_rect_color', Image)
    pc2_sub = message_filters.Subscriber('/vehicle/velodyne_points', PointCloud2)
    imu_sub = message_filters.Subscriber('/vehicle/imu/data_raw', Imu)
    cmd_vel_sub = message_filters.Subscriber('/vehicle/cmd_vel', Twist)
    throttle_sub = message_filters.Subscriber('/vehicle/throttle_cmd', ThrottleCmd)
    steering_sub = message_filters.Subscriber('/vehicle/steering_cmd', SteeringCmd)
    brake_sub = message_filters.Subscriber('/vehicle/brake_cmd', BrakeCmd)

    ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc2_sub, imu_sub, cmd_vel_sub, throttle_sub, steering_sub, brake_sub],
                                                     10, 0.05, allow_headerless=True)

    ts.registerCallback(callback)
    rospy.init_node("data_collect",anonymous=True)

    print("Start data collection")
    n_data = 300
    rate = rospy.Rate(10)
    for i in range(n_data):
       rate.sleep()

    img_list = np.asarray(img_list)
    pc2_list = np.asarray(pc2_list)
    imu_list = np.asarray(imu_list)

    cmd_vel_list = np.asarray(cmd_vel_list)
    throttle_list = np.asarray(throttle_list)
    steering_list = np.asarray(steering_list)
    brake_list = np.asarray(brake_list)

    print("Data shape print:")
    print(img_list.shape)
    print(pc2_list.shape)
    print(imu_list.shape)
    print(cmd_vel_list.shape)
    print(throttle_list.shape)
    print(steering_list.shape)
    print(brake_list.shape)

    training_data = {'img_list':img_list,
                     'pc2_list': pc2_list,
                     'imu_list': imu_list,
                     'cmd_vel_list':cmd_vel_list,
                     'throttle_list':throttle_list,
                     'steering_list':steering_list,
                     'brake_list':brake_list}

    with open('training_data.pkl', 'wb') as handle:
        pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Save {} training data".format(n_data))
