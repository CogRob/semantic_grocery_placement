#!/usr/bin/env python
import os
import rospy
from fetch_robot import Fetch_Robot



if __name__=='__main__':
    rospy.init_node('test_node')
    robot = Fetch_Robot(sim=isSim)
    tf_helper = TF_Helper()
