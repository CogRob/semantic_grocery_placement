#!/usr/bin/env python
import os
import rospy
import numpy as np
from fetch_robot import Fetch_Robot
from regrasp_planner import RegripPlanner
from tf_util import TF_Helper, PandaPosMax_t_PosMat, transformProduct, getMatrixFromQuaternionAndTrans, getTransformFromPoseMat, align_vectors
from rail_segmentation.srv import SearchTable
from scipy.spatial.transform import Rotation as R
from scipy.special import softmax

from std_srvs.srv import Empty
from rail_manipulation_msgs.srv import SegmentObjects 

import pandaplotutils.pandageom as pandageom
import pandaplotutils.pandactrl as pandactrl
from manipulation.grip.fetch_gripper import fetch_grippernm
from utils import robotmath as rm
from utils import dbcvt as dc
from database import dbaccess as db


from semantic_grocery_placement.srv import StopOctoMap

from object_cluster_planner import Grocery_cluster

SHELF1 = .31
SHELF2 = 0




def add_table(robot, tf_helper):
    """
    add the table into the planning scene
    """
    # get table pose
    table_transform = tf_helper.getTransform('/base_link', '/Table')
    # add it to planning scene
    robot.addCollisionTable("table", table_transform[0][0], table_transform[0][1], table_transform[0][2]+.001, \
        table_transform[1][0], table_transform[1][1], table_transform[1][2], table_transform[1][3], \
        0.7, 1.5, 0.09) #.7 1.5, .06 

def add_object(robot, tf_helper,object_name,object_path=None):
    """
    add the object into the planning scene
    """
    this_dir, filename = os.path.split(os.path.realpath(__file__)) 
    object_path = os.path.join(os.path.split(this_dir)[0], "objects", object_name + ".stl") 
    # add the object into the planning scene 
    current_object_transform = tf_helper.getTransform('/base_link', '/' + object_name)# get object pose
    robot.addCollisionObject(object_name + "_collision", current_object_transform, object_path, size_scale = .15)# add it to planning scene



# pickup is the action to move the gripper up in the base_link frame
def pickup(tf_helper, height):
  """Pick up object"""
  
  target_transform = tf_helper.getTransform('/base_link', '/gripper_link')
  target_transform[0][2] += height

  robot.switchController('my_cartesian_motion_controller', 'arm_controller')

  while not rospy.is_shutdown():
    if robot.moveToFrame(target_transform, True):
      break
    rospy.sleep(0.05)

  robot.switchController('arm_controller', 'my_cartesian_motion_controller')

  return True, target_transform


# grasp_object will try to grasp the object
# input: object pose
# return: issuccess, grasp transform in object frame
def grasp_object( planner, object_pose, given_grasps=None, object_name = None):

  # this function will return both pre grasp pose and grasp pose in base link frame
  def find_grasping_point(planner, tran_base_object, gripper_pos_list=None):
    if gripper_pos_list == None:
      # filter out based on placement so we know which is the actuall grasp
      gripper_pos_list = planner.getGraspsbyPlacementPose(tran_base_object)
      if len(gripper_pos_list) == 0:
        # we have to try all grasps
        gripper_pos_list = planner.getAllGrasps()
    else:
      pre_definedGrasps = []
      for obj_grasp_pos, jaw_width in gripper_pos_list:
        pre_definedGrasps.append([obj_grasp_pos, jaw_width])
      possibleGrasps = planner.getAllGrasps()
      gripper_pos_list = []
      for t_pose, _ in pre_definedGrasps:
        grasp_inv_rot = np.transpose(t_pose[:3,:3])
        grasp_trans = t_pose[:,3:]

        grasp_temp = []
        rot_diff = []
        tran_diff = []
        for ppose, pwidth in possibleGrasps:
          rot_diff.append(np.linalg.norm(R.from_dcm(np.dot(grasp_inv_rot,ppose[:3,:3])).as_rotvec()))
          tran_diff.append(np.linalg.norm(ppose[:,3:] - grasp_trans))
          grasp_temp.append([ppose, pwidth, 0])
        tran_diff = softmax(tran_diff)
        rot_diff = softmax(rot_diff)
        for i in range(len(grasp_temp)):
          grasp_temp[i][2] = tran_diff[i] + rot_diff[i]

        def sortfun(e):
            return e[2]
        grasp_temp.sort(key=sortfun)

        gripper_pos_list.append((grasp_temp[0][0], grasp_temp[0][1]))

    print "Going through this many grasp pose: " ,len(gripper_pos_list)
    for i, (obj_grasp_pos, jaw_width) in enumerate(gripper_pos_list):

        

        obj_grasp_trans_obframe = getTransformFromPoseMat(obj_grasp_pos) #Tranfrom gripper posmatx to (trans,rot)
        # obj_grasp_trans_obframe = transformProduct(obj_grasp_trans_obframe, [[0.01,0,0],[0,0,0,1]]) # try to move the gripper forward little
        obj_pre_grasp_trans =  transformProduct(obj_grasp_trans_obframe, [[-0.06,0,0],[0,0,0,1]]) #adjust the grasp pos to be a little back 
        obj_pre_grasp_trans = transformProduct(tran_base_object, obj_pre_grasp_trans)
        obj_grasp_trans = transformProduct(tran_base_object, obj_grasp_trans_obframe)

        # need to ensure both grasp and pre-grasp is valid for robot
        # grasp_ik_result = robot.solve_ik_sollision_free_in_base(obj_grasp_trans, 80)

        # if grasp_ik_result == None:
        #     print 'check on grasp ', i
        #     continue

        # publish the next regrasp pos in the tf for debug
        tf_helper.pubTransform("grasp pos", obj_pre_grasp_trans)

        pre_grasp_ik_result = robot.solve_ik_sollision_free_in_base(obj_pre_grasp_trans, 30) #30
        
        if pre_grasp_ik_result == None:
            print 'check on grasp ', i
            continue
        
        return obj_pre_grasp_trans, pre_grasp_ik_result, obj_grasp_trans, jaw_width, obj_grasp_trans_obframe
    # if not solution, then return None
    return None, None, None, None, None

  #Move to starting position
  robot.openGripper()

  #Go through all grasp pos and find a valid pos. 

  obj_pre_grasp_trans,  pre_grasp_ik_result, obj_grasp_trans, gripper_width, obj_grasp_trans_obframe = find_grasping_point(planner, object_pose, given_grasps)

  if pre_grasp_ik_result == None: # can't find any solution then return false.
      return False, None, None

  raw_input("found solution to pick")

  # move to pre grasp pose
  plan = robot.planto_pose(obj_pre_grasp_trans)
  robot.display_trajectory(plan)
  raw_input("ready to pre-grasp")
  robot.execute_plan(plan)

  robot.switchController('my_cartesian_motion_controller', 'arm_controller')

  # move to grasp pose
  raw_input("ready to grasp")
  while not rospy.is_shutdown():
    if robot.moveToFrame(obj_grasp_trans, True):
      break
    rospy.sleep(0.05)

  robot.switchController('arm_controller', 'my_cartesian_motion_controller')

  # close the gripper with proper width
  print("grasping with width: ", gripper_width)
  # raw_input("ready to close grasp")
  robot.closeGripper()

  # attach object into the hand
  robot.attachManipulatedObject(object_name + "_collision")

  return True, obj_grasp_trans_obframe, gripper_width

# get init grasps will return a set of init grasp of the object
# return: issuccess, list of target grasps
def getInitGrasps(gdb, object_name):
    sql = "SELECT * FROM object WHERE object.name LIKE '%s'" % object_name
    result = gdb.execute(sql)
    if not result:
        print "please add the object name to the table first!!"
        return False, None
    else:
        objectId = int(result[0][0])

    sql = "SELECT grasppose, jawwidth FROM initgrasps WHERE idobject = '%d'" % objectId
    initgrasp_result = gdb.execute(sql)

    # target grasps (grasp pose(in numpy format), jawwidth(in meter))
    init_grasps = []
    for grasppose, jawwidth in initgrasp_result:
        init_grasps.append((PandaPosMax_t_PosMat(pandageom.cvtMat4(rm.rodrigues([0, 1, 0], 180)) * dc.strToMat4(grasppose)), float(jawwidth) / 1000))

    return True, init_grasps

def set_ArmPos(robot, defult_joints=None):
  robot.getjointNamesNValues()
  defult_joints = [1.4157397747039795, 0.6173441410064697, 0.07052537798881531, -2.2417545318603516, 
                  2.046666383743286, -1.8662853240966797, -2.707148790359497]
  plan = robot.planto_joints(defult_joints)
  robot.display_trajectory(plan)
  raw_input("ready to execute")
  robot.execute_plan(plan)
  raw_input("done moving to pos?")

def liftRobotTorso(robot,isSim,value):
  robot.getjointNamesNValues()
  if isSim:
    robot.lift_robot_torso([value])
  else:
    print("No controler yet for torso not in Sim")

def tiltHeadJoint(robot):
  pass

def StopNStart_octomap():
   #Stop octomap
  rospy.wait_for_service('stop_octo_map')
  try:
      octoclient = rospy.ServiceProxy('stop_octo_map', StopOctoMap)
      octoclient()
  except rospy.ServiceException as e:
      print ("Fail to stop octo map controller: %s"%e)  


def findMatching_PantryItem(robot,isSim):
  #liftRobotTorso(robot,isSim,SHELF2)
  #rospy.sleep(1) # might not wait long enough to lift torso.
  
  # Stop octo map
  StopNStart_octomap()


  # call the table searcher server for search for table and objects.
  rospy.wait_for_service('table_searcher/segment_objects')
  tableSearcher = rospy.ServiceProxy('table_searcher/segment_objects', SegmentObjects)
   
  try:
    tableresult = tableSearcher()
  except rospy.ServiceException as exc:
    print("Service did not process request: " + str(exc))
    return False, None 
  print("Done with table and objects segment.") 
  
  #start octo_map
  StopNStart_octomap()

  #picking a random object for now untill we implement findMatching_pantry
  target_object_pose = tableresult.segmented_objects.objects[0]

  for obj in tableresult.segmented_objects.objects:
    tf_helper.pubTransform("obj_pos_rail", ((obj.centroid.x, obj.centroid.y, obj.centroid.z), \
                  (obj.orientation.x, obj.orientation.y, obj.orientation.z, obj.orientation.w)))
  
  return True, target_object_pose

def findGrocery_Placement(robot,isSim,target_object_pose):

  #liftRobotTorso(robot,isSim,SHELF2)

  if isSim:
    target_object_pose[0][1] = target_object_pose[0][1] - .2
    # left_of_target_object_pose = target_object_pose[0][1] + .13
    # right_of_target_object_pose = target_object_pose[0][1] - .13
    # tf_helper.pubTransform("place_pos_rail", ((target_object_pose[0][0], target_object_pose[0][1] , target_object_pose[0][2]), \
    #               (target_object_pose[1][0], target_object_pose[1][1], target_object_pose[1][2], target_object_pose[1][3])))

    placement_pose = getMatrixFromQuaternionAndTrans( [target_object_pose[1][0], target_object_pose[1][1], target_object_pose[1][2], target_object_pose[1][3]] , \
                        [target_object_pose[0][0], target_object_pose[0][1], target_object_pose[0][2]])
  # else:
    #  # show the position tf in rviz
    # tf_helper.pubTransform("place_pos_rail", ((target_object_pose.centroid.x, left_of_target_object_pose, target_object_pose.centroid.z), \
    #                 (target_object_pose.orientation.x, target_object_pose.orientation.y, target_object_pose.orientation.z, target_object_pose.orientation.w)))

    # placement_pose = getMatrixFromQuaternionAndTrans([target_object_pose.orientation.x, target_object_pose.orientation.y, target_object_pose.orientation.z, target_object_pose.orientation.w], \
    #                        [target_object_pose.centroid.x, left_of_target_object_pose, target_object_pose.centroid.z])
  
  return True, placement_pose

def move_to_Placement(robot, target_placement_pos, tf_helper):
  current_gripper_pos = tf_helper.getPoseMat('/base_link', '/gripper_link')
  current_gripper_trans = tf_helper.getTransform('/base_link', '/gripper_link')
  #change gripper rotation 
  R_t = align_vectors(current_gripper_pos[:,2][:3], [0,0,1])
  target_gripper_pos = current_gripper_pos.dot(R_t)
  # R_t = np.identity(4)
  # R_t[:,3] = current_gripper_pos[:,3]
  # target_gripper_pos = R_t
  # do gripper trans lation 
  T = np.linalg.inv(target_gripper_pos).dot(target_placement_pos) #this is starting at 000
  placement_pos = target_gripper_pos.dot(T)
  # Move to placement point 
  #placement_pos = target_placement_pos
  for tableangle in [0.0, 0.7853975, 1.570795, 2.3561925, 3.14159, 3.9269875, 4.712385, 5.4977825]:
    print("check table angle")
    rotationInZ = np.identity(4)
    rotationInZ[:3,:3] = R.from_rotvec(tableangle * np.array([0,0,1])).as_dcm()
    placement_pos = placement_pos.dot(rotationInZ)
    placement_trans = getTransformFromPoseMat(placement_pos)
    tf_helper.pubTransform("trans_place_pos_rail", ((placement_trans[0][0], placement_trans[0][1] , placement_trans[0][2]), \
                (placement_trans[1][0], placement_trans[1][1], placement_trans[1][2], placement_trans[1][3])))


    placement_ik_result = robot.solve_ik_sollision_free_in_base(placement_trans, 30) #30
    if placement_ik_result != None:
      break  
  if placement_ik_result == None:
    return False


  raw_input("found solution to pick")
  # move to pre grasp pose
  plan = robot.planto_pose(placement_trans)
  robot.display_trajectory(plan)
  raw_input("ready display placement")
  robot.execute_plan(plan)
  return True
  

  

if __name__=='__main__':
  
  object_name = "sugar_box_simp" #"sugar_box_simp"
  isSim = True

  rospy.init_node('simtest_node')
  robot = Fetch_Robot(sim=isSim)
  tf_helper = TF_Helper()

  #Add object model to allow it to find grasping points. 
  base = pandactrl.World(camp=[700,300,1400], lookatp=[0,0,0])
  this_dir, this_filename = os.path.split(os.path.realpath(__file__))   
  objpath = os.path.join(os.path.split(this_dir)[0], "objects", object_name + ".stl") 
  handpkg = fetch_grippernm  #SQL grasping database interface 
  gdb = db.GraspDB()   #SQL grasping database interface
  planner = RegripPlanner(objpath, handpkg, gdb)

  grocery_planner = Grocery_cluster()
  grocery_planner.cluster()
  

 # add objects into planning scene
  add_table(robot, tf_helper)
  print("Added objects")

  #This gets the transfrom from the object to baselink 
  object_pose_in_base_link = tf_helper.getTransform('/base_link', '/' + object_name) # add the object into the planning scene
  object_pose_in_base_link = tf_helper.getTransform('/base_link', '/' + "sugar_box") # add the object into the planning scene
  #robot.addCollisionObject(object_name + "_collision", object_pose_in_base_link, objpath,size_scale = 1)
  raw_input("chedk")
  
  #obj_pose_in_base_link = findMatching_PantryItem(robot, isSim)

  result, placement_pose = findGrocery_Placement(robot,isSim, object_pose_in_base_link) #Object_pose_in_base_link
  print("finding placement")
  if result:
    print("---SUCCESS---")
  else:
    print("---FAIL")
  
  set_ArmPos(robot)
  result = move_to_Placement(robot, placement_pose, tf_helper)
  print("Move to placement")
  if result:
    print("---SUCCESS---")
  else:
    print("---FAIL")

  print("Exit")
  exit()

  """ The code below allows the robot to grasp the object and move to a defult config"""
  # extract the list of init grasps
  result, init_grasps = getInitGrasps(gdb, object_name=object_name)
  print("get init grasps")
  if result:
    print("---SUCCESS---")
  else:
    print("---FAIL")
  print("Number of init grasp: ", len(init_grasps))
  # #raw_input("Press enter to continue")
  # exit()

  result, init_grasp_transform_in_object_frame, init_jawwidth = grasp_object(planner, object_pose_in_base_link, given_grasps = init_grasps, object_name=object_name)
  #result, init_grasp_transform_in_object_frame, init_jawwidth = grasp_object(planner, object_pose_in_base_link, object_name=object_name)
  print("grasping object")
  if result:
    print("---SUCCESS---")
  else:
    print("---FAILURE---")
    exit()

  #raw_input("Ready to pick up object?")
  
  result = pickup(tf_helper, 0.15)
  print "pick up object"
  if result:
    print "---SUCCESS---"
  else:
    print "---FAILURE---"
    exit()

  # move to inital arm pos
  set_ArmPos(robot)
  raw_input("finished moving to defult art pose?")


