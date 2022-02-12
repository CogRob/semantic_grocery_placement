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
from object_segmentation.srv import Object_seg_result
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




def add_shelfs(robot, tf_helper):
    """
    add the table into the planning scene
    """
    # get table pose
    #table_transform = tf_helper.getTransform('/base_link', '/Table')
    # add it to planning scene
    robot.addCollisionTable("shelf1", 0.7761257886886597+.13, 0.2525942027568817 -.15, 1.2541508674621582 -.09, \
        0.0, 0.0, 0.8010766954545328 -.2, 0.5985617161159289, \
        1.5, .5, 0.05) #.7 1.5, .06  width(x), lenght(y), hight(z )

    robot.addCollisionTable("shelf2", 0.7761257886886597+.13, 0.2525942027568817 -.15, 1.2541508674621582 -.55, \
      0.0, 0.0, 0.8010766954545328 -.2, 0.5985617161159289, \
      1.5, .5, 0.05) #.7 1.5, .06  width(x), lenght(y), hight(z )

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

    print("Going through this many grasp pose: " ,len(gripper_pos_list))
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
            print('check on grasp ', i)
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
        print("please add the object name to the table first!!")
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
  res = robot.getjointNamesNValues()
  #defult joint with arm streached out. 
  defult_joints = [ 1.4855620731445311, 0.7087086353057861, 3.1121989442993163, 0.8934669902954102, -0.08984714265785218, -0.14463894778442382, 3.1516605679718017]
  
  #defult joint with gripper compressed
  #defult_joints = [1.6654212345214843, 0.6676746520751953, 3.110664959161377, 1.6581564357910157, -1.7185513031097412, -1.6759356778259278, -2.5796753104003907]

  plan = robot.planto_joints(defult_joints)
  robot.display_trajectory(plan)
  raw_input("ready to execute defult arm pos")
  robot.execute_plan(plan)

  #robot.openGripper()
  raw_input("ready to grasp obj?")
  robot.closeGripper()


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

def matchCentroidNBoundingBox(tableresult,classifier, target_name):
  #Intrensic cam matrix 
  #cam_K: [527.3758609346917, 0.0, 326.6388366771264, 0.0, 523.6181455086474, 226.4866800158784, 0.0, 0.0, 1.0]
  cam_K = np.array([ [527.3758609346917, 0.0, 326.6388366771264], [ 0.0, 523.6181455086474, 226.4866800158784],
         [ 0.0, 0.0, 1.0] ])
  box = None
  for i in range(0, len(classifier.Labels)):
    if classifier.Labels[i] == target_name:
      box = classifier.boxes[i].box

  if not box:
    print("Failed to find matching bounding box")
    return False, -1

  target_transfom = tf_helper.getPoseMat( '/head_camera_depth_optical_frame', '/base_link')
  for obj in tableresult.segmented_objects.objects:
    obj_trans = np.array( [obj.centroid.x, obj.centroid.y, obj.centroid.z, 1])
    obj_trans = target_transfom.dot(obj_trans)
    obj_trans = obj_trans[:3]
    point = cam_K.dot(obj_trans)
    twoD_point = point[0]/point[2], point[1]/point[2]
    if twoD_point[1] > box[1] and twoD_point[1] < box[3]:
      if twoD_point[0] > box[0] and twoD_point[0] < box[2]:
        return True, obj 

  print("Failed to find matching bounding box")
  return False, -1 

def findMatching_PantryItem(robot,isSim, target_object_name):
  #liftRobotTorso(robot,isSim,SHELF2)
  rospy.sleep(1) # might not wait long enough to lift torso.

  # Stop octo map
  StopNStart_octomap()

  """ 
   Object centroid detection
   """
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
  
  """ 
  Target object classification
  """
  #Object Filter 
  rospy.wait_for_service("object_filter")
  obj_classifier = rospy.ServiceProxy('object_filter',Object_seg_result)
  while True:
    try:
      classifier_result = obj_classifier()  
    except rospy.ServiceException as e:
      print ("Failed to find object: %s"%e)  
    break
  print("Object detection finished")

  return matchCentroidNBoundingBox(tableresult ,classifier_result,target_object_name)

# Know informaiton about test objects from Mesh model. 
PringalsCan = (.13, .05)
def findGrocery_Placement(robot,isSim,target_object_pose, object_size):



  #liftRobotTorso(robot,isSim,SHELF2)
  rospy.sleep(1) # might not wait long enough to lift torso.


  left_of_target_object_pose = target_object_pose.centroid.y + object_size[0]
  right_of_target_object_pose = target_object_pose.centroid.y - object_size[0]
  up_of_target_placement_pose = target_object_pose.centroid.z + object_size[1]
  tf_helper.pubTransform("obj_pos_rail", ((target_object_pose.centroid.x, target_object_pose.centroid.y, target_object_pose.centroid.z), \
                  (target_object_pose.orientation.x, target_object_pose.orientation.y, target_object_pose.orientation.z, target_object_pose.orientation.w)))

  placement_trans = ((target_object_pose.centroid.x, right_of_target_object_pose,up_of_target_placement_pose), \
                  (target_object_pose.orientation.x, target_object_pose.orientation.y, target_object_pose.orientation.z, target_object_pose.orientation.w))
  target_placement_pos = getMatrixFromQuaternionAndTrans(placement_trans[1],placement_trans[0])

   # show the position tf in rviz
  tf_helper.pubTransform("place_pos_rail", ( placement_trans[0], placement_trans[1]) )

 
  return True, target_placement_pos

def move_to_placement(robot, target_placement_pos, tf_helper):
  current_gripper_pos = tf_helper.getPoseMat('/base_link', '/gripper_link')
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


  raw_input("found solution to place")
  # move to pre grasp pose
  plan = robot.planto_pose(placement_trans)
  robot.display_trajectory(plan)
  raw_input("ready display placement")
  robot.execute_plan(plan)
  return True


if __name__=='__main__':
  
  object_name = "sugar_box_simp"
  isSim = False

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
  

  #add objects into planning scene
  add_shelfs(robot, tf_helper)
  print("Added objects")

  #This gets the transfrom from the object to baselink 
  #object_pose_in_base_link = tf_helper.getTransform('/base_link', '/' + object_name) # add the object into the planning scene
  #robot.addCollisionObject(object_name + "_collision", object_pose_in_base_link, objpath,size_scale = 1)

  set_ArmPos(robot)

  result, matchin_obj_pose_base_link = findMatching_PantryItem(robot,isSim,'pringles_red')
  print("Finding matching pantry Item")
  if result:
    print ("---SUCCESS---")
  else:
    print("----FAIL---")
    exit()

  result,target_placement_pos = findGrocery_Placement(robot,isSim, matchin_obj_pose_base_link, PringalsCan)
  print("Finding Placement")
  if result:
    print ("---SUCCESS---")
  else:
    print("----FAIL---")
    exit()
  
  result = move_to_placement(robot, target_placement_pos, tf_helper)
  if result:
    print ("---SUCCESS---")
  else:
    print("----FAIL---")
    exit()
  
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
  print("pick up object")
  if result:
    print("---SUCCESS---")
  else:
    print("---FAILURE---")
    exit()

  # move to inital arm pos
  set_ArmPos(robot)
  raw_input("finished moving to defult art pose?")


