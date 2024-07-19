# Semantic Grocery Placement

This repositor is for the 2022 CASE paper [Putting away the Groceries with Precise Semantic Placements](https://ieeexplore.ieee.org/document/9926691).

Part of this project is in conjunction with a pervious project and [paper](https://www.researchgate.net/publication/369075349_Place-and-Pick-Based_Re-grasping_Using_Unstable_Placement) 

# Installation 
This code base is heavily based on the ROS architecture. Please see more information [here](http://wiki.ros.org/ROS/Tutorials) if you are unfamiliar. 

This repositor is a package to be added in the workspace/src repo available here: [Fixturless_fixturiing_place_and_pick](https://github.com/jih189/fixtureless_fixturing_place_and_pick_re-grasping/tree/sim-melodic)
Please download and follow the instructions in the sim-melodic branch the repo above. Then add this repository as a package with in the workspace. 

Install [rail_segmentation](https://github.com/GT-RAIL/rail_segmentation) package to the workspace/src directory created above. Be sure to run catkin_make. 

To test with in the same simulation environment, go to the folder simulation fix. ADD the 'scene' folder to workspace/src/fetch_copeliasim/, to replace the scene folder there. Next add the objects_description folder to workspace/src and replace the folder there.

# Running Code
To run in simulation or on a real Fetch robot, navigate to the `scripts` folder. The files `sim_test.py` and `real_test.py` are both used accordingly. 



