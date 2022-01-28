#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

#include <semantic_grocery_placement/StopOctoMap.h>

ros::Publisher pub;
// pcl::PCLPointCloud2 cloud;
sensor_msgs::PointCloud2 output;
bool hasfreepoint;

bool cleanOcto(semantic_grocery_placement::StopOctoMap::Request &req,
               semantic_grocery_placement::StopOctoMap::Response &res)
{
  hasfreepoint = !hasfreepoint;
  ROS_INFO("stop updating the Octo map");
  return true;
}

void 
cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{
  // Create a container for the data.
  // sensor_msgs::PointCloud2 output;
  // pcl_conversions::toPCL(*input, cloud);

  // pcl_conversions::fromPCL(cloud, output);
  if(!hasfreepoint){
    output = sensor_msgs::PointCloud2(*input);
  }

 //Filter output for shelf
   // convert Sensor msgs to pcl
  // pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_filter_output(new pcl::PointCloud<pcl::PointXYZ>); 
  // pcl::fromROSMsg(output, *pcl_filter_output);

    
  //   // filter with pcl 
  // pcl::PassThrough<pcl::PointXYZ> pass;
  // pass.setInputCloud (pcl_filter_output);
  // pass.setFilterFieldName ("y"); //y
  //   // "y"  first value increases view upper y direction. second inccrease view in the lower y direction. 
  // pass.setFilterLimits (0.001, 0.9);
  // pass.filter (*pcl_filter_output);
  //   // convert pcl back to sensor msgs pointc2. 
  // sensor_msgs::PointCloud2 filter_output;  
  // pcl::toROSMsg(*pcl_filter_output, filter_output);


  

  // Publish the data.
  pub.publish (output);
  //pub.publish (filter_output);
}

int
main (int argc, char** argv)
{

  hasfreepoint = false;

  // Initialize ROS
  ros::init (argc, argv, "octomap_filter");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/head_camera/depth_downsample/points", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("/head_camera/depth_downsample/free_space_points", 1);

  // Create a ROS server
  ros::ServiceServer service = nh.advertiseService("stop_octo_map", cleanOcto);

  // Spin
  ros::spin ();
}