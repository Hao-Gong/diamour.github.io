---
layout: post
title:  "ROS：基于rviz和urdf的3D模型制作与控制"
categories: ROS
tags: ROS RVIZ Robot 经验分享
author: Hao
description: 基于ROS系统3D建模，并且控制
---
### 写这篇文章的动力
想要明年参加的KIT的KAMARO社团的比赛，就要尽快熟悉和理解ROS系统的各项功能，里面的老成员都已经在设团呆了3-4年了才担当主力，我必须加把劲。说真的这边的社团节奏好慢，一个礼拜也就只有一个周四晚上实验室开门。中国人做什么都追求速度，德国人不紧不慢的态度确实有点不习惯。前面已经完成了通过电脑控制树莓派进行串口操作，来控制单片机对LED的控制(下次PO代码和ROS局域网分机控制的注意点)。所以现在尝试一下ROS下面RVIZ的机器人小车的建模和事实控制。RVIZ相对于GAZEBO来说，GAZEBO还能在虚拟环境下进行整个虚拟物理环境（world）的搭建，并且设置虚拟传感器接收数据，这样让我有一个大胆的想法，就是在全虚拟的环境中利用Pytorch进行自动驾驶的实验。

好了不讲这么多了，先赶紧开始今天内容。

#### urdf介绍：

urdf(Unified Robot Description Format)按照字面理解就是统一的机器人描述格式，它本质是一种XML结构语言。它有什么好处呢，可以快速的搭建机器人的各个零部件并且默认设置了tf(transformation tree)坐标树进行坐标转换。假设没有这个的话，光在rviz创建一个长方体就要几十行的C++/Python代码(这个蠢事我已经做过了)。来，现在我们开始一步步操作保证运行：

[urdf官网传送门](http://wiki.ros.org/urdf)，先安装一点东西，支持urdf格式：

	sudo apt-get install ros-kinetic-urdf-tutorial //你是什么版本ros就填什么
	sudo apt-get install liburdfdom-tools

首先你在ROS工程文件src下创建catkin_pkg，我们就取名model好了:

	catkin_create_pkg model std_msgs rospy roscpp urdf tf
	cd model 
	mkdir urdf

这边我展示一下我的tree：

	.
	├── CMakeLists.txt
	├── include
	│   └── model
	├── package.xml
	├──scripts
	│   └── controler.py
	├── src
	│   └── robot.cpp
	└── urdf
	    ├── model.urdf
	    └── model_xacro.xacro

然后创建model.urdf文件：

	cd urdf
	gedit model.urdf
	
#### model.urdf代码：
```
<robot name="test_robot">
  <link name="base_link">
    <visual>
       <geometry>
         <box size="0.2 0.3 0.1"/>
       </geometry>
	//rpy的含义: roll, pitch, yaw
       <origin rpy="0 0 0" xyz="0 0 0.05"/> 
       <material name="white">
         <color rgba="1 1 1 1"/>
       </material>
    </visual>
  </link>

  <link name="wheel_1">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.05"/>
      </geometry>
      <origin rpy="0 1.57075 0" xyz="0.1 0.15 0"/>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
  </link>

  <link name="wheel_2">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.05"/>
      </geometry>
      <origin rpy="0 1.57075 0" xyz="-0.1 0.15 0"/>
      <material name="black"/>
    </visual>
  </link>

  <link name="wheel_3">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.05"/>
      </geometry>
      <origin rpy="0 -1.57075 0" xyz="0.1 -0.15 0"/>
      <material name="black"/>
    </visual>
  </link>

  <link name="wheel_4">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.05"/>
      </geometry>
      <origin rpy="0 -1.57075 0" xyz="-0.1 -0.15 0"/>
      <material name="black"/>
    </visual>
  </link>

  <joint name="joint_base_wheel1" type="fixed">
    <parent link="base_link"/>
    <child link="wheel_1"/>
  </joint>

  <joint name="joint_base_wheel2" type="fixed">
    <parent link="base_link"/>
    <child link="wheel_2"/>
  </joint>

  <joint name="joint_base_wheel3" type="fixed">
    <parent link="base_link"/>
    <child link="wheel_3"/>
  </joint>

  <joint name="joint_base_wheel4" type="fixed">
    <parent link="base_link"/>
    <child link="wheel_4"/>
  </joint>
</robot>
```

其中，link就是部件的关系，设置了初始化的时候的部件的位置，joint是关节的意思，parent是继承的参考坐标部件，child是参考parent坐标的子部件，这样生成的tf坐标树能够自动转换坐标系。

写完代码后运行，检查模型是否正确：

	check_urdf model.urdf

显示：

	robot name is: test_robot
	---------- Successfully Parsed XML ---------------
	root Link: base_link has 4 child(ren)
	    child(1):  wheel_1
	    child(2):  wheel_2
	    child(3):  wheel_3
	    child(4):  wheel_4

也可以生成图形pdf文档：

	urdf_to_graphiz model.urdf

pdf显示：

![urdf1](/assets/images/ros/urdf1.png)

然后可以通过rviz就来看一下：

	source /opt/ros/kinetic/setup.bash //运行urdf_tutorial需要source到ros安装包
	roslaunch urdf_tutorial display.launch model:=model.urdf

rviz就自己启动了：

![urdf2](/assets/images/ros/urdf2.png)


#### 现在我们来说一下怎么控制这个模型在rviz里面运动：
其实这个就是node和topic的操作了，上代码：
#### robot.cpp代码：
```
#include <string>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <tf/transform_broadcaster.h>
#include "std_msgs/String.h"
#define STEP 0.1
#define ANGLE 0.1

struct model_position{
    float x;
    float y;
    float z;
    float roll;
    float pitch;
    float yaw;
};

struct model_param{
    model_position position;
    bool flag;
};

class robot_model
{
    public:
        robot_model();
        model_param m_param;
        int model(tf::TransformBroadcaster broadcaster,ros::Publisher joint_pub,geometry_msgs::TransformStamped odom_trans,sensor_msgs::JointState joint_state);
};

void ControlCallback(const std_msgs::String::ConstPtr& msg);

robot_model rm;

int main(int argc, char** argv) {
    ros::init(argc, argv, "state_publisher");
    ros::NodeHandle n;
    ros::Rate r(20);

    ros::Publisher joint_pub = n.advertise<sensor_msgs::JointState>("joint_states", 1);
    ros::Subscriber sub = n.subscribe("/controler", 1000, ControlCallback);
    tf::TransformBroadcaster broadcaster;

    // message declarations
    geometry_msgs::TransformStamped odom_trans;
    sensor_msgs::JointState joint_state;
    odom_trans.header.frame_id = "odom";
    odom_trans.child_frame_id = "base_link";


    while (ros::ok())
    {
        if(1){
            joint_state.header.stamp = ros::Time::now();
            joint_state.name.resize(4);
            joint_state.position.resize(4);
            joint_state.name[0] ="base_to_wheel1";
            joint_state.position[0] = 0;
            joint_state.name[1] ="base_to_wheel2";
            joint_state.position[1] = 0;
            joint_state.name[2] ="base_to_wheel3";
            joint_state.position[2] = 0;
            joint_state.name[3] ="base_to_wheel4";
            joint_state.position[3] = 0;
            rm.model(broadcaster,joint_pub,odom_trans,joint_state);
            odom_trans.transform.rotation = tf::createQuaternionMsgFromYaw(0);
            rm.m_param.flag=0;
        }
        ros::spinOnce();
        r.sleep();
    }

    return 0;
}

//class function definition
robot_model::robot_model(){
    m_param.position.x=0;
    m_param.position.y=0;
    m_param.position.z=0;
    m_param.position.roll=0;
    m_param.position.pitch=0;
    m_param.position.yaw=0;
    m_param.flag=0;
}


void ControlCallback(const std_msgs::String::ConstPtr& msg)
{
  ROS_INFO("I heard: [%s]", msg->data.c_str());

  std::string accept_str = msg->data;
  rm.m_param.flag=0;
  if(accept_str=="forwards"){
        rm.m_param.position.y+=STEP*cos(rm.m_param.position.yaw);
        rm.m_param.position.x+=STEP*sin(-rm.m_param.position.yaw);
        ROS_INFO("forwards");
   }else if(accept_str=="backwards"){
        rm.m_param.position.y+=-STEP*cos(rm.m_param.position.yaw);
        rm.m_param.position.x+=-STEP*sin(-rm.m_param.position.yaw);
        ROS_INFO("backwards");
   }else if(accept_str=="right"){
        rm.m_param.position.yaw+=-ANGLE;
        ROS_INFO("right");
   }else if(accept_str=="left"){
        rm.m_param.position.yaw+=ANGLE;
        ROS_INFO("left");
   }else if(accept_str=="origin"){
        rm.m_param.position.x=0.0;
        rm.m_param.position.y=0.0;
        rm.m_param.position.z=0.0;
	rm.m_param.position.yaw=0.0;
        ROS_INFO("set origin");}
}

int robot_model::model(tf::TransformBroadcaster broadcaster,ros::Publisher joint_pub,geometry_msgs::TransformStamped odom_trans,sensor_msgs::JointState joint_state)
{

        // update transform
        // (moving in a circle with radius)
        odom_trans.header.stamp = ros::Time::now();
        odom_trans.transform.translation.x = rm.m_param.position.x;
        odom_trans.transform.translation.y = rm.m_param.position.y;
        odom_trans.transform.translation.z = 0.0;
        odom_trans.transform.rotation = tf::createQuaternionMsgFromYaw(rm.m_param.position.yaw);
        //send the joint state and transform
        joint_pub.publish(joint_state);
        broadcaster.sendTransform(odom_trans);
}
```

我们要记得在CMakeLists.txt里面加上，让C++文件可以运行：

	add_executable(robot src/robot.cpp)
	target_link_libraries(robot ${catkin_LIBRARIES})


#### controler.py代码：

在scripts里面加入

	gedit controler.py
	sudo chmod +x controler.py //这一步不要忘记

这里我们使用tkinter界面来控制，你可能需要pip3来安装(对了，首先你的rospy要支持python3,官网的教程默认是python2.7，下一次有机会讲一下)，并且输入条支持键盘检测控制：

```
#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from tkinter import *


rospy.init_node('model_controler', anonymous=0)
pub = rospy.Publisher('/controler', String, queue_size=10)
rate = rospy.Rate(100)  # 10hz

class ui_process(Frame):
    def __init__(self, master):

        master.geometry("400x150+600+300")
        master.resizable(width=False, height=False)

        fm0 = Frame(master)
        Label(fm0,text='Rviz_3D_model_controler by gong').pack(side=LEFT,anchor=W, fill=X, expand=YES)
        fm0.pack(side=TOP, fill=BOTH, expand=YES)

        #frame
        fm1=Frame(master)
        #botton
        Button(fm1, text="for+left", command=self.forwards_left_press).pack(side=LEFT,anchor=W, fill=X, expand=YES)
        Button(fm1, text="forwards(key:w)", command=self.forwards_press).pack(side=LEFT,anchor=W, fill=X, expand=YES)
        Button(fm1, text="for+right", command=self.forwards_right_press).pack(side=LEFT, anchor=W, fill=X, expand=YES)
        fm1.pack(side=TOP, fill=BOTH, expand=YES)

        fm2 = Frame(master)
        Button(fm2, text="left(key:a)", command=self.left_press).pack(side=LEFT,anchor=W, fill=X, expand=YES)
        Button(fm2, text="      origin      ", command=self.origin).pack(side=LEFT,anchor=W, fill=X, expand=YES)
        Button(fm2, text="right(key:d)", command=self.right_press).pack(side=LEFT,anchor=W, fill=X, expand=YES)
        fm2.pack(side=TOP, fill=BOTH, expand=YES)

        fm3 = Frame(master)
        Button(fm3, text="back+left", command=self.backwards_left_press).pack(side=LEFT,anchor=W, fill=X, expand=YES)
        Button(fm3, text="backwards(key:s)", command=self.backwards_press).pack(side=LEFT,anchor=W, fill=X, expand=YES)
        Button(fm3, text="back+right", command=self.backwards_right_press).pack(side=LEFT,anchor=W, fill=X, expand=YES)
        fm3.pack(side=TOP, fill=BOTH, expand=YES)

        fm4= Frame(master)
        #keyboard listening
        Label(fm4, text='Input from keyboard:').pack(side=LEFT,anchor=W, fill=X, expand=YES)
        self.keyboard_listening=Entry(fm4)
        self.keyboard_listening.pack(side=LEFT,anchor=W, fill=X, expand=YES)
        self.keyboard_listening.bind('<Key>',self.key_detect)
        Button(fm4, text=" clear", command=self.keyboard_clear).pack(side=LEFT, anchor=W, fill=X, expand=YES)
        fm4.pack(side=TOP, fill=BOTH, expand=YES)
        master.mainloop()

    def forwards_left_press(self):
        pub.publish("left")
        pub.publish("forwards")

    def forwards_right_press(self):
        pub.publish("right")
        pub.publish("forwards")

    def backwards_left_press(self):
        pub.publish("left")
        pub.publish("backwards")

    def backwards_right_press(self):
        pub.publish("right")
        pub.publish("backwards")

    def origin(self):
        pub.publish("origin")

    def left_press(self):
        pub.publish("left")

    def right_press(self):
        pub.publish("right")

    def forwards_press(self):
        pub.publish("forwards")

    def backwards_press(self):
        pub.publish("backwards")

    def key_detect(self,Event):
        #print("key:"+Event.char)
        if Event.char=="w":
            pub.publish("forwards")
        elif Event.char=="s":
            pub.publish("backwards")
        elif Event.char == "a":
            pub.publish("left")
        elif Event.char == "d":
            pub.publish("right")

    def keyboard_clear(self):
        self.keyboard_listening.delete('0', END)


def main():
    root = Tk()
    ui=ui_process(master=root)
    #ui.mainloop()

if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass
```
界面图片：

![urdf3](/assets/images/ros/urdf3.png)

#### roslannch文件，建议建一个roslaunch_pkg，专门放launch文件，这里model.launch：

	<launch>
	    <node pkg="rviz" type="tk.py" name="tk"/>
	    <node pkg="rviz" type="model" name="model"/>
	</launch>

#### 最后我说一下怎么连贯的运行，你可以写成一个sh文件：
	source devel/setup.bash
	roslaunch launch model.launch #我的roslaunch_pkg名字就叫launch
	source /opt/ros/kinetic/setup.bash
	cd src/model/urdf
	roslaunch urdf_tutorial display.launch model:=model3.urdf

然后你就可以自由自在的控制了！！！

[我的github工程](https://github.com/diamour/ros_network_simple)
##### 版权归@Hao所有，转载标记来源。
	
