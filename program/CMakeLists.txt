cmake_minimum_required(VERSION 2.8) 
project( hw2 ) 
find_package( OpenCV REQUIRED ) 
add_executable( stitching main.cpp feature.cpp image_stitching.cpp ) 
target_link_libraries( stitching ${OpenCV_LIBS} )
