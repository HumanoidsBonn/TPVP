#include <windows.h>
#include <iostream>
#include <random>
#include <thread>
#include <atomic>
#include <chrono>
typedef unsigned long long pop_t;

using namespace std;

#include "Share_Data.hpp"
#include "View_Space.hpp"
#include "Information.hpp"

//Virtual_Perception_3D.hpp
void precept_thread_process(int i, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, octomap::point3d* _origin, octomap::point3d* _end, Eigen::Matrix4d* _view_pose_world, octomap::ColorOcTree* _ground_truth_model, Share_Data* share_data);

class Perception_3D {
public:
	Share_Data* share_data;
	octomap::ColorOcTree* ground_truth_model;
	int full_voxels;

	Perception_3D(Share_Data* _share_data) {
		share_data = _share_data;
		ground_truth_model = new octomap::ColorOcTree(share_data->ground_truth_resolution);
		for (octomap::ColorOcTree::leaf_iterator it = share_data->ground_truth_model->begin_leafs(), end = share_data->ground_truth_model->end_leafs(); it != end; ++it) {
			ground_truth_model->setNodeValue(it.getKey(), it->getLogOdds(), true);
			ground_truth_model->setNodeColor(it.getKey(), it->getColor().r, it->getColor().g, it->getColor().b);
		}
		if (share_data->has_table) {
			for (double x = share_data->object_center_world(0) - 0.2; x <= share_data->object_center_world(0) + 0.2; x += share_data->ground_truth_resolution)
				for (double y = share_data->object_center_world(1) - 0.2; y <= share_data->object_center_world(1) + 0.2; y += share_data->ground_truth_resolution) {
					double z = share_data->min_z_table;
					ground_truth_model->setNodeValue(x, y, z, ground_truth_model->getProbHitLog(), true);
					ground_truth_model->setNodeColor(x, y, z, 0, 0, 255);
				}
			//ground_truth_model->write(share_data->save_path + "/GT_table.ot");
		}
		ground_truth_model->updateInnerOccupancy();
		//统计GT模型中体素个数
		full_voxels = 0;
		for (octomap::ColorOcTree::leaf_iterator it = ground_truth_model->begin_leafs(), end = ground_truth_model->end_leafs(); it != end; ++it) {
			full_voxels++;
		}
	}

	~Perception_3D() {
		delete ground_truth_model;
	}

	bool precept(View* now_best_view) {
		//如果使用保存的点云加速的话，尝试读取
		if (share_data->use_saved_cloud) {
			int view_id = now_best_view->id;
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
			//pcl::PointCloud<pcl::PointXYZRGB>::Ptr no_table(new pcl::PointCloud<pcl::PointXYZRGB>);
			string view_cloud_file_path = share_data->gt_path + "/GT_points_" + share_data->look_at_group_str + "/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "/cloud_view" + to_string(view_id) + ".pcd";
			//string view_cloud_notable_file_path = share_data->gt_path + "/GT_points_" + share_data->look_at_group_str + "/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "/cloud_notable_view" + to_string(view_id) + ".pcd";
			//if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(view_cloud_file_path, *cloud) != -1 && pcl::io::loadPCDFile<pcl::PointXYZRGB>(view_cloud_notable_file_path, *no_table) != -1) {
			if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(view_cloud_file_path, *cloud) != -1) {
				cout << "Load view clouds success. Use saved cloud to speed up evaluation." << endl;
				//记录当前采集点云
				share_data->vaild_clouds++;
				share_data->clouds.push_back(cloud);
				//旋转至世界坐标系
				//share_data->clouds_notable.push_back(no_table);
				*share_data->cloud_final += *cloud;
				return true;
			}
			else {
				cout << "Load view cloud failed. Use virtual perception." << endl;
			}
		}
		//如果不使用保存数据或读取失败，在线生成
		double now_time = clock();
		//创建当前成像点云
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_parallel(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_parallel->is_dense = false;
		cloud_parallel->points.resize(full_voxels);
		//获取视点位姿
		Eigen::Matrix4d view_pose_world;
		now_best_view->get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
		view_pose_world = (share_data->now_camera_pose_world * now_best_view->pose.inverse()).eval();
		//检查视点的key
		octomap::OcTreeKey key_origin;
		bool key_origin_have = ground_truth_model->coordToKeyChecked(now_best_view->init_pos(0), now_best_view->init_pos(1), now_best_view->init_pos(2), key_origin);
		if (key_origin_have) {
			octomap::point3d origin = ground_truth_model->keyToCoord(key_origin);
			//遍历每个体素
			octomap::point3d* end = new octomap::point3d[full_voxels];
			octomap::ColorOcTree::leaf_iterator it = ground_truth_model->begin_leafs();
			for (int i = 0; i < full_voxels; i++) {
				end[i] = it.getCoordinate();
				it++;
			}
			//ground_truth_model->write(share_data->save_path + "/test_camrea.ot");

			//多线程处理
			vector<thread> precept_process;
			for (int i = 0; i < full_voxels; i += share_data->max_num_of_thread) {
				for (int j = 0; j < share_data->max_num_of_thread && i + j < full_voxels; j++) {
					precept_process.push_back(thread(precept_thread_process, i + j, cloud_parallel, &origin, &end[i + j], &view_pose_world, ground_truth_model, share_data));
				}
				for (int j = 0; j < share_data->max_num_of_thread && i + j < full_voxels; j++) {
					precept_process[i + j].join();
				}
			}
			//释放内存
			delete[] end;
			precept_process.clear();
			precept_process.shrink_to_fit();
		}
		else {
			cout << "View out of map.check." << endl;
		}
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		//pcl::PointCloud<pcl::PointXYZRGB>::Ptr no_table(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud->is_dense = false;
		//no_table->is_dense = false;
		cloud->points.resize(full_voxels);
		//no_table->points.resize(full_voxels);
		auto ptr = cloud->points.begin();
		//auto pt = no_table->points.begin();
		int vaild_point = 0;
		//int table_point = 0;
		auto p = cloud_parallel->points.begin();
		for (int i = 0; i < cloud_parallel->points.size(); i++, p++)
		{
			if ((*p).x == 0 && (*p).y == 0 && (*p).z == 0) continue;
			//if ((*p).z > share_data->min_z_table + share_data->ground_truth_resolution) {
			//	(*pt).x = (*p).x;
			//	(*pt).y = (*p).y;
			//	(*pt).z = (*p).z;
			//	(*pt).b = (*p).b;
			//	(*pt).g = (*p).g;
			//	(*pt).r = (*p).r;
			//	table_point++;
			//	pt++;
			//}
			(*ptr).x = (*p).x;
			(*ptr).y = (*p).y;
			(*ptr).z = (*p).z;
			(*ptr).b = (*p).b;
			(*ptr).g = (*p).g;
			(*ptr).r = (*p).r;
			vaild_point++;
			ptr++;
		}
		cloud->width = vaild_point;
		//no_table->width = table_point;
		cloud->height = 1;
		//no_table->height = 1;
		cloud->points.resize(vaild_point);
		//no_table->points.resize(table_point);
		//记录当前采集点云
		share_data->vaild_clouds++;
		share_data->clouds.push_back(cloud);
		//旋转至世界坐标系
		//share_data->clouds_notable.push_back(no_table);
		*share_data->cloud_final += *cloud;
		//cout << "virtual cloud num is " << vaild_point << endl;
		//cout << "virtual cloud table num is " << table_point << endl;
		//cout << "virtual cloud get with executed time " << clock() - now_time << " ms." << endl;
		if (share_data->show) { //显示成像点云
			pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Camera"));
			viewer1->setBackgroundColor(255, 255, 255);
			//viewer1->addCoordinateSystem(0.1);
			viewer1->initCameraParameters();
			viewer1->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
			viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
			viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0, 0, "cloud");

			Eigen::Vector4d O(0, 0, 0, 1);
			O = view_pose_world * O;
			//show camera axis
			Eigen::Vector4d X(0.1, 0, 0, 1);
			Eigen::Vector4d Y(0, 0.1, 0, 1);
			Eigen::Vector4d Z(0, 0, 0.1, 1);
			X = view_pose_world * X;
			Y = view_pose_world * Y;
			Z = view_pose_world * Z;
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
			//show camera frustum
			double line_length = 0.1;
			Eigen::Vector3d LeftTop = project_pixel_to_ray_end_eigen(0, 0, share_data->color_intrinsics, view_pose_world, line_length);
			Eigen::Vector3d RightTop = project_pixel_to_ray_end_eigen(0, share_data->color_intrinsics.height, share_data->color_intrinsics, view_pose_world, line_length);
			Eigen::Vector3d LeftBottom = project_pixel_to_ray_end_eigen(share_data->color_intrinsics.width, 0, share_data->color_intrinsics, view_pose_world, line_length);
			Eigen::Vector3d RightBottom = project_pixel_to_ray_end_eigen(share_data->color_intrinsics.width, share_data->color_intrinsics.height, share_data->color_intrinsics, view_pose_world, line_length);
			Eigen::Vector4d LT(LeftTop(0), LeftTop(1), LeftTop(2), 1);
			Eigen::Vector4d RT(RightTop(0), RightTop(1), RightTop(2), 1);
			Eigen::Vector4d LB(LeftBottom(0), LeftBottom(1), LeftBottom(2), 1);
			Eigen::Vector4d RB(RightBottom(0), RightBottom(1), RightBottom(2), 1);
			double view_color[3] = { 0, 0, 255 };
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(LT(0), LT(1), LT(2)), view_color[0], view_color[1], view_color[2], "O-LT" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(RT(0), RT(1), RT(2)), view_color[0], view_color[1], view_color[2], "O-RT" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(LB(0), LB(1), LB(2)), view_color[0], view_color[1], view_color[2], "O-LB" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(RB(0), RB(1), RB(2)), view_color[0], view_color[1], view_color[2], "O-RB" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(LT(0), LT(1), LT(2)), pcl::PointXYZ(RT(0), RT(1), RT(2)), view_color[0], view_color[1], view_color[2], "LT-RT" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(LT(0), LT(1), LT(2)), pcl::PointXYZ(LB(0), LB(1), LB(2)), view_color[0], view_color[1], view_color[2], "LT-LB" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(RT(0), RT(1), RT(2)), pcl::PointXYZ(RB(0), RB(1), RB(2)), view_color[0], view_color[1], view_color[2], "RT-RB" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(LB(0), LB(1), LB(2)), pcl::PointXYZ(RB(0), RB(1), RB(2)), view_color[0], view_color[1], view_color[2], "LB-RB" + to_string(-1));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "O-LT" + to_string(-1));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "O-RT" + to_string(-1));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "O-LB" + to_string(-1));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "O-RB" + to_string(-1));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "LT-RT" + to_string(-1));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "LT-LB" + to_string(-1));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "RT-RB" + to_string(-1));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "LB-RB" + to_string(-1));

			// show rays
			int point_interval = cloud->points.size() / 5;
			for (int i = 0; i < cloud->points.size(); i+=point_interval) {
				viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z), 0.75, 0.75, 0, "point" + to_string(i));
			}
			viewer1->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_ground_truth");

			while (!viewer1->wasStopped()) {
				viewer1->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		}

		cloud_parallel->points.clear();
		cloud_parallel->points.shrink_to_fit();

		cout <<"Virtual cloud getted with time "<< clock() - now_time<<" ms." << endl;
		return true;
	}
};

void precept_thread_process(int i, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, octomap::point3d* _origin, octomap::point3d* _end, Eigen::Matrix4d* _view_pose_world, octomap::ColorOcTree* _ground_truth_model, Share_Data* share_data) {
	//num++;
	octomap::point3d origin = *_origin;
	Eigen::Matrix4d view_pose_world = *_view_pose_world;
	octomap::ColorOcTree* ground_truth_model = _ground_truth_model;
	pcl::PointXYZRGB point;
	point.x = 0; point.y = 0; point.z = 0;

	//投影检测是否在成像范围内
	Eigen::Vector4d end_3d(_end->x(), _end->y(), _end->z(), 1);
	Eigen::Vector4d vertex = view_pose_world.inverse() * end_3d;

	// z <= 0: behind camera
	if (vertex(2) <= 0) {  
		cloud->points[i] = point;
		return;
	}

	float point_3d[3] = { vertex(0), vertex(1),vertex(2) };
	float pixel[2];
	rs2_project_point_to_pixel(pixel, &share_data->color_intrinsics, point_3d);
	if (pixel[0] < 0 || pixel[0]>share_data->color_intrinsics.width || pixel[1] < 0 || pixel[1]>share_data->color_intrinsics.height) {
		cloud->points[i] = point;
		//cout << "point " << i << " out of image plane." << endl;
		return;
	}
	//反向投影找到终点
	octomap::point3d end = project_pixel_to_ray_end(pixel[0], pixel[1], share_data->color_intrinsics, view_pose_world, share_data->ray_max_dis);
	//octomap::point3d end(_end->x(), _end->y(), _end->z() );
	//cout << "end_2d_to_3d: " << end.x() << " " << end.y() << " " << end.z() << endl;
	//cout << "end_3d: " << end_3d.x() << " " << end_3d.y() << " " << end_3d.z() << endl;

	octomap::OcTreeKey key_end;
	bool key_end_have = ground_truth_model->coordToKeyChecked(*_end, key_end);
	if (!key_end_have) {
		cloud->points[i] = point;
		cout << "Warning: end point out of map." << endl;
		return;
	}

	octomap::point3d direction = end - origin;
	octomap::point3d end_point;
	octomap::OcTreeKey key_end_point;
	//如果方向为空，那不需要射线
	int step[3];
	octomap::point3d direction_norm = direction.normalized();
	for (unsigned int i = 0; i < 3; ++i) {
		// compute step direction_norm
		if (direction_norm(i) > 0.0) step[i] = 1;
		else if (direction_norm(i) < 0.0)   step[i] = -1;
		else step[i] = 0;
	}
	if (step[0] == 0 && step[1] == 0 && step[2] == 0) {
		//cout << "direction_norm is non. skip." << endl;
		return;
	}
	//越过未知区域，找到终点
	bool found_end_point = ground_truth_model->castRay(origin, direction, end_point, true, share_data->ray_max_dis);
	if (!found_end_point) {//未找到终点，无观测数据
		cloud->points[i] = point;
		return;
	}
	if (end_point == origin) {
		//cout << "view in the object. check!" << endl;
		cloud->points[i] = point;
		return;
	}
	//检查一下末端是否在地图限制范围内
	bool key_end_point_have = ground_truth_model->coordToKeyChecked(end_point, key_end_point);
	if (key_end_point_have) {
		////!!!检查末端是否在同一体素内，这里是强制要求去掉命中了临近的surface（非常重要，因为蹭到的表面从离散/像素化是可见的，应该include，所以不要加入下面的）!!!
		//if (key_end_point != key_end) {
		//	//cout << "end point not in the same voxel with end point after ray. hit target's occluded voxel or hit a near voxel due to too high directional density" << endl;
		//	cloud->points[i] = point;
		//	return;
		//}
		//获取末端颜色
		octomap::ColorOcTreeNode* node = ground_truth_model->search(key_end_point);
		if (node != NULL) {
			octomap::ColorOcTreeNode::Color color = node->getColor();
			point.x = end_point.x();
			point.y = end_point.y();
			point.z = end_point.z();
			point.b = color.b;
			point.g = color.g;
			point.r = color.r;
		}
	}
	cloud->points[i] = point;
}

//Global_Path_Planner.hpp
/* Solve a traveling salesman problem on a randomly generated set of
	points using lazy constraints.   The base MIP model only includes
	'degree-2' constraints, requiring each node to have exactly
	two incident edges.  Solutions to this model may contain subtours -
	tours that don't visit every node.  The lazy constraint callback
	adds new constraints to cut them off. */
	// Given an integer-feasible solution 'sol', find the smallest
	// sub-tour.  Result is returned in 'tour', and length is
	// returned in 'tourlenP'.
void findsubtour(int n, double** sol, int* tourlenP, int* tour) {
	{
		bool* seen = new bool[n];
		int bestind, bestlen;
		int i, node, len, start;

		for (i = 0; i < n; i++)
			seen[i] = false;

		start = 0;
		bestlen = n + 1;
		bestind = -1;
		node = 0;
		while (start < n) {
			for (node = 0; node < n; node++)
				if (!seen[node])
					break;
			if (node == n)
				break;
			for (len = 0; len < n; len++) {
				tour[start + len] = node;
				seen[node] = true;
				for (i = 0; i < n; i++) {
					if (sol[node][i] > 0.5 && !seen[i]) {
						node = i;
						break;
					}
				}
				if (i == n) {
					len++;
					if (len < bestlen) {
						bestlen = len;
						bestind = start;
					}
					start += len;
					break;
				}
			}
		}

		for (i = 0; i < bestlen; i++)
			tour[i] = tour[bestind + i];
		*tourlenP = bestlen;

		delete[] seen;
	}
}
// Subtour elimination callback.  Whenever a feasible solution is found,
// find the smallest subtour, and add a subtour elimination constraint
// if the tour doesn't visit every node.
class subtourelim : public GRBCallback
{
public:
	GRBVar** vars;
	int n;
	subtourelim(GRBVar** xvars, int xn) {
		vars = xvars;
		n = xn;
	}
protected:
	void callback() {
		try {
			if (where == GRB_CB_MIPSOL) {
				// Found an integer feasible solution - does it visit every node?
				double** x = new double* [n];
				int* tour = new int[n];
				int i, j, len;
				for (i = 0; i < n; i++)
					x[i] = getSolution(vars[i], n);

				findsubtour(n, x, &len, tour);

				if (len < n) {
					// Add subtour elimination constraint
					GRBLinExpr expr = 0;
					for (i = 0; i < len; i++)
						for (j = i + 1; j < len; j++)
							expr += vars[tour[i]][tour[j]];
					addLazy(expr <= len - 1);
				}

				for (i = 0; i < n; i++)
					delete[] x[i];
				delete[] x;
				delete[] tour;
			}
		}
		catch (GRBException e) {
			cout << "Error number: " << e.getErrorCode() << endl;
			cout << e.getMessage() << endl;
		}
		catch (...) {
			cout << "Error during callback" << endl;
		}
	}
};
class Global_Path_Planner {
public:
	Share_Data* share_data;
	int now_view_id;
	int end_view_id;
	bool solved;
	int n;
	map<int, int>* view_id_in;
	map<int, int>* view_id_out;
	vector<vector<double>> graph;
	double total_shortest;
	vector<int> global_path;
	GRBEnv* env = NULL;
	GRBVar** vars = NULL;
	GRBModel* model = NULL;
	subtourelim* cb = NULL;

	Global_Path_Planner(Share_Data* _share_data, vector<View>& views, vector<int>& view_set_label, int _now_view_id, int _end_view_id = -1) {
		share_data = _share_data;
		now_view_id = _now_view_id;
		end_view_id = _end_view_id;
		solved = false;
		total_shortest = -1;
		//检查输入now_view_id是否包含在view_set_label里
		bool now_view_id_in_label = false;
		for (int i = 0; i < view_set_label.size(); i++) {
			if (view_set_label[i] == now_view_id) {
				now_view_id_in_label = true;
				break;
			}
		}
		assert(now_view_id_in_label && "now_view_id must be in view_set_label");
		//检查输入end_view_id是否包含在view_set_label里（如果有的话）
		bool end_view_id_in_label = false;
		if (end_view_id != -1) {
			for (int i = 0; i < view_set_label.size(); i++) {
				if (view_set_label[i] == end_view_id) {
					end_view_id_in_label = true;
					break;
				}
			}
			if (!end_view_id_in_label) {
				cout << "Warning: end_view_id not in view_set_label. No specific end point will be set." << endl;
				end_view_id = -1;
			}
			if (end_view_id == now_view_id) {
				cout << "Warning: end_view_id is the same as now_view_id. No specific end point will be set." << endl;
				end_view_id = -1;
			}
		}
		//构造下标映射
		view_id_in = new map<int, int>();
		view_id_out = new map<int, int>();
		for (int i = 0; i < view_set_label.size(); i++) {
			(*view_id_in)[view_set_label[i]] = i;
			(*view_id_out)[i] = view_set_label[i];
		}
		(*view_id_in)[views.size()] = view_set_label.size(); //注意复制节点应该是和视点空间个数相关，映射到所需视点个数
		(*view_id_out)[view_set_label.size()] = views.size();
		//节点数为原始个数+起点的复制节点
		n = view_set_label.size() + 1;
		//local path 完全无向图
		graph.resize(n);
		for (int i = 0; i < n; i++)
			graph[i].resize(n);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				if (i == n - 1 || j == n - 1) {
					//如果是起点的复制节点，距离为0
					graph[i][j] = 0.0;
				}
				else {
					//交换id
					int u = (*view_id_out)[i];
					int v = (*view_id_out)[j];
					//求两点路径
					pair<int, double> local_path = get_local_path(views[u].init_pos.eval(), views[v].init_pos.eval(), (share_data->object_center_world + Eigen::Vector3d(1e-10, 1e-10, 1e-10)).eval(), share_data->predicted_size * sqrt(2));
					if (local_path.first < 0) {
						cout << "local path not found." << endl;
						graph[i][j] = 1e10;
					}
					else graph[i][j] = local_path.second;
				}
				//cout << "graph[" << i << "][" << j << "] = " << graph[i][j] << endl;
			}
		//创建Gurobi的TSP优化器
		vars = new GRBVar * [n];
		for (int i = 0; i < n; i++)
			vars[i] = new GRBVar[n];
		env = new GRBEnv();
		model = new GRBModel(*env);
		//cout << "Gurobi model created." << endl;
		// Must set LazyConstraints parameter when using lazy constraints
		model->set(GRB_IntParam_LazyConstraints, 1);
		//cout << "Gurobi set LazyConstraints." << endl;
		// Create binary decision variables
		for (int i = 0; i < n; i++) {
			for (int j = 0; j <= i; j++) {
				vars[i][j] = model->addVar(0.0, 1.0, graph[i][j], GRB_BINARY, "x_" + to_string(i) + "_" + to_string(j));
				vars[j][i] = vars[i][j];
			}
		}
		//cout << "Gurobi addVar complete." << endl;
		// Degree-2 constraints
		for (int i = 0; i < n; i++) {
			GRBLinExpr expr = 0;
			for (int j = 0; j < n; j++)
				expr += vars[i][j];
			model->addConstr(expr == 2, "deg2_" + to_string(i));
		}
		//cout << "Gurobi add Degree-2 Constr complete." << endl;
		// Forbid edge from node back to itself
		for (int i = 0; i < n; i++)
			vars[i][i].set(GRB_DoubleAttr_UB, 0);
		//cout << "Gurobi set Forbid edge complete." << endl;
		// Make copy node linked to starting node
		vars[n - 1][(*view_id_in)[now_view_id]].set(GRB_DoubleAttr_LB, 1);
		// 默认不设置终点，多解只返回一个
		if (end_view_id != -1) vars[(*view_id_in)[end_view_id]][n - 1].set(GRB_DoubleAttr_LB, 1);
		//cout << "Gurobi set Make copy node complete." << endl;
		// Set callback function
		cb = new subtourelim(vars, n);
		model->setCallback(cb);
		//cout << "Gurobi set callback function complete." << endl;
		cout << "Global_Path_Planner inited." << endl;
	}

	~Global_Path_Planner() {
		delete view_id_in;
		delete view_id_out;
		graph.clear();
		graph.shrink_to_fit();
		global_path.clear();
		global_path.shrink_to_fit();
		for (int i = 0; i < n; i++)
			delete[] vars[i];
		delete[] vars;
		delete env;
		delete model;
		delete cb;
	}

	double solve() {
		auto start_time = chrono::high_resolution_clock::now();
		// Optimize model
		model->optimize();
		// Extract solution
		if (model->get(GRB_IntAttr_SolCount) > 0) {
			solved = true;
			total_shortest = 0.0;
			double** sol = new double* [n];
			for (int i = 0; i < n; i++)
				sol[i] = model->get(GRB_DoubleAttr_X, vars[i], n);

			int* tour = new int[n];
			int len;

			findsubtour(n, sol, &len, tour);
			assert(len == n);

			//cout << "Tour: ";
			for (int i = 0; i < len; i++) {
				global_path.push_back(tour[i]);
				if (i != len - 1) {
					total_shortest += graph[tour[i]][tour[i + 1]];
				}
				else {
					total_shortest += graph[tour[i]][tour[0]];
				}
				//cout << tour[i] << " ";
			}
			//cout << endl;

			for (int i = 0; i < n; i++)
				delete[] sol[i];
			delete[] sol;
			delete[] tour;
		}
		else {
			cout << "No solution found" << endl;
		}
		auto end_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_seconds = end_time - start_time;
		double cost_time = elapsed_seconds.count();
		cout << "Global Path length " << total_shortest << " getted with executed time " << cost_time << " s." << endl;
		return total_shortest;
	}

	vector<int> get_path_id_set() {
		if (!solved) cout << "call solve() first" << endl;
		cout << "Node ids on global_path form start to end are: ";
		vector<int> ans = global_path;
		//调准顺序把复制的起点置于末尾
		int copy_node_id = -1;
		for (int i = 0; i < ans.size(); i++) {
			if (ans[i] == n - 1) {
				copy_node_id = i;
				break;
			}
		}
		if (copy_node_id == -1) {
			cout << "copy_node_id == -1" << endl;
		}
		for (int i = 0; i < copy_node_id; i++) {
			ans.push_back(ans[0]);
			ans.erase(ans.begin());
		}
		//删除复制的起点
		for (int i = 0; i < ans.size(); i++) {
			if (ans[i] == n - 1) {
				ans.erase(ans.begin() + i);
				break;
			}
		}
		//如果起点是第一个就不动，是最后一个就反转
		if (ans[0] != (*view_id_in)[now_view_id]) {
			reverse(ans.begin(), ans.end());
		}
		for (int i = 0; i < ans.size(); i++) {
			ans[i] = (*view_id_out)[ans[i]];
			cout << ans[i] << " ";
		}
		cout << endl;
		//删除出发点
		ans.erase(ans.begin());
		return ans;
	}
};


//views_voxels_LM.hpp
//用于GT的集合覆盖
class views_voxels_LM {
public:
	Share_Data* share_data;
	View_Space* view_space;
	vector<vector<bool>> graph;
	unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_id_map;	//体素下标
	int num_of_voxel;
	set<int> chosen_views;
	GRBEnv* env;
	GRBModel* model;
	vector<GRBVar> x;
	GRBLinExpr obj;

	void solve() {
		// Optimize model
		model->optimize();
		// show nonzero variables
		/*for (int i = 0; i < share_data->num_of_views; i++)
			if (x[i].get(GRB_DoubleAttr_X) == 1.0)
				cout << x[i].get(GRB_StringAttr_VarName) << " " << x[i].get(GRB_DoubleAttr_X) << endl;
		// show num of views
		cout << "Obj: " << model->get(GRB_DoubleAttr_ObjVal) << endl;*/
	}

	vector<int> get_view_id_set() {
		vector<int> ans;
		for (int i = 0; i < share_data->num_of_views; i++)
			if (x[i].get(GRB_DoubleAttr_X) == 1.0) ans.push_back(i);
		return ans;
	}

	views_voxels_LM(Share_Data* _share_data, View_Space* _view_space, set<int> _chosen_views, vector<unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>*> voxels, set<int> forbidden_views = set<int>() ) {
		double now_time = clock();
		share_data = _share_data;
		view_space = _view_space;
		chosen_views = _chosen_views;
		//建立体素的id表
		assert(voxels.size() == share_data->num_of_views);
		num_of_voxel = 0;
		voxel_id_map = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		for (int i = 0; i < voxels.size(); i++) {
			for (auto& it : *voxels[i]) {
				if (voxel_id_map->find(it.first) == voxel_id_map->end()) {
					(*voxel_id_map)[it.first] = num_of_voxel++;
				}
			}
		}
		//cout << num_of_voxel << " real | gt " << share_data->full_voxels << endl;
		graph.resize(share_data->num_of_views);
		for (int i = 0; i < share_data->num_of_views; i++) {
			graph[i].resize(num_of_voxel);
			for (int j = 0; j < num_of_voxel; j++) {
				graph[i][j] = 0;
			}
		}
		set<int> voxels_not_need;
		for (int i = 0; i < voxels.size(); i++) {
			for (auto& it : *voxels[i]) {
				graph[i][(*voxel_id_map)[it.first]] = 1;
				if (chosen_views.find(i) != chosen_views.end()) {
					voxels_not_need.insert((*voxel_id_map)[it.first]);
					//cout << "voxel " << (*voxel_id_map)[it.first] << " is already covered by chosen view " << i << " and will not be added to the linear program." << endl;
				}
			}
		}
		for (int j = 0; j < num_of_voxel; ++j) {
			if (voxels_not_need.count(j)) continue;  // 已经覆盖的不需要管
			bool has_valid_cover = false;
			for (int i = 0; i < share_data->num_of_views; ++i) {
				if (forbidden_views.count(i)) continue;
				if (graph[i][j] == 1) {
					has_valid_cover = true; 
					break; 
				}
			}
			if (!has_valid_cover) {
				voxels_not_need.insert(j); //把它也从 LP 里移除
				//cout << "voxel " << j << " removed (no valid cover)" << endl;
			}
		}
		//建立对应的线性规划求解器
		now_time = clock();
		env = new GRBEnv();
		model = new GRBModel(*env);
		x.resize(share_data->num_of_views);
		// Create variables
		for (int i = 0; i < share_data->num_of_views; i++)
			x[i] = model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "x" + to_string(i));
		// Set objective : \sum_{s\in S} x_s
		for (int i = 0; i < share_data->num_of_views; i++) {
			if (forbidden_views.count(i)) continue; //跳过被禁止的视点
			obj += x[i];
		}
		model->setObjective(obj, GRB_MINIMIZE);
		// Add linear constraint: \sum_{S:e\in S} x_s\geq1
		for (int j = 0; j < num_of_voxel; j++)
		{
			if (voxels_not_need.find(j) != voxels_not_need.end()) continue; //跳过不需要覆盖的体素，或者说不需要覆盖的体素不加入约束
			GRBLinExpr subject_of_voxel;
			for (int i = 0; i < share_data->num_of_views; i++) {
				if (forbidden_views.count(i)) continue; //跳过被禁止的视点，或者说被禁止的视点不加入约束
				if (graph[i][j] == 1) subject_of_voxel += x[i];
			}
			model->addConstr(subject_of_voxel >= 1, "c" + to_string(j));
		}
		// set forbidden views' variables to 0
		for (int i = 0; i < share_data->num_of_views; i++) {
			if (forbidden_views.count(i)) x[i].set(GRB_DoubleAttr_UB, 0);
		}
		model->set("TimeLimit", "100");
		//cout << "Integer linear program formulated with executed time " << clock() - now_time << " ms." << endl;
	}

	~views_voxels_LM() {
		for (int i = 0; i < graph.size(); i++) {
			graph[i].clear();
			graph[i].shrink_to_fit();
		}
		graph.clear();
		graph.shrink_to_fit();
		chosen_views.clear();
		delete voxel_id_map;
		delete env;
		delete model;
	}
};

//NVB_Planner.hpp
#define Over 0
#define WaitData 1
#define WaitViewSpace 2
#define WaitInformation 3
#define WaitMoving 4

void save_cloud_mid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string name, Share_Data* share_data);
void create_view_space(View_Space** now_view_space, View* now_best_view, Share_Data* share_data, int iterations);
void create_views_information(Views_Information** now_views_infromation, View* now_best_view, View_Space* now_view_space, Share_Data* share_data, int iterations);
void move_robot(View* now_best_view, View_Space* now_view_space, Share_Data* share_data);
void show_cloud(pcl::visualization::PCLVisualizer::Ptr viewer);

class NBV_Planner
{
public:
	atomic<int> status;
	int iterations;
	Perception_3D* percept;
	Voxel_Information* voxel_information;
	View_Space* now_view_space;
	Views_Information* now_views_infromation;
	View* now_best_view;
	Share_Data* share_data;
	pcl::visualization::PCLVisualizer::Ptr viewer;
	bool now_views_infromation_created = false;

	~NBV_Planner() {
		delete percept;
		delete now_best_view;
		delete voxel_information;
		delete now_view_space;
		//只有使用过搜索方法的生成了information
		if (now_views_infromation_created) delete now_views_infromation;
	}

	double check_size(double predicted_size, Eigen::Vector3d object_center_world, vector<Eigen::Vector3d>& points) {
		int vaild_points = 0;
		for (auto& ptr : points) {
			if (ptr(0) < object_center_world(0) - predicted_size || ptr(0) > object_center_world(0) + predicted_size) continue;
			if (ptr(1) < object_center_world(1) - predicted_size || ptr(1) > object_center_world(1) + predicted_size) continue;
			if (ptr(2) < object_center_world(2) - predicted_size || ptr(2) > object_center_world(2) + predicted_size) continue;
			vaild_points++;
		}
		return (double)vaild_points / (double)points.size();
	}

	NBV_Planner(Share_Data* _share_data, int _status = WaitData) {
		share_data = _share_data;
		iterations = 0;
		share_data->iterations = 0;
		status = _status;
		share_data->now_view_space_processed = false;
		share_data->now_views_infromation_processed = false;
		share_data->move_on = false;
		voxel_information = new Voxel_Information(share_data->p_unknown_lower_bound, share_data->p_unknown_upper_bound);
		voxel_information->init_mutex_views(share_data->num_of_views);
		//只有初始选择了搜索方法才会生成信息增益类，结合的pipeline也是在初始化时有搜索才需要删除
		if (share_data->method_of_IG == 0 || share_data->method_of_IG == 1 || share_data->method_of_IG == 2 || share_data->method_of_IG == 3 || share_data->method_of_IG == 4 || share_data->method_of_IG == 5 || share_data->method_of_IG == 10) {
			now_views_infromation_created = true;
		}
		//初始化GT
		//share_data->access_directory(share_data->save_path);
		//cloud_ground_truth copy from cloud_pcd
		share_data->cloud_ground_truth = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::copyPointCloud(*share_data->cloud_pcd, *share_data->cloud_ground_truth);
		//obb中心作为物体中心，注意目前没有使用完整的BBX（暂不影响实验），所以要取一个巨大的BBX（包括所有的视点）
		share_data->object_center_world = share_data->plant_obb.c;
		share_data->predicted_size = 1.5 * max(share_data->plant_obb.half()(0), max(share_data->plant_obb.half()(1), share_data->plant_obb.half()(2))); 
		cout << "object " << share_data->name_of_pcd << " size is " << share_data->predicted_size << " m." << endl;

		double predicted_octomap_resolution = -1.0;
		if ((share_data->Combined_on == true && !share_data->use_history_model_for_covering) || share_data->method_of_IG == 7 || share_data->method_of_IG == 6 || share_data->method_of_IG == 9) {
			//动态分辨率，对于网络而言
			predicted_octomap_resolution = share_data->predicted_size * 2.0 / 32.0;
		}
		else {
			//对于体素方法
			predicted_octomap_resolution = share_data->ground_truth_resolution * share_data->voxel_resolution_factor;
		}
		cout << "choose octomap_resolution: " << predicted_octomap_resolution << " m." << endl;
		share_data->octomap_resolution = predicted_octomap_resolution;
		share_data->octo_model = new octomap::ColorOcTree(share_data->octomap_resolution);
		share_data->octo_model->setOccupancyThres(0.65);
		share_data->GT_sample = new octomap::ColorOcTree(share_data->octomap_resolution);
		//转换点云
		double min_z = share_data->object_center_world(2);
		for (auto ptr = share_data->cloud_ground_truth->points.begin(); ptr != share_data->cloud_ground_truth->points.end(); ptr++)
		{
			//GT插入点云
			octomap::OcTreeKey key;  bool key_have = share_data->ground_truth_model->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key);
			if (key_have) {
				octomap::ColorOcTreeNode* voxel = share_data->ground_truth_model->search(key);
				if (voxel == NULL) {
					share_data->ground_truth_model->setNodeValue(key, share_data->ground_truth_model->getProbHitLog(), true);
					share_data->ground_truth_model->integrateNodeColor(key, (*ptr).r, (*ptr).g, (*ptr).b);
				}
			}
			min_z = min(min_z, (double)(*ptr).z);
			//GT_sample插入点云
			octomap::OcTreeKey key_sp;  bool key_have_sp = share_data->GT_sample->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key_sp);
			if (key_have_sp) {
				octomap::ColorOcTreeNode* voxel_sp = share_data->GT_sample->search(key_sp);
				if (voxel_sp == NULL) {
					share_data->GT_sample->setNodeValue(key_sp, share_data->GT_sample->getProbHitLog(), true);
					share_data->GT_sample->integrateNodeColor(key_sp, (*ptr).r, (*ptr).g, (*ptr).b);
				}
			}
		}
		//cout << min_z << endl;
		//pcl::io::savePCDFile<pcl::PointXYZRGB>("C:\\Users\\yixinizhu\\Desktop\\" + share_data->name_of_pcd + ".pcd", *share_data->cloud_ground_truth);
		//记录桌面
		share_data->min_z_table = min_z - share_data->ground_truth_resolution;
		cout << "min_z_table is " << share_data->min_z_table << endl;

		share_data->ground_truth_model->updateInnerOccupancy();
		//share_data->ground_truth_model->write(share_data->save_path + "/GT.ot");
		//GT_sample_voxels
		share_data->GT_sample->updateInnerOccupancy();
		//share_data->GT_sample->write(share_data->save_path + "/GT_sample.ot");
		share_data->init_voxels = 0;
		int full_voxels = 0;
		//在sample中统计总个数
		for (octomap::ColorOcTree::leaf_iterator it = share_data->GT_sample->begin_leafs(), end = share_data->GT_sample->end_leafs(); it != end; ++it){
			if(it.getY() > share_data->min_z_table + share_data->octomap_resolution)
				share_data->init_voxels++;
			full_voxels++;
		}
		cout << "Map_GT_sample has voxels " << share_data->init_voxels << endl;
		cout << "Map_GT_sample has voxels with bottom " << full_voxels << endl;
		share_data->init_voxels = full_voxels;
		//ofstream fout_sample(share_data->save_path + "/GT_sample_voxels.txt");
		//fout_sample << share_data->init_voxels << endl;
		//在GT中统计总个数
		share_data->cloud_points_number = 0;
		for (octomap::ColorOcTree::leaf_iterator it = share_data->ground_truth_model->begin_leafs(), end = share_data->ground_truth_model->end_leafs(); it != end; ++it) {
			share_data->cloud_points_number++;
		}
		cout << "Map_GT has voxels " << share_data->cloud_points_number << endl;
		assert(share_data->cloud_points_number == share_data->cloud_points_number_file && "cloud_points_number should be equal to cloud_points_number_file");
		//ofstream fout_gt(share_data->save_path + "/GT_voxels.txt");
		//fout_gt << share_data->cloud_points_number << endl;

		//初始化viewspace
		int first_view_id = share_data->first_view_id;
		now_view_space = new View_Space(iterations, share_data, voxel_information, share_data->cloud_ground_truth, first_view_id);
		//设置初始视点为统一的位置
		now_view_space->views[first_view_id].vis++;
		now_best_view = new View(now_view_space->views[first_view_id]);
		cout << "initial view id is " << first_view_id << endl;
		//运动代价：视点id，当前代价，总体代价
		share_data->movement_cost = 0;
		//share_data->access_directory(share_data->save_path + "/movement");
		//ofstream fout_move(share_data->save_path + "/movement/path" + to_string(-1) + ".txt");
		//fout_move << 0 << '\t' << 0.0 << '\t' << 0.0 << endl;
		now_best_view->get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
		Eigen::Matrix4d view_pose_world = (share_data->now_camera_pose_world * now_best_view->pose.inverse()).eval();
		cout << "object_center_world is: " << share_data->object_center_world << endl;
		//cout << "now_camera_pose_world is: " << share_data->now_camera_pose_world << endl;
		//cout << "Initial camera pose in world is: " << view_pose_world << endl;
		//相机类初始化
		percept = new Perception_3D(share_data);
		if (share_data->show) { //显示BBX、相机位置、GT
			pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Iteration"));
			viewer->setBackgroundColor(0, 0, 0);
			viewer->addCoordinateSystem(0.1);
			viewer->initCameraParameters();
			//第一帧相机位置
			Eigen::Vector4d X(0.05, 0, 0, 1);
			Eigen::Vector4d Y(0, 0.05, 0, 1);
			Eigen::Vector4d Z(0, 0, 0.05, 1);
			Eigen::Vector4d O(0, 0, 0, 1);
			X = view_pose_world * X;
			Y = view_pose_world * Y;
			Z = view_pose_world * Z;
			O = view_pose_world * O;
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
			//test_viewspace
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr test_viewspace(new pcl::PointCloud<pcl::PointXYZRGB>);
			test_viewspace->is_dense = false;
			test_viewspace->points.resize(now_view_space->views.size());
			auto pt = test_viewspace->points.begin();
			for (int i = 0; i < now_view_space->views.size(); i++, pt++) {
				(*pt).x = now_view_space->views[i].init_pos(0);
				(*pt).y = now_view_space->views[i].init_pos(1);
				(*pt).z = now_view_space->views[i].init_pos(2);
				//第一次显示所有点为白色
				(*pt).r = 255, (*pt).g = 255, (*pt).b = 255;
			}
			viewer->addPointCloud<pcl::PointXYZRGB>(test_viewspace, "test_viewspace");
			now_view_space->add_bbx_to_cloud(viewer);
			viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_ground_truth");
			while (!viewer->wasStopped())
			{
				viewer->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		}

	}

	int plan() {
		switch (status)
		{
		case Over:
			break;
		case WaitData:
			if (percept->precept(now_best_view)) {
				thread next_view_space(create_view_space, &now_view_space, now_best_view, share_data, iterations);
				next_view_space.detach();
				status = WaitViewSpace;
			}
			break;
		case WaitViewSpace:
			if (share_data->now_view_space_processed) {
				thread next_views_information(create_views_information, &now_views_infromation, now_best_view, now_view_space, share_data, iterations);
				next_views_information.detach();
				status = WaitInformation;
			}
			break;
		case WaitInformation:
			if (share_data->now_views_infromation_processed) {
				if (share_data->method_of_IG == 14) { //SamplingNBV

					//初始化planner
					if (iterations == 0) {
						//分割成10个部分，按行和层顺序
						//share_data->ordered_blocks = buildOrderedRowLayerBlocks(
						//	share_data->view_obb,
						//	share_data->dynamic_candidate_views,
						//	now_best_view->init_pos,
						//	/*row_axis_idx=*/0,
						//	/*layer_axis_idx=*/2,
						//	/*num_rows=*/5,
						//	/*forward_to_row_max=*/true,
						//	/*serpentine_other_layer=*/false
						//);
						share_data->ordered_blocks = buildOrderedRowLayerBlocks(
							share_data->view_obb,
							share_data->dynamic_candidate_views,
							now_best_view->init_pos,
							/*row_axis_idx=*/0,
							/*layer_axis_idx=*/2,
							/*num_rows=*/5,
							/*forward_to_row_max=*/true,
							/*serpentine_other_layer=*/true
						);

						if (share_data->ordered_blocks.empty()) {
							std::cout << "ordered_blocks is empty. Abort SamplingNBV." << std::endl;
							status = Over;
							break;
						}
						//把默认的15个视点index偏移加上
						for (size_t b = 0; b < share_data->ordered_blocks.size(); ++b) {
							for (int index = 0; index < share_data->ordered_blocks[b].size(); index++) {
								share_data->ordered_blocks[b][index] += share_data->passive_init_views.size();
							}
						}
						share_data->current_block = 0;

						if (share_data->show || share_data->debug) {
							auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("block viewer"));
							viewer->setBackgroundColor(0.05, 0.05, 0.05);
							viewer->addCoordinateSystem(0.2);
							viewer->initCameraParameters();
							//显示分块结果
							pcl::PointCloud<pcl::PointXYZRGB>::Ptr block_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
							for (size_t b = 0; b < share_data->ordered_blocks.size(); ++b) {
								for (int vid : share_data->ordered_blocks[b]) {
									Eigen::Vector3d pos = now_view_space->views[vid].init_pos;
									pcl::PointXYZRGB p;
									p.x = pos(0);
									p.y = pos(1);
									p.z = pos(2);
									//不同block显示不同颜色
									uint8_t r = (b * 50) % 256;
									uint8_t g = (b * 80) % 256;
									uint8_t b_color = (b * 110) % 256;
									p.r = r;
									p.g = g;
									p.b = b_color;
									block_cloud->points.push_back(p);
								}
							}
							viewer->addPointCloud<pcl::PointXYZRGB>(block_cloud, "block_cloud");
							viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "block_cloud");
							viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_ground_truth");
							while (!viewer->wasStopped())
							{
								viewer->spinOnce(100);
								boost::this_thread::sleep(boost::posix_time::microseconds(100000));
							}
						}

						//保存分块结果
						share_data->access_directory(share_data->save_path + "/block_info");
						ofstream fout_block(share_data->save_path + "/block_info/ordered_blocks.txt");
						for (size_t b = 0; b < share_data->ordered_blocks.size(); ++b) {
							fout_block << "Block " << b << ": ";
							for (int vid : share_data->ordered_blocks[b]) {
								fout_block << vid << " ";
							}
							fout_block << endl;
						}
						cout << "Ordered blocks built and saved." << endl;
						share_data->current_budget = share_data->num_of_max_iteration / share_data->ordered_blocks.size() - 1; //固定10个block，那么当前block的资源是10-1
						
						//把被动视点设置为不能访问
						for (int i = 0; i < share_data->passive_init_views.size(); i++) {
							now_view_space->views[i].can_move = false;
						}
					}

					//如果当前block已经没有资源了，就移动到下一个block
					if (share_data->current_budget <= 0) {
						//移动到下一个block
						share_data->current_block++;
						//检查是否越界
						if (share_data->current_block >= share_data->ordered_blocks.size()) {
							std::cout << "Out of block range; stop planner loop or clamp." << std::endl;
							share_data->current_block = (int)share_data->ordered_blocks.size() - 1; // or return
						}
						share_data->current_budget = share_data->num_of_max_iteration / share_data->ordered_blocks.size(); //重置资源
					}

					//根据当前地图采样look ats（表面和探索frontier）
					auto cs = share_data->plant_obb.corners();
					Eigen::Vector3d bb_min = cs[0];
					Eigen::Vector3d bb_max = cs[0];
					for (int i = 1; i < 8; ++i) {
						bb_min = bb_min.cwiseMin(cs[i]);
						bb_max = bb_max.cwiseMax(cs[i]);
					}
					const double pad = 2.0 * share_data->octomap_resolution;  // 壳厚度同量级的 padding，避免角落漏 key
					octomap::OcTreeKey kmin, kmax;
					bool ok_min = share_data->octo_model->coordToKeyChecked(octomap::point3d(bb_min.x() - pad, bb_min.y() - pad, bb_min.z() - pad), kmin);
					bool ok_max = share_data->octo_model->coordToKeyChecked(octomap::point3d(bb_max.x() + pad, bb_max.y() + pad, bb_max.z() + pad), kmax);
					if (!ok_min || !ok_max) {
						std::cout << "Warning: plant OBB AABB is out of octomap bounds when converting to keys." << std::endl;
						// continue;
					}
					vector<octomap::point3d> surface_frontier_pts;
					vector<octomap::point3d> exploration_frontier_pts;
					for (unsigned int kx = kmin[0]; kx <= kmax[0]; ++kx) {
						for (unsigned int ky = kmin[1]; ky <= kmax[1]; ++ky) {
							for (unsigned int kz = kmin[2]; kz <= kmax[2]; ++kz) {
								octomap::OcTreeKey key(kx, ky, kz);
								octomap::point3d pt = share_data->octo_model->keyToCoord(key);
								octomap::OcTreeNode* node = share_data->octo_model->search(key); //确保这个key在树里，没的话会被创建成unknown
								if (node == NULL) continue;
								double occupancy = node->getOccupancy();
								if (voxel_information->is_unknown(occupancy)) {
									int node_type = frontier_check(pt, share_data->octo_model, voxel_information, share_data->octomap_resolution);
									if (node_type == 2) surface_frontier_pts.push_back(pt); //if it has a surface neibour and a free neibour
									if (node_type == 1) exploration_frontier_pts.push_back(pt); //if it has a free neibour
								}
							}
						}
					}
					cout << "surface frontier has " << surface_frontier_pts.size() << " points." << endl;
					cout << "exploration frontier has " << exploration_frontier_pts.size() << " points." << endl;

					//对当前block里的视点分配frontier
					vector<int> current_block_view_id = share_data->ordered_blocks[share_data->current_block];
					vector<Eigen::Vector3d> current_block_view_postion;
					for (auto view_id : current_block_view_id) {
						current_block_view_postion.push_back(now_view_space->views[view_id].init_pos);
					}
					// 计算当前 block 在 XY 平面的范围，并扩张 scale_xy 倍
					double xmin = std::numeric_limits<double>::infinity();
					double xmax = -std::numeric_limits<double>::infinity();
					double ymin = std::numeric_limits<double>::infinity();
					double ymax = -std::numeric_limits<double>::infinity();
					for (const auto& p : current_block_view_postion) {
						xmin = std::min(xmin, p.x());
						xmax = std::max(xmax, p.x());
						ymin = std::min(ymin, p.y());
						ymax = std::max(ymax, p.y());
					}
					double cx = 0.5 * (xmin + xmax);
					double cy = 0.5 * (ymin + ymax);
					double hx = 0.5 * (xmax - xmin);
					double hy = 0.5 * (ymax - ymin);
					const double scale_xy = 1.1;  // 当前 block XY 范围放大倍数
					hx *= scale_xy;
					hy *= scale_xy;
					// 防止 block 在某方向退化（半宽=0）
					const double min_half = 2.0 * share_data->octomap_resolution;
					hx = std::max(hx, min_half);
					hy = std::max(hy, min_half);
					double x_low = cx - hx, x_high = cx + hx;
					double y_low = cy - hy, y_high = cy + hy;
					// Z 方向上使用层级
					double zmin = std::numeric_limits<double>::infinity();
					double zmax = -std::numeric_limits<double>::infinity();
					for (const auto& p : current_block_view_postion) {
						zmin = std::min(zmin, p.z());
						zmax = std::max(zmax, p.z());
					}
					double cz = 0.5 * (zmin + zmax);
					double hz = 0.5 * (zmax - zmin);
					const double scale_z = 1.2;  // Z 比 XY 放宽
					hz *= scale_z;
					// 防退化：block若在同一层，给一个最小半高
					const double min_half_z = 3.0 * share_data->octomap_resolution;
					hz = std::max(hz, min_half_z);
					double z_low = cz - hz;
					double z_high = cz + hz;
					auto in_block_xyz = [&](const octomap::point3d& pt) {
						return (pt.x() >= x_low && pt.x() <= x_high &&
							pt.y() >= y_low && pt.y() <= y_high &&
							pt.z() >= z_low && pt.z() <= z_high);
					};
					pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_surface_frontier(new pcl::PointCloud<pcl::PointXYZ>);
					pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_exploration_frontier(new pcl::PointCloud<pcl::PointXYZ>);
					int kept_surface = 0, kept_explore = 0;
					for (const auto& pt : surface_frontier_pts) {
						if (!in_block_xyz(pt)) continue;
						pcl::PointXYZ p;
						p.x = pt.x();
						p.y = pt.y();
						p.z = pt.z();
						cloud_surface_frontier->points.push_back(p);
						kept_surface++;
					}
					cloud_surface_frontier->width = cloud_surface_frontier->points.size();
					cloud_surface_frontier->height = 1;
					cloud_surface_frontier->is_dense = false;
					for (const auto& pt : exploration_frontier_pts) {
						if (!in_block_xyz(pt)) continue;
						pcl::PointXYZ p;
						p.x = pt.x();
						p.y = pt.y();
						p.z = pt.z();
						cloud_exploration_frontier->points.push_back(p);
						kept_explore++;
					}
					cloud_exploration_frontier->width = cloud_exploration_frontier->points.size();
					cloud_exploration_frontier->height = 1;
					cloud_exploration_frontier->is_dense = false;
					std::cout << "Block XY filter range: x[" << x_low << ", " << x_high << "], y[" << y_low << ", " << y_high << "]" << std::endl;
					std::cout << "Filtered frontier kept: surface=" << kept_surface<< ", explore=" << kept_explore << std::endl;
					std::mt19937 rng(share_data->random_seed + iterations);
					int K = std::max(1, (int)(share_data->num_targets / share_data->ordered_blocks.size()));
					int K_surface = K;              // 优先全部分配给 surface
					std::vector<Eigen::Vector3d> surface_targets = farthestPointSampling(cloud_surface_frontier, K_surface, rng);
					std::cout << "FPS surface_targets: " << surface_targets.size() << "\n";
					// 剩余给 exploration
					int K_explore = std::max(0, K - (int)surface_targets.size());
					std::vector<Eigen::Vector3d> exploration_targets;
					if (K_explore > 0) exploration_targets = farthestPointSampling(cloud_exploration_frontier, K_explore, rng);
					std::cout << "FPS exploration_targets: " << exploration_targets.size() << "\n";
					std::vector<Eigen::Vector3d> targets = surface_targets;
					targets.insert(targets.end(), exploration_targets.begin(), exploration_targets.end());
					int remain = K - (int)targets.size();
					if (remain > 0) {
						std::cout << "Not enough frontier targets for current block, need " << remain << " more. Re-sampling from current surface." << endl;
						pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_current_surface(new pcl::PointCloud<pcl::PointXYZ>);
						for (const auto& pt : share_data->cloud_final->points) {
							octomap::point3d pt_vec(pt.x, pt.y, pt.z);
							//if (!in_block_xyz(pt_vec)) continue;
							cloud_current_surface->points.push_back(pcl::PointXYZ(pt.x, pt.y, pt.z));
						}
						cloud_current_surface->width = cloud_current_surface->points.size();
						cloud_current_surface->height = 1;
						cloud_current_surface->is_dense = false;
						std::vector<Eigen::Vector3d> extra_targets;
						if (!cloud_current_surface->empty()) extra_targets = farthestPointSampling(cloud_current_surface, remain, rng);
						targets.insert(targets.end(), extra_targets.begin(), extra_targets.end());
					}
					std::cout << "Total targets for current block: " << targets.size() << " / " << K << "\n";

					// 对当前 block 的视点和目标进行分配
					vector<int> current_valid_block_view_id;
					vector<Eigen::Vector3d> current_valid_block_view_position;
					for (auto view_id : current_block_view_id) {
						if (now_view_space->views[view_id].vis) continue; //访问过的不去
						if (!now_view_space->views[view_id].can_move) continue; //禁止的不去
						current_valid_block_view_id.push_back(view_id);
						current_valid_block_view_position.push_back(now_view_space->views[view_id].init_pos);
					}

					// round-robin assign candidate -> lookat targets
					ViewAssignResult assign = assignViewsRoundRobin(current_valid_block_view_position, targets, share_data->num_rounds, rng);
					//save and update the assigned views and targets
					const size_t candidateN = share_data->dynamic_candidate_views.size();
					const size_t passiveN = share_data->passive_init_views.size();
					assert(now_view_space->views.size() == passiveN + candidateN);
					std::vector<Eigen::Vector3d> lookat_all(candidateN, Eigen::Vector3d(0, 0, 0));
					for (size_t i = 0; i < assign.used_idx.size(); ++i) {
						size_t global_vid = (size_t)current_valid_block_view_id[assign.used_idx[i]]; // passiveN + candidate_idx
						if (global_vid >= passiveN && global_vid < passiveN + candidateN) {
							size_t cand_idx = global_vid - passiveN;
							lookat_all[cand_idx] = assign.lookat[i];
						}
					}
					share_data->access_directory(share_data->save_path + "/assign");
					std::ofstream fout_assign(share_data->save_path + "/assign/" + to_string(iterations) + ".txt");
					for (size_t c = 0; c < candidateN; ++c) {
						const auto& la = lookat_all[c];
						fout_assign << la.x() << "\t" << la.y() << "\t" << la.z() << "\n";
						now_view_space->views[passiveN + c].look_at = la;
					}

					if (share_data->debug || share_data->show) { //显示分配结果
						auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("simple viewer"));
						viewer->setBackgroundColor(0.05, 0.05, 0.05);
						viewer->addCoordinateSystem(0.2);
						viewer->initCameraParameters();
						// 0) 当前的地图点云
						viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_final, "cloud_final");
						viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud_final");
						// 1) 原始 downsample 点云：灰色，小点
						pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col1(cloud_surface_frontier, 220, 220, 220);
						viewer->addPointCloud<pcl::PointXYZ>(cloud_surface_frontier, col1, "cloud_surface_frontier");
						viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_surface_frontier");
						pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col_explore(cloud_exploration_frontier, 128, 0, 128);
						viewer->addPointCloud<pcl::PointXYZ>(cloud_exploration_frontier, col_explore, "cloud_exploration_frontier");
						viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_exploration_frontier");
						// 2) targets / lookat：红色，大点
						auto pc_targets = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
						for (size_t k = 0; k < assign.lookat.size(); ++k) {
							const Eigen::Vector3d& p = assign.lookat[k];   // 你也可以换成 targets[i]
							pc_targets->push_back(pcl::PointXYZ((float)p.x(), (float)p.y(), (float)p.z()));
						}
						pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col2(pc_targets, 255, 60, 60);
						viewer->addPointCloud<pcl::PointXYZ>(pc_targets, col2, "targets");
						viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "targets");
						// 3) views：蓝色，中点
						auto pc_views = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
						for (size_t k = 0; k < assign.used_idx.size(); ++k) {
							int local_idx = assign.used_idx[k];
							int global_vid = current_valid_block_view_id[local_idx];
							int cand_idx = global_vid - (int)share_data->passive_init_views.size();
							Eigen::Vector3d cam = share_data->dynamic_candidate_views[cand_idx];
							pc_views->push_back(pcl::PointXYZ((float)cam.x(), (float)cam.y(), (float)cam.z()));
						}
						pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col3(pc_views, 80, 160, 255);
						viewer->addPointCloud<pcl::PointXYZ>(pc_views, col3, "views");
						viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "views");
						cout << "Number of assigned views: " << assign.used_idx.size() << endl;
						// 4) rays：每条匹配线段上采样若干点，绿色，小点，形成虚线箭头的效果
						auto pc_rays = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
						int N = 30;                 // 每条匹配线段采样点数（越大越密）
						for (size_t k = 0; k < assign.used_idx.size(); ++k) {
							int local_idx = assign.used_idx[k];
							int global_vid = current_valid_block_view_id[local_idx];
							int cand_idx = global_vid - (int)share_data->passive_init_views.size();
							Eigen::Vector3d cam = share_data->dynamic_candidate_views[cand_idx];
							Eigen::Vector3d tgt = assign.lookat[k];
							Eigen::Vector3d d = tgt - cam;
							double len = d.norm();
							if (len < 1e-9) continue;
							for (int i = 0; i <= 30; ++i) {
								double t = double(i) / double(N);         // [0,1]
								Eigen::Vector3d p = cam + t * d;
								pc_rays->push_back(pcl::PointXYZ((float)p.x(), (float)p.y(), (float)p.z()));
							}
						}
						cout << "Number of ray points: " << pc_rays->points.size() << endl;
						pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col4(pc_rays, 60, 255, 120);
						viewer->addPointCloud<pcl::PointXYZ>(pc_rays, col4, "rays");
						viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "rays");

						//显示分块结果
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr block_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
						for (size_t b = 0; b < share_data->ordered_blocks.size(); ++b) {
							for (int vid : share_data->ordered_blocks[b]) {
								Eigen::Vector3d pos = now_view_space->views[vid].init_pos;
								pcl::PointXYZRGB p;
								p.x = pos(0);
								p.y = pos(1);
								p.z = pos(2);
								//不同block显示不同颜色
								uint8_t r = (b * 50) % 256;
								uint8_t g = (b * 80) % 256;
								uint8_t b_color = (b * 110) % 256;
								p.r = r;
								p.g = g;
								p.b = b_color;
								block_cloud->points.push_back(p);
							}
						}
						viewer->addPointCloud<pcl::PointXYZRGB>(block_cloud, "block_cloud");
						viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "block_cloud");

						while (!viewer->wasStopped()) {
							viewer->spinOnce(10);
							std::this_thread::sleep_for(std::chrono::milliseconds(10));
						}
					}

					//对每个候选视点进行打分
					double ray_casting_start_time = clock();
					int best_view_id = -1;
					double highest_utility = - std::numeric_limits<double>::infinity();
					const double cost_alpha = 0.0;
					for (int i = 0; i < now_view_space->views.size(); i++) {

						if (now_view_space->views[i].vis) continue; //访问过的不去
						if (!now_view_space->views[i].can_move) continue; //禁止的不去
						if ((now_view_space->views[i].look_at).norm() < 1e-6) continue; //look_at是(0,0,0)不去

						double robot_cost = now_view_space->views[i].robot_cost;
						double information_gain = 0.0;
						int num_of_rays = 0;
						//采样并遍历ray
						now_view_space->views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
						Eigen::Matrix4d view_pose_world = (share_data->now_camera_pose_world * now_view_space->views[i].pose.inverse()).eval();
						double skip_coefficient = share_data->skip_coefficient;
						int pixel_interval = 1;
						double max_range = share_data->ray_max_dis;
						octomap::OcTreeKey key_origin;
						bool key_origin_have = share_data->octo_model->coordToKeyChecked(now_view_space->views[i].init_pos(0), now_view_space->views[i].init_pos(1), now_view_space->views[i].init_pos(2), key_origin);
						if (key_origin_have) {
							octomap::point3d origin = share_data->octo_model->keyToCoord(key_origin);
							int stride = std::max(1, (int)std::round(pixel_interval * skip_coefficient));
							for (int x = 0; x < share_data->color_intrinsics.width; x += stride)
								for (int y = 0; y < share_data->color_intrinsics.height; y += stride)
								{
									cv::Point2f pixel(x, y);
									octomap::point3d end = project_pixel_to_ray_end(x, y, share_data->color_intrinsics, view_pose_world, max_range);
									octomap::point3d direction = end - origin;
									octomap::point3d end_point;
									//如果方向为空，那不需要射线
									int step[3];
									octomap::point3d direction_norm = direction.normalized();
									for (unsigned int i = 0; i < 3; ++i) {
										// compute step direction_norm
										if (direction_norm(i) > 0.0) step[i] = 1;
										else if (direction_norm(i) < 0.0)   step[i] = -1;
										else step[i] = 0;
									}
									if (step[0] == 0 && step[1] == 0 && step[2] == 0) {
										//cout << "direction_norm is non. skip." << endl;
										continue;
									}
									//终点设置为最大距离
									end_point = origin + direction.normalized() * max_range; // use max range instead of stopping at the hit
									octomap::KeyRay ray_set;
									//获取射线数组，不包含末节点
									bool point_on_ray_getted = share_data->octo_model->computeRayKeys(origin, end_point, ray_set);
									if (!point_on_ray_getted) {
										cout << "Warning. ray cast with wrong max_range." << endl;
										continue;
									}
									//把终点放入射线组
									octomap::OcTreeKey key_end;
									if (share_data->octo_model->coordToKeyChecked(end_point, key_end)) {
										ray_set.addKey(key_end); // now valid
									}
									//统计未知体素个数
									int num_of_unknown = 0;
									int num_of_valid_points_on_ray = 0;
									for (octomap::KeyRay::iterator it = ray_set.begin(); it != ray_set.end(); ++it) {
										auto node = share_data->octo_model->search(*it);
										if (node == NULL) { //初始化了BBX，所以空节点说明在BBX外，无视掉即可
											continue;
										}
										double occupancy = node->getOccupancy();
										if (voxel_information->is_occupied(occupancy)) break;
										num_of_valid_points_on_ray++;
										if (voxel_information->is_unknown(occupancy)) num_of_unknown++;
									}
									if (num_of_valid_points_on_ray > 0) {
										information_gain += 1.0 * num_of_unknown / num_of_valid_points_on_ray;
										num_of_rays++;
									}
								}
						}
						if (num_of_rays > 0) information_gain /= num_of_rays;

						double utility = information_gain - cost_alpha * robot_cost;

						//cout << "View " << i << ": robot_cost = " << robot_cost << ", information_gain = " << information_gain << ", utility = " << utility << endl;

						if (highest_utility < utility) {
							highest_utility = utility;
							best_view_id = i;
						}
					}
					cout << "highest_utility: " << highest_utility << endl;
					double ray_casting_end_time = clock();
					double ray_casting_time = double(ray_casting_end_time - ray_casting_start_time) / CLOCKS_PER_SEC;
					cout << "Ray casting time for this iteration: " << ray_casting_time << " seconds." << endl;
					//save time to file
					share_data->access_directory(share_data->save_path + "/ray_casting_time");
					ofstream fout_time(share_data->save_path + "/ray_casting_time/time" + to_string(iterations) + ".txt");
					fout_time << ray_casting_time << endl;

					if (best_view_id < 0) {
						std::cout << "No valid view found in this iteration." << std::endl;
						best_view_id = now_best_view->id; // fallback to current view to avoid issues, or you can choose to skip movement
					}

					delete now_best_view;
					now_best_view = new View(now_view_space->views[best_view_id]);
					now_view_space->views[best_view_id].vis++;
					cout << "choose the " << best_view_id << " th view." << endl;
					//运动代价：视点id，当前代价，总体代价
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;

					//保存当前视点target
					share_data->access_directory(share_data->save_path + "/target");
					ofstream fout_target(share_data->save_path + "/target/nbv" + to_string(iterations) + ".txt");
					fout_target << now_best_view->look_at.x() << '\t' << now_best_view->look_at.y() << '\t' << now_best_view->look_at.z() << endl;

					//更新预算
					share_data->current_budget--;

					//end of SamplingNBV
				}
				else if (share_data->method_of_IG == 13) { //PriorBBXRandom
					
					if (iterations == 0) {
						//读取BBX随机泊松采样的候选观察点
						vector<Eigen::Vector3d> rand_look_at_pts;
						string rand_look_at_pts_path = share_data->environment_path + "/" + share_data->room_str + "_rand_look_at_pts.txt";
						ifstream fin_rand_look_at_pts(rand_look_at_pts_path);
						if (fin_rand_look_at_pts.is_open()) {
							while (!fin_rand_look_at_pts.eof()) {
								double x, y, z;
								fin_rand_look_at_pts >> x >> y >> z;
								if (fin_rand_look_at_pts.eof()) break;
								rand_look_at_pts.push_back(Eigen::Vector3d(x, y, z));
								if (!share_data->plant_obb.contains(rand_look_at_pts.back())) {
									cout << "views out of BBX. check." << endl;
								}
							}
							fin_rand_look_at_pts.close();
						}
						else {
							cout << "No random look at points file found." << endl;
						}
						cout << "rand_look_at_pts size: " << rand_look_at_pts.size() << endl;
						//如果候选目标多于需要的目标，裁剪掉多余的
						if (share_data->num_targets < rand_look_at_pts.size()) {
							rand_look_at_pts.resize(share_data->num_targets);
						}
						// round-robin assign candidate -> lookat targets
						std::mt19937 rng(share_data->random_seed);
						ViewAssignResult assign = assignViewsRoundRobin(share_data->dynamic_candidate_views, rand_look_at_pts, share_data->num_rounds, rng);
						//save and update the assigned views and targets
						const size_t candidateN = share_data->dynamic_candidate_views.size();
						const size_t passiveN = share_data->passive_init_views.size();
						assert(now_view_space->views.size() == passiveN + candidateN);
						std::vector<Eigen::Vector3d> lookat_all(candidateN, Eigen::Vector3d(0, 0, 0));
						for (size_t i = 0; i < assign.used_idx.size(); ++i) {
							size_t c = (size_t)assign.used_idx[i];
							if (c < candidateN) lookat_all[c] = assign.lookat[i];
						}
						std::ofstream fout_assign(share_data->save_path + "/bbx_assign.txt");
						for (size_t c = 0; c < candidateN; ++c) {
							const auto& la = lookat_all[c];
							fout_assign << la.x() << "\t" << la.y() << "\t" << la.z() << "\n";
							now_view_space->views[passiveN + c].look_at = la;
						}
						//check if the look at points are saved to file correctly
						share_data->access_directory(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series));
						ifstream fin_check_look_at(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt");
						if (!fin_check_look_at.is_open()) {
							//copy the look at file
							ofstream fout_look_at(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt");
							for (size_t k = passiveN; k < passiveN + candidateN; ++k) {
								fout_look_at << now_view_space->views[k].look_at(0) << "\t" << now_view_space->views[k].look_at(1) << "\t" << now_view_space->views[k].look_at(2) << endl;
							}
							cout << "Look at points saved to " << share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt" << endl;
						}
						else {
							cout << "Look at points already exist in " << share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt" << endl;
							//read from file to make sure we use the same look at points 
							ifstream fin_look_at(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt");
							for (size_t k = passiveN; k < passiveN + candidateN; ++k) {
								double x, y, z;
								fin_look_at >> x >> y >> z;
								now_view_space->views[k].look_at = Eigen::Vector3d(x, y, z);
								if ((now_view_space->views[k].look_at - lookat_all[k - passiveN]).norm() > 1e-6) {
									cout << "Warning: look at point from file is different from assigned result for view " << k << ". Using the one from file." << endl;
								}
							}
							cout << "Look at points read from file and updated to view space." << endl;
						}

						if (share_data->debug || share_data->show) { //显示分配结果
							auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("simple viewer"));
							viewer->setBackgroundColor(255, 255, 255);
							//viewer->addCoordinateSystem(0.2);
							viewer->initCameraParameters();
							// 0) 当前的地图点云
							//viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_final, "cloud_final");
							//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud_final");
							// 1) plant obb 红色线框（不是点）
							AddObbWireframe(viewer, share_data->plant_obb, "plant_obb", 1.0, 60.0 / 255.0, 60.0 / 255.0, 1);
							// 2) targets / lookat：红色，大点
							auto pc_targets = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
							for (size_t k = 0; k < assign.lookat.size(); ++k) {
								const Eigen::Vector3d& p = assign.lookat[k];   // 你也可以换成 targets[i]
								pc_targets->push_back(pcl::PointXYZ((float)p.x(), (float)p.y(), (float)p.z()));
							}
							pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col2(pc_targets, 255, 60, 60);
							viewer->addPointCloud<pcl::PointXYZ>(pc_targets, col2, "targets");
							viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "targets");
							// 3) views：蓝色，中点
							auto pc_views = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
							for (size_t k = 0; k < assign.used_idx.size(); ++k) {
								int vidx = assign.used_idx[k];
								Eigen::Vector3d cam = share_data->dynamic_candidate_views[vidx];
								pc_views->push_back(pcl::PointXYZ((float)cam.x(), (float)cam.y(), (float)cam.z()));
							}
							pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col3(pc_views, 80, 160, 255);
							viewer->addPointCloud<pcl::PointXYZ>(pc_views, col3, "views");
							viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "views");
							cout << "Number of assigned views: " << assign.used_idx.size() << endl;
							// 4) rays：每条匹配线段上采样若干点，绿色，小点，形成虚线箭头的效果
							auto pc_rays = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
							int N = 30;                 // 每条匹配线段采样点数（越大越密）
							for (size_t k = 0; k < assign.used_idx.size(); ++k) {
								int vidx = assign.used_idx[k];
								Eigen::Vector3d cam = share_data->dynamic_candidate_views[vidx];
								Eigen::Vector3d tgt = assign.lookat[k];
								Eigen::Vector3d d = tgt - cam;
								double len = d.norm();
								if (len < 1e-9) continue;
								for (int i = 0; i <= 30; ++i) {
									double t = double(i) / double(N);         // [0,1]
									Eigen::Vector3d p = cam + t * d;
									pc_rays->push_back(pcl::PointXYZ((float)p.x(), (float)p.y(), (float)p.z()));
								}
							}
							cout << "Number of ray points: " << pc_rays->points.size() << endl;
							pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col4(pc_rays, 60, 255, 120);
							viewer->addPointCloud<pcl::PointXYZ>(pc_rays, col4, "rays");
							viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "rays");
							while (!viewer->wasStopped()) {
								viewer->spinOnce(10);
								std::this_thread::sleep_for(std::chrono::milliseconds(10));
							}
						}

						//!!!Debug用，只生成look at candidate就结束!!!
						if (share_data->debug) {
							cout << "Debug mode: only generate look at candidates and then exit." << endl;
							status = Over;
							break;
						}

						//记录一下最后一个被访问的passive视点的id，后续如果需要重新访问可以直接从这个id开始
						share_data->last_passive_view_id = now_best_view->id;
						//如果是动态情况才需要随机选
						for (int i = 0; i < now_view_space->views.size(); i++) {
							if ((now_view_space->views[i].look_at).norm() < 1e-6) { //如果look_at是(0,0,0)，说明这个视点没有被分配到目标，直接设置为不可达，避免后续计算
								now_view_space->views[i].can_move = false;
							}
						}
					}

					//收集所有没有访问过的视点，并且这些视点是可达的
					std::vector<int> now_candidates;
					for (int i = 0; i < (int)now_view_space->views.size(); ++i) {
						if (!now_view_space->views[i].vis && now_view_space->views[i].can_move)
							now_candidates.push_back(i);
					}
					//如果没有这样的视点了，说明所有的视点要么访问过了，要么不可达了，这时你可以选择：break / return / 或者放宽条件（比如允许重复访问）
					int next_id = -1;
					if (now_candidates.empty()) {
						cout << "warning: no available next view (all visited or unreachable)" << endl;
						next_id = now_best_view->id; //保持原地不动
					}
					else {
						next_id = now_candidates[rand() % now_candidates.size()];
					}
					now_view_space->views[next_id].vis++;
					//now_view_space->views[next_id].can_move = true;
					delete now_best_view;
					now_best_view = new View(now_view_space->views[next_id]);
					cout << "choose the " << next_id << "th view." << endl;
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;

					if (share_data->set_of_num_best_views_for_compare.count(iterations + 2)) {
						//计算一下按oneshot的长度
						vector<int> view_set;
						for (int i = 0; i < now_view_space->views.size(); i++) {
							if (now_view_space->views[i].vis) view_set.push_back(i);
						}
						vector<int> view_set_label_group1, view_set_label_group2;
						vector<int> view_id_path_group1, view_id_path_group2;
						//根据当前视点Z高度分上下两层，确保G1是当前视点层
						const double z_split = share_data->view_obb.c(2);
						if (now_view_space->views[share_data->last_passive_view_id].init_pos(2) >= z_split) {
							//初始视点在上层，上层是G1
							for (auto& view_id : view_set) {
								if (now_view_space->views[view_id].init_pos(2) >= z_split) {
									view_set_label_group1.push_back(view_id);
								}
								else {
									view_set_label_group2.push_back(view_id);
								}
							}
						}
						else {
							//初始视点在下层，下层是G1
							for (auto& view_id : view_set) {
								if (now_view_space->views[view_id].init_pos(2) < z_split) {
									view_set_label_group1.push_back(view_id);
								}
								else {
									view_set_label_group2.push_back(view_id);
								}
							}
						}
						//先规划组1路径，起点是当前视点
						view_set_label_group1.insert(view_set_label_group1.begin(), share_data->last_passive_view_id); //把last_passive_view_id视点放到组1开头，确保路径从当前视点开始
						if (view_set_label_group1.size() >= 2) { //2个以上视点才规划
							int start_view_id = view_set_label_group1.front();
							//保证层切换和运动顺溜，强制终点（行最远 + z接近分层平面）
							const double alpha = 0.5; //距离和Z奖励的权重，可以调整
							const Eigen::Vector3d p0 = now_view_space->views[start_view_id].init_pos;
							double best_score = -1e100;
							int end_view_id = -1;
							for (int i = 1; i < view_set_label_group1.size(); ++i) { // 从1开始，跳过起点
								int vid = view_set_label_group1[i];
								const Eigen::Vector3d p = now_view_space->views[vid].init_pos;
								double dist_norm = (p.head<2>() - p0.head<2>()).norm() / std::max(share_data->view_obb.extent(0), share_data->view_obb.extent(1));          // ~0..1
								double dz_norm = std::abs(p(2) - z_split) / (share_data->view_obb.extent(2) * 0.5); // ~0..1
								double score = dist_norm - alpha * dz_norm;   // alpha 0.2~0.5: 轻偏好；alpha 1: 强偏好
								if (score > best_score) {
									best_score = score;
									end_view_id = vid;
								}
							}
							//运行全局路径规划，强制终点
							Global_Path_Planner* global_path_planner_group1 = new Global_Path_Planner(share_data, now_view_space->views, view_set_label_group1, start_view_id, end_view_id);
							global_path_planner_group1->solve();
							view_id_path_group1 = global_path_planner_group1->get_path_id_set();
							delete global_path_planner_group1;
						}
						else if (view_set_label_group1.size() == 1) { //1个视点就不动了
							view_id_path_group1 = view_set_label_group1;
						}
						//再规划组2路径，起点是组1终点，由于组1一定至少有个当前视点，所以没问题
						view_set_label_group2.insert(view_set_label_group2.begin(), view_id_path_group1.back()); //把组1终点放到组2开头，确保组2路径从组1终点开始
						if (view_set_label_group2.size() >= 2) {
							int start_view_id = view_set_label_group2.front();
							//保证运动顺溜，强制终点（行最远）
							const Eigen::Vector3d p0 = now_view_space->views[start_view_id].init_pos;
							double best_score = -1e100;
							int end_view_id = -1;
							for (int i = 1; i < view_set_label_group2.size(); ++i) { // 从1开始，跳过起点
								int vid = view_set_label_group2[i];
								const Eigen::Vector3d p = now_view_space->views[vid].init_pos;
								double dist = (p - p0).norm(); // 距离项：越远越好
								double score = dist;
								if (score > best_score) {
									best_score = score;
									end_view_id = vid;
								}
							}
							Global_Path_Planner* global_path_planner_group2 = new Global_Path_Planner(share_data, now_view_space->views, view_set_label_group2, start_view_id, end_view_id);
							global_path_planner_group2->solve();
							view_id_path_group2 = global_path_planner_group2->get_path_id_set();
							delete global_path_planner_group2;
						}
						else if (view_set_label_group2.size() == 1) {
							view_id_path_group2.clear(); //1个视点就空置，因为组1的终点已经是它了，不需要再动了
						}
						//最终路径是两个的拼接，两个出来的结果已经去掉了起点
						vector<int> view_id_path_full = view_id_path_group1;
						for (auto& view_id : view_id_path_group2) {
							view_id_path_full.push_back(view_id);
						}
						//保存到单独的文件夹里，方便查看
						share_data->access_directory(share_data->save_path + "/movement/" + to_string(iterations + 2));
						double total_distance = share_data->passive_map_cost; //如果有passive阶段，先把passive阶段的代价加上
						ofstream fout_passive(share_data->save_path + "/movement/" + to_string(iterations + 2) + "/path" + to_string(-1) + ".txt");
						fout_passive << share_data->last_passive_view_id << '\t' << 0.0 << '\t' << total_distance << endl; //起点
						vector<int> view_id_path_with_start = view_id_path_full;
						view_id_path_with_start.insert(view_id_path_with_start.begin(), share_data->last_passive_view_id); //把起点放到路径开头
						assert(view_id_path_with_start.size() == static_cast<size_t>(iterations + 2));
						for (int i = 0; i + 1 < view_id_path_with_start.size(); i++) {
							ofstream fout(share_data->save_path + "/movement/" + to_string(iterations + 2) + "/path" + to_string(i) + ".txt");
							pair<int, double> local_path = get_local_path(now_view_space->views[view_id_path_with_start[i]].init_pos.eval(), now_view_space->views[view_id_path_with_start[i + 1]].init_pos.eval(), share_data->object_center_world.eval(), share_data->predicted_size * sqrt(2));
							if (local_path.first < 0) cout << "local path wrong." << endl;
							total_distance += local_path.second;
							fout << view_id_path_with_start[i + 1] << '\t' << local_path.second << '\t' << total_distance << endl;
						}
					}
					//end of PriorBBXRandom
				}
				else if (share_data->method_of_IG == 12) { //PriorPassiveRandom
					//先建立passive map
					if (iterations + 1 < share_data->passive_init_views.size()) {
						//如果是passive按顺序读
						int next_id = now_best_view->id;
						if (share_data->first_view_id == 0) {
							next_id++; //Order
							if (next_id >= share_data->passive_init_views.size()) {
								cout << "error: no next view in Order method. check num_of_nbvs_combined == passive_init_views.size()-1" << endl;
								next_id = 0;
							}
						}
						else if (share_data->first_view_id == share_data->passive_init_views.size() - 1) {
							next_id--; //Reverse Order
							if (next_id < 0) {
								cout << "error: no next view in Order method. check num_of_nbvs_combined == passive_init_views.size()-1" << endl;
								next_id = share_data->passive_init_views.size() - 1;
							}
						}
						else {
							cout << "error: first_view_id is not correct in Order or Reverse Order method." << endl;
							next_id = 0;
						}
						cout << "next view id is " << next_id << endl;
						now_view_space->views[next_id].vis++;
						delete now_best_view;
						now_best_view = new View(now_view_space->views[next_id]);
						//运动代价：视点id，当前代价，总体代价
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					}
					else {
						//如果是passive的都访问过了，第一次先检查一下文件里保存的look at点是否正确
						if (iterations + 1 == share_data->passive_init_views.size()) {
							cout << "all passive views have been visited, start random selection." << endl;

							//Create passive map 
							const double res = share_data->ground_truth_resolution * share_data->frontier_resolution_factor;
							const OBB& obb = share_data->plant_obb;
							octomap::ColorOcTree passive_map(res);
							cout << "create passive map with resolution " << res << endl;
							//OBB corners -> AABB (world)
							auto cs = obb.corners();
							Eigen::Vector3d bb_min = cs[0];
							Eigen::Vector3d bb_max = cs[0];
							for (int i = 1; i < 8; ++i) {
								bb_min = bb_min.cwiseMin(cs[i]);
								bb_max = bb_max.cwiseMax(cs[i]);
							}
							//AABB -> key range (IMPORTANT: pad a little to avoid boundary miss)
							const double pad = 2.0 * res;  // 壳厚度同量级的 padding，避免角落漏 key
							octomap::OcTreeKey kmin, kmax;
							bool ok_min = passive_map.coordToKeyChecked(octomap::point3d(bb_min.x() - pad, bb_min.y() - pad, bb_min.z() - pad), kmin);
							bool ok_max = passive_map.coordToKeyChecked(octomap::point3d(bb_max.x() + pad, bb_max.y() + pad, bb_max.z() + pad), kmax);
							if (!ok_min || !ok_max) {
								std::cout << "Warning: view OBB AABB is out of octomap bounds when converting to keys." << std::endl;
								// continue;
							}
							//Set BBX to unkown
							for (unsigned int kx = kmin[0]; kx <= kmax[0]; ++kx) {
								for (unsigned int ky = kmin[1]; ky <= kmax[1]; ++ky) {
									for (unsigned int kz = kmin[2]; kz <= kmax[2]; ++kz) {
										octomap::OcTreeKey key(kx, ky, kz);
										octomap::point3d c = passive_map.keyToCoord(key);
										Eigen::Vector3d p(c.x(), c.y(), c.z());
										if (!obb.contains(p)) continue;
										passive_map.setNodeValue(key, (float)0, true); //初始化概率0.5，即logodds为0
									}
								}
							}
							passive_map.updateInnerOccupancy();
							//Insert all passive views' pointclouds
							for (int i = 0; i < share_data->passive_init_views.size(); ++i) {
								octomap::Pointcloud cloud_octo;
								for (auto p : share_data->clouds[i]->points) {
									cloud_octo.push_back(p.x, p.y, p.z);
								}
								passive_map.insertPointCloud(cloud_octo, octomap::point3d(now_view_space->views[i].init_pos(0), now_view_space->views[i].init_pos(1), now_view_space->views[i].init_pos(2)), share_data->ray_max_dis, true, false);
								for (auto p : share_data->clouds[i]->points) {
									passive_map.integrateNodeColor(p.x, p.y, p.z, p.r, p.g, p.b);
								}
								passive_map.updateInnerOccupancy();
							}
							//passive_map.write(share_data->save_path + "/passive_map.ot");

							//Get frontier of surface and unknown
							vector<octomap::point3d> frontier_pts;
							for (unsigned int kx = kmin[0]; kx <= kmax[0]; ++kx) {
								for (unsigned int ky = kmin[1]; ky <= kmax[1]; ++ky) {
									for (unsigned int kz = kmin[2]; kz <= kmax[2]; ++kz) {
										octomap::OcTreeKey key(kx, ky, kz);
										octomap::point3d pt = passive_map.keyToCoord(key);
										octomap::OcTreeNode* node = passive_map.search(key); //确保这个key在树里，没的话会被创建成unknown
										if (node == NULL) continue;
										double occupancy = node->getOccupancy();
										if (voxel_information->is_unknown(occupancy)) {
											int node_type = frontier_check(pt, &passive_map, voxel_information, res);
											if (node_type == 2) frontier_pts.push_back(pt); //if it has a surface neibour and a free neibour
										}
									}
								}
							}
							cout << "frontier has " << frontier_pts.size() << " points." << endl;

							//FPS on frontier
							pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_frontier(new pcl::PointCloud<pcl::PointXYZ>);
							for (const auto& pt : frontier_pts) {
								pcl::PointXYZ p;
								p.x = pt.x();
								p.y = pt.y();
								p.z = pt.z();
								cloud_frontier->points.push_back(p);
							}
							cloud_frontier->width = cloud_frontier->points.size();
							cloud_frontier->height = 1;
							cloud_frontier->is_dense = false;
							std::mt19937 rng(share_data->random_seed);
							std::vector<Eigen::Vector3d> targets = farthestPointSampling(cloud_frontier, share_data->num_targets, rng);
							std::cout << "FPS targets: " << targets.size() << "\n";
							// round-robin assign candidate -> lookat targets
							ViewAssignResult assign = assignViewsRoundRobin(share_data->dynamic_candidate_views, targets, share_data->num_rounds, rng);
							//save and update the assigned views and targets
							const size_t candidateN = share_data->dynamic_candidate_views.size();
							const size_t passiveN = share_data->passive_init_views.size();
							assert(now_view_space->views.size() == passiveN + candidateN);
							std::vector<Eigen::Vector3d> lookat_all(candidateN, Eigen::Vector3d(0, 0, 0));
							for (size_t i = 0; i < assign.used_idx.size(); ++i) {
								size_t c = (size_t)assign.used_idx[i];
								if (c < candidateN) lookat_all[c] = assign.lookat[i];
							}
							std::ofstream fout_assign(share_data->save_path + "/frontier_assign.txt");
							for (size_t c = 0; c < candidateN; ++c) {
								const auto& la = lookat_all[c];
								fout_assign << la.x() << "\t" << la.y() << "\t" << la.z() << "\n";
								now_view_space->views[passiveN + c].look_at = la;
							}
							//check if the look at points are saved to file correctly
							share_data->access_directory(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series));
							ifstream fin_check_look_at(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt");
							if (!fin_check_look_at.is_open()) {
								//copy the look at file
								ofstream fout_look_at(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt");
								for (size_t k = passiveN; k < passiveN + candidateN; ++k) {
									fout_look_at << now_view_space->views[k].look_at(0) << "\t" << now_view_space->views[k].look_at(1) << "\t" << now_view_space->views[k].look_at(2) << endl;
								}
								cout << "Look at points saved to " << share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt" << endl;
							}
							else {
								cout << "Look at points already exist in " << share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt" << endl;
								//read from file to make sure we use the same look at points 
								ifstream fin_look_at(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt");
								for (size_t k = passiveN; k < passiveN + candidateN; ++k) {
									double x, y, z;
									fin_look_at >> x >> y >> z;
									now_view_space->views[k].look_at = Eigen::Vector3d(x, y, z);
									if ((now_view_space->views[k].look_at - lookat_all[k - passiveN]).norm() > 1e-6) {
										cout << "Warning: look at point from file is different from assigned result for view " << k << ". Using the one from file." << endl;
									}
								}
								cout << "Look at points read from file and updated to view space." << endl;
							}

							if (share_data->debug || share_data->show) { //显示分配结果
								auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("simple viewer"));
								viewer->setBackgroundColor(255, 255, 255);
								//viewer->addCoordinateSystem(0.2);
								viewer->initCameraParameters();
								// 0) 当前的地图点云
								viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_final, "cloud_final");
								viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_final");
								// 1) plant obb 红色线框（不是点）
								AddObbWireframe(viewer, share_data->plant_obb, "plant_obb", 1.0, 60.0 / 255.0, 60.0 / 255.0, 1);
								// 1) 原始 downsample 点云：灰色，小点
								//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col1(cloud_frontier, 220, 220, 220);
								//viewer->addPointCloud<pcl::PointXYZ>(cloud_frontier, col1, "cloud_frontier");
								//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_frontier");
								// 2) targets / lookat：红色，大点
								auto pc_targets = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
								for (size_t k = 0; k < assign.lookat.size(); ++k) {
									const Eigen::Vector3d& p = assign.lookat[k];   // 你也可以换成 targets[i]
									pc_targets->push_back(pcl::PointXYZ((float)p.x(), (float)p.y(), (float)p.z()));
								}
								pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col2(pc_targets, 255, 60, 60);
								viewer->addPointCloud<pcl::PointXYZ>(pc_targets, col2, "targets");
								viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "targets");
								// 3) views：蓝色，中点
								auto pc_views = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
								for (size_t k = 0; k < assign.used_idx.size(); ++k) {
									int vidx = assign.used_idx[k];
									Eigen::Vector3d cam = share_data->dynamic_candidate_views[vidx];
									pc_views->push_back(pcl::PointXYZ((float)cam.x(), (float)cam.y(), (float)cam.z()));
								}
								pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col3(pc_views, 80, 160, 255);
								viewer->addPointCloud<pcl::PointXYZ>(pc_views, col3, "views");
								viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "views");
								cout << "Number of assigned views: " << assign.used_idx.size() << endl;
								// 4) rays：每条匹配线段上采样若干点，绿色，小点，形成虚线箭头的效果
								auto pc_rays = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
								int N = 30;                 // 每条匹配线段采样点数（越大越密）
								for (size_t k = 0; k < assign.used_idx.size(); ++k) {
									int vidx = assign.used_idx[k];
									Eigen::Vector3d cam = share_data->dynamic_candidate_views[vidx];
									Eigen::Vector3d tgt = assign.lookat[k];
									Eigen::Vector3d d = tgt - cam;
									double len = d.norm();
									if (len < 1e-9) continue;
									for (int i = 0; i <= 30; ++i) {
										double t = double(i) / double(N);         // [0,1]
										Eigen::Vector3d p = cam + t * d;
										pc_rays->push_back(pcl::PointXYZ((float)p.x(), (float)p.y(), (float)p.z()));
									}
								}
								cout << "Number of ray points: " << pc_rays->points.size() << endl;
								pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col4(pc_rays, 60, 255, 120);
								viewer->addPointCloud<pcl::PointXYZ>(pc_rays, col4, "rays");
								viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "rays");
								while (!viewer->wasStopped()) {
									viewer->spinOnce(10);
									std::this_thread::sleep_for(std::chrono::milliseconds(10));
								}
							}

							//!!!Debug用，只生成look at candidate就结束!!!
							if (share_data->debug) {
								cout << "Debug mode: only generate look at candidates and then exit." << endl;
								status = Over;
								break;
							}

							//记录一下最后一个被访问的passive视点的id，后续如果需要重新访问可以直接从这个id开始
							share_data->last_passive_view_id = now_best_view->id;
							//如果是动态情况才需要随机选
							for (int i = 0; i < now_view_space->views.size(); i++) {
								if ((now_view_space->views[i].look_at).norm() < 1e-6) { //如果look_at是(0,0,0)，说明这个视点没有被分配到目标，直接设置为不可达，避免后续计算
									now_view_space->views[i].can_move = false;
								}
							}
						}

						//收集所有没有访问过的视点，并且这些视点是可达的
						std::vector<int> now_candidates;
						for (int i = 0; i < (int)now_view_space->views.size(); ++i) {
							if (!now_view_space->views[i].vis && now_view_space->views[i].can_move)
								now_candidates.push_back(i);
						}
						//如果没有这样的视点了，说明所有的视点要么访问过了，要么不可达了，这时你可以选择：break / return / 或者放宽条件（比如允许重复访问）
						int next_id = -1;
						if (now_candidates.empty()) {
							cout << "warning: no available next view (all visited or unreachable)" << endl;
							next_id = now_best_view->id; //保持原地不动
						}
						else {
							next_id = now_candidates[rand() % now_candidates.size()];
						}
						now_view_space->views[next_id].vis++;
						//now_view_space->views[next_id].can_move = true;
						delete now_best_view;
						now_best_view = new View(now_view_space->views[next_id]);
						cout << "choose the " << next_id << "th view." << endl;
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;

						if (share_data->set_of_num_best_views_for_compare.count(iterations + 2)) {
							//计算一下按oneshot的长度
							vector<int> view_set;
							for (int i = 0; i < now_view_space->views.size(); i++) {
								if (i < share_data->passive_init_views.size()) continue;
								if (now_view_space->views[i].vis) view_set.push_back(i);
							}
							vector<int> view_set_label_group1, view_set_label_group2;
							vector<int> view_id_path_group1, view_id_path_group2;
							//根据当前视点Z高度分上下两层，确保G1是当前视点层
							const double z_split = share_data->view_obb.c(2);
							if (now_view_space->views[share_data->last_passive_view_id].init_pos(2) >= z_split) {
								//初始视点在上层，上层是G1
								for (auto& view_id : view_set) {
									if (now_view_space->views[view_id].init_pos(2) >= z_split) {
										view_set_label_group1.push_back(view_id);
									}
									else {
										view_set_label_group2.push_back(view_id);
									}
								}
							}
							else {
								//初始视点在下层，下层是G1
								for (auto& view_id : view_set) {
									if (now_view_space->views[view_id].init_pos(2) < z_split) {
										view_set_label_group1.push_back(view_id);
									}
									else {
										view_set_label_group2.push_back(view_id);
									}
								}
							}
							//先规划组1路径，起点是当前视点
							view_set_label_group1.insert(view_set_label_group1.begin(), share_data->last_passive_view_id); //把last_passive_view_id视点放到组1开头，确保路径从当前视点开始
							if (view_set_label_group1.size() >= 2) { //2个以上视点才规划
								int start_view_id = view_set_label_group1.front();
								//保证层切换和运动顺溜，强制终点（行最远 + z接近分层平面）
								const double alpha = 0.5; //距离和Z奖励的权重，可以调整
								const Eigen::Vector3d p0 = now_view_space->views[start_view_id].init_pos;
								double best_score = -1e100;
								int end_view_id = -1;
								for (int i = 1; i < view_set_label_group1.size(); ++i) { // 从1开始，跳过起点
									int vid = view_set_label_group1[i];
									const Eigen::Vector3d p = now_view_space->views[vid].init_pos;
									double dist_norm = (p.head<2>() - p0.head<2>()).norm() / std::max(share_data->view_obb.extent(0), share_data->view_obb.extent(1));          // ~0..1
									double dz_norm = std::abs(p(2) - z_split) / (share_data->view_obb.extent(2) * 0.5); // ~0..1
									double score = dist_norm - alpha * dz_norm;   // alpha 0.2~0.5: 轻偏好；alpha 1: 强偏好
									if (score > best_score) {
										best_score = score;
										end_view_id = vid;
									}
								}
								//运行全局路径规划，强制终点
								Global_Path_Planner* global_path_planner_group1 = new Global_Path_Planner(share_data, now_view_space->views, view_set_label_group1, start_view_id, end_view_id);
								global_path_planner_group1->solve();
								view_id_path_group1 = global_path_planner_group1->get_path_id_set();
								delete global_path_planner_group1;
							}
							else if (view_set_label_group1.size() == 1) { //1个视点就不动了
								view_id_path_group1 = view_set_label_group1;
							}
							//再规划组2路径，起点是组1终点，由于组1一定至少有个当前视点，所以没问题
							view_set_label_group2.insert(view_set_label_group2.begin(), view_id_path_group1.back()); //把组1终点放到组2开头，确保组2路径从组1终点开始
							if (view_set_label_group2.size() >= 2) {
								int start_view_id = view_set_label_group2.front();
								//保证运动顺溜，强制终点（行最远）
								const Eigen::Vector3d p0 = now_view_space->views[start_view_id].init_pos;
								double best_score = -1e100;
								int end_view_id = -1;
								for (int i = 1; i < view_set_label_group2.size(); ++i) { // 从1开始，跳过起点
									int vid = view_set_label_group2[i];
									const Eigen::Vector3d p = now_view_space->views[vid].init_pos;
									double dist = (p - p0).norm(); // 距离项：越远越好
									double score = dist;
									if (score > best_score) {
										best_score = score;
										end_view_id = vid;
									}
								}
								Global_Path_Planner* global_path_planner_group2 = new Global_Path_Planner(share_data, now_view_space->views, view_set_label_group2, start_view_id, end_view_id);
								global_path_planner_group2->solve();
								view_id_path_group2 = global_path_planner_group2->get_path_id_set();
								delete global_path_planner_group2;
							}
							else if (view_set_label_group2.size() == 1) {
								view_id_path_group2.clear(); //1个视点就空置，因为组1的终点已经是它了，不需要再动了
							}
							//最终路径是两个的拼接，两个出来的结果已经去掉了起点
							vector<int> view_id_path_full = view_id_path_group1;
							for (auto& view_id : view_id_path_group2) {
								view_id_path_full.push_back(view_id);
							}
							//保存到单独的文件夹里，方便查看
							share_data->access_directory(share_data->save_path + "/movement/" + to_string(iterations + 2));
							double total_distance = share_data->passive_map_cost; //如果有passive阶段，先把passive阶段的代价加上
							ofstream fout_passive(share_data->save_path + "/movement/" + to_string(iterations + 2) + "/path" + to_string(-1) + ".txt");
							fout_passive << share_data->last_passive_view_id << '\t' << 0.0 << '\t' << total_distance << endl; //起点
							vector<int> view_id_path_with_start = view_id_path_full;
							view_id_path_with_start.insert(view_id_path_with_start.begin(), share_data->last_passive_view_id); //把起点放到路径开头
							assert(view_id_path_with_start.size() == static_cast<size_t>(iterations + 2 - share_data->passive_init_views.size()));
							for (int i = 0; i + 1 < view_id_path_with_start.size(); i++) {
								ofstream fout(share_data->save_path + "/movement/" + to_string(iterations + 2) + "/path" + to_string(i) + ".txt");
								pair<int, double> local_path = get_local_path(now_view_space->views[view_id_path_with_start[i]].init_pos.eval(), now_view_space->views[view_id_path_with_start[i + 1]].init_pos.eval(), share_data->object_center_world.eval(), share_data->predicted_size * sqrt(2));
								if (local_path.first < 0) cout << "local path wrong." << endl;
								total_distance += local_path.second;
								fout << view_id_path_with_start[i + 1] << '\t' << local_path.second << '\t' << total_distance << endl;
							}
						}
					
					}
					//end of PriorPassiveRandom
				}
				else if (share_data->method_of_IG == 8) { //PriorTemporalRandom
					//先建立passive map
					if (iterations + 1 < share_data->passive_init_views.size()) {
						//如果是passive按顺序读
						int next_id = now_best_view->id;
						if (share_data->first_view_id == 0) {
							next_id++; //Order
							if (next_id >= share_data->passive_init_views.size()) {
								cout << "error: no next view in Order method. check num_of_nbvs_combined == passive_init_views.size()-1" << endl;
								next_id = 0;
							}
						}
						else if (share_data->first_view_id == share_data->passive_init_views.size() - 1) {
							next_id--; //Reverse Order
							if (next_id < 0) {
								cout << "error: no next view in Order method. check num_of_nbvs_combined == passive_init_views.size()-1" << endl;
								next_id = share_data->passive_init_views.size() - 1;
							}
						}
						else {
							cout << "error: first_view_id is not correct in Order or Reverse Order method." << endl;
							next_id = 0;
						}
						cout << "next view id is " << next_id << endl;
						now_view_space->views[next_id].vis++;
						delete now_best_view;
						now_best_view = new View(now_view_space->views[next_id]);
						//运动代价：视点id，当前代价，总体代价
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					}
					else {
						//如果是passive的都访问过了，第一次先检查一下文件里保存的look at点是否正确
						if (iterations + 1 == share_data->passive_init_views.size()) {
							cout << "all passive views have been visited, start random selection." << endl;
							//check if the look at points are saved to file correctly
							const size_t candidateN = share_data->dynamic_candidate_views.size();
							const size_t passiveN = share_data->passive_init_views.size();
							assert(now_view_space->views.size() == passiveN + candidateN);
							share_data->access_directory(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series));
							ifstream fin_check_look_at(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt");
							if (!fin_check_look_at.is_open()) {
								cout << "Run method PriorTemporalCovering Debug to get the candidate_look_ats! exit." << endl;
								status = Over;
								break;
							}
							//记录一下最后一个被访问的passive视点的id，后续如果需要重新访问可以直接从这个id开始
							share_data->last_passive_view_id = now_best_view->id;
							//如果是动态情况才需要随机选
							for (int i = 0; i < now_view_space->views.size(); i++) {
								if ((now_view_space->views[i].look_at).norm() < 1e-6) { //如果look_at是(0,0,0)，说明这个视点没有被分配到目标，直接设置为不可达，避免后续计算
									now_view_space->views[i].can_move = false;
								}
							}
						}
						
						//收集所有没有访问过的视点，并且这些视点是可达的
						std::vector<int> now_candidates;
						for (int i = 0; i < (int)now_view_space->views.size(); ++i) {
							if (!now_view_space->views[i].vis && now_view_space->views[i].can_move)
								now_candidates.push_back(i);
						}
						//如果没有这样的视点了，说明所有的视点要么访问过了，要么不可达了，这时你可以选择：break / return / 或者放宽条件（比如允许重复访问）
						int next_id = -1;
						if (now_candidates.empty()) {
							cout << "warning: no available next view (all visited or unreachable)" << endl;
							next_id = now_best_view->id; //保持原地不动
						}
						else {
							next_id = now_candidates[rand() % now_candidates.size()];
						}
						now_view_space->views[next_id].vis++;
						//now_view_space->views[next_id].can_move = true;
						delete now_best_view;
						now_best_view = new View(now_view_space->views[next_id]);
						cout << "choose the " << next_id << "th view." << endl;
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;

						if (share_data->set_of_num_best_views_for_compare.count(iterations + 2)) {
							//计算一下按oneshot的长度
							vector<int> view_set;
							for (int i = 0; i < now_view_space->views.size(); i++) {
								if (i < share_data->passive_init_views.size()) continue;
								if (now_view_space->views[i].vis) view_set.push_back(i);
							}
							vector<int> view_set_label_group1, view_set_label_group2;
							vector<int> view_id_path_group1, view_id_path_group2;
							//根据当前视点Z高度分上下两层，确保G1是当前视点层
							const double z_split = share_data->view_obb.c(2);
							if (now_view_space->views[share_data->last_passive_view_id].init_pos(2) >= z_split) {
								//初始视点在上层，上层是G1
								for (auto& view_id : view_set) {
									if (now_view_space->views[view_id].init_pos(2) >= z_split) {
										view_set_label_group1.push_back(view_id);
									}
									else {
										view_set_label_group2.push_back(view_id);
									}
								}
							}
							else {
								//初始视点在下层，下层是G1
								for (auto& view_id : view_set) {
									if (now_view_space->views[view_id].init_pos(2) < z_split) {
										view_set_label_group1.push_back(view_id);
									}
									else {
										view_set_label_group2.push_back(view_id);
									}
								}
							}
							//先规划组1路径，起点是当前视点
							view_set_label_group1.insert(view_set_label_group1.begin(), share_data->last_passive_view_id); //把last_passive_view_id视点放到组1开头，确保路径从当前视点开始
							if (view_set_label_group1.size() >= 2) { //2个以上视点才规划
								int start_view_id = view_set_label_group1.front();
								//保证层切换和运动顺溜，强制终点（行最远 + z接近分层平面）
								const double alpha = 0.5; //距离和Z奖励的权重，可以调整
								const Eigen::Vector3d p0 = now_view_space->views[start_view_id].init_pos;
								double best_score = -1e100;
								int end_view_id = -1;
								for (int i = 1; i < view_set_label_group1.size(); ++i) { // 从1开始，跳过起点
									int vid = view_set_label_group1[i];
									const Eigen::Vector3d p = now_view_space->views[vid].init_pos;
									double dist_norm = (p.head<2>() - p0.head<2>()).norm() / std::max(share_data->view_obb.extent(0), share_data->view_obb.extent(1));          // ~0..1
									double dz_norm = std::abs(p(2) - z_split) / (share_data->view_obb.extent(2) * 0.5); // ~0..1
									double score = dist_norm - alpha * dz_norm;   // alpha 0.2~0.5: 轻偏好；alpha 1: 强偏好
									if (score > best_score) {
										best_score = score;
										end_view_id = vid;
									}
								}
								//运行全局路径规划，强制终点
								Global_Path_Planner* global_path_planner_group1 = new Global_Path_Planner(share_data, now_view_space->views, view_set_label_group1, start_view_id, end_view_id);
								global_path_planner_group1->solve();
								view_id_path_group1 = global_path_planner_group1->get_path_id_set();
								delete global_path_planner_group1;
							}
							else if (view_set_label_group1.size() == 1) { //1个视点就不动了
								view_id_path_group1 = view_set_label_group1;
							}
							//再规划组2路径，起点是组1终点，由于组1一定至少有个当前视点，所以没问题
							view_set_label_group2.insert(view_set_label_group2.begin(), view_id_path_group1.back()); //把组1终点放到组2开头，确保组2路径从组1终点开始
							if (view_set_label_group2.size() >= 2) {
								int start_view_id = view_set_label_group2.front();
								//保证运动顺溜，强制终点（行最远）
								const Eigen::Vector3d p0 = now_view_space->views[start_view_id].init_pos;
								double best_score = -1e100;
								int end_view_id = -1;
								for (int i = 1; i < view_set_label_group2.size(); ++i) { // 从1开始，跳过起点
									int vid = view_set_label_group2[i];
									const Eigen::Vector3d p = now_view_space->views[vid].init_pos;
									double dist = (p - p0).norm(); // 距离项：越远越好
									double score = dist;
									if (score > best_score) {
										best_score = score;
										end_view_id = vid;
									}
								}
								Global_Path_Planner* global_path_planner_group2 = new Global_Path_Planner(share_data, now_view_space->views, view_set_label_group2, start_view_id, end_view_id);
								global_path_planner_group2->solve();
								view_id_path_group2 = global_path_planner_group2->get_path_id_set();
								delete global_path_planner_group2;
							}
							else if (view_set_label_group2.size() == 1) {
								view_id_path_group2.clear(); //1个视点就空置，因为组1的终点已经是它了，不需要再动了
							}
							//最终路径是两个的拼接，两个出来的结果已经去掉了起点
							vector<int> view_id_path_full = view_id_path_group1;
							for (auto& view_id : view_id_path_group2) {
								view_id_path_full.push_back(view_id);
							}
							//保存到单独的文件夹里，方便查看
							share_data->access_directory(share_data->save_path + "/movement/" + to_string(iterations + 2));
							double total_distance = share_data->passive_map_cost; //如果有passive阶段，先把passive阶段的代价加上
							ofstream fout_passive(share_data->save_path + "/movement/" + to_string(iterations + 2) + "/path" + to_string(-1) + ".txt");
							fout_passive << share_data->last_passive_view_id << '\t' << 0.0 << '\t' << total_distance << endl; //起点
							vector<int> view_id_path_with_start = view_id_path_full;
							view_id_path_with_start.insert(view_id_path_with_start.begin(), share_data->last_passive_view_id); //把起点放到路径开头
							assert(view_id_path_with_start.size() == static_cast<size_t>(iterations + 2 - share_data->passive_init_views.size()));
							for (int i = 0; i + 1 < view_id_path_with_start.size(); i++) {
								ofstream fout(share_data->save_path + "/movement/" + to_string(iterations + 2) + "/path" + to_string(i) + ".txt");
								pair<int, double> local_path = get_local_path(now_view_space->views[view_id_path_with_start[i]].init_pos.eval(), now_view_space->views[view_id_path_with_start[i + 1]].init_pos.eval(), share_data->object_center_world.eval(), share_data->predicted_size * sqrt(2));
								if (local_path.first < 0) cout << "local path wrong." << endl;
								total_distance += local_path.second;
								fout << view_id_path_with_start[i + 1] << '\t' << local_path.second << '\t' << total_distance << endl;
							}
						}

					}
					//end of PriorTemporalRandom
				}
				else if (share_data->method_of_IG == 11) { //PriorTemporalCovering
					int next_id = now_best_view->id;
					if (share_data->first_view_id == 0) {
						next_id++; //Order
						if (next_id >= share_data->passive_init_views.size()) {
							cout << "error: no next view in Order method. check num_of_nbvs_combined == passive_init_views.size()-1" << endl;
							next_id = 0;
						}
					}
					else if (share_data->first_view_id == share_data->passive_init_views.size() - 1) {
						next_id--; //Reverse Order
						if (next_id < 0) {
							cout << "error: no next view in Order method. check num_of_nbvs_combined == passive_init_views.size()-1" << endl;
							next_id = share_data->passive_init_views.size() - 1;
						}
					}
					else {
						cout << "error: first_view_id is not correct in Order or Reverse Order method." << endl;
						next_id = 0;
					}
					cout << "next view id is " << next_id << endl;
					now_view_space->views[next_id].vis++;
					delete now_best_view;
					now_best_view = new View(now_view_space->views[next_id]);
					//运动代价：视点id，当前代价，总体代价
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
				}
				else if (share_data->method_of_IG == 6) { //NBV-NET
					share_data->access_directory(share_data->nbv_net_path + "/log");
					ifstream ftest;
					do {
						//ftest.open(share_data->nbv_net_path + "/log/ready.txt"); //串行版本
						ftest.open(share_data->nbv_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + "_ready.txt"); //并行版本
					} while (!ftest.is_open());
					ftest.close();
					ifstream fin(share_data->nbv_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + ".txt");
					int id;
					fin >> id;
					cout << "next view id is " << id << endl;
					now_view_space->views[id].vis++;
					delete now_best_view;
					now_best_view = new View(now_view_space->views[id]);
					//运动代价：视点id，当前代价，总体代价
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					//更新标志文件
					this_thread::sleep_for(chrono::seconds(1));
					//int removed = remove((share_data->nbv_net_path + "/log/ready.txt").c_str()); //串行版本
					int removed = remove((share_data->nbv_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + "_ready.txt").c_str()); //并行版本
					if (removed != 0) cout << "cannot remove ready.txt." << endl;
				}
				else if (share_data->method_of_IG == 9) { //PCNBV
					share_data->access_directory(share_data->pcnbv_path + "/log");
					ifstream ftest;
					do {
						//ftest.open(share_data->pcnbv_path + "/log/ready.txt"); //串行版本
						ftest.open(share_data->pcnbv_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + "_ready.txt"); //并行版本
					} while (!ftest.is_open());
					ftest.close();
					ifstream fin(share_data->pcnbv_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + ".txt");
					int id;
					fin >> id;
					cout << "next view id is " << id << endl;
					now_view_space->views[id].vis++;
					delete now_best_view;
					now_best_view = new View(now_view_space->views[id]);
					//运动代价：视点id，当前代价，总体代价
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					//更新标志文件
					this_thread::sleep_for(chrono::seconds(1));
					//int removed = remove((share_data->pcnbv_path + "/log/ready.txt").c_str()); //串行版本
					int removed = remove((share_data->pcnbv_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + "_ready.txt").c_str()); //并行版本
					if (removed != 0) cout << "cannot remove ready.txt." << endl;
				}
				else if (share_data->method_of_IG == 7) { //(MA-)SCVP
					if (iterations == 0 + share_data->num_of_nbvs_combined) {
						//视点覆盖结果
						vector<int> view_set_label;
						if (share_data->use_history_model_for_covering) {
							//读取NRCIP结果
							share_data->access_directory(share_data->nricp_path + "/log");
							ifstream ftest;
							do {
								ftest.open(share_data->nricp_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_ready.txt"); //并行版本
							} while (!ftest.is_open());
							ftest.close();
							//读取参考点云
							pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_nricp_in(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
							if (pcl::io::loadPLYFile<pcl::PointXYZRGBNormal>(share_data->nricp_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_nricp.ply", *cloud_nricp_in) == -1) {
								cout << "Error loading NRICP reference cloud." << endl;
							}
							pcl::io::savePCDFileBinary(share_data->save_path + "/current_nricp.pcd", *cloud_nricp_in);
							//NRICP结果转换为XYZ格式
							pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nricp(new pcl::PointCloud<pcl::PointXYZ>);
							pcl::copyPointCloud(*cloud_nricp_in, *cloud_nricp);
							cout << "NRICP reference cloud has " << cloud_nricp->points.size() << " points." << endl;	

							//构建FPS采样点，即H\C，历史观测去掉当前观测上采样候选look at
							pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_final_xyz(new pcl::PointCloud<pcl::PointXYZ>);
							pcl::copyPointCloud(*share_data->cloud_final, *cloud_final_xyz);
							// 体素大小同时作用于FPS和inflation
							const float inflation_voxel_size = share_data->ground_truth_resolution * share_data->inflation_resolution_factor;
							// 连续距离阈值（米）
							const float near_current = share_data->ground_truth_resolution * share_data->near_current_inflation_distance_factor;
							const float far_history = share_data->ground_truth_resolution * share_data->far_history_inflation_distance_factor;
							// 预计算球形偏移（离散膨胀核）
							const auto off_near = MakeSphereOffsets(near_current, inflation_voxel_size);
							const auto off_far = MakeSphereOffsets(far_history, inflation_voxel_size);
							// 可选：ROI（把 object_center +/- predicted_size 的盒子映射到体素坐标，避免无限扩张）
							bool use_roi = true;
							VoxelKey roi_min, roi_max;
							if (use_roi) {
								pcl::PointXYZ pmin(
									share_data->object_center_world(0) - share_data->predicted_size,
									share_data->object_center_world(1) - share_data->predicted_size,
									share_data->object_center_world(2) - share_data->predicted_size
								);
								pcl::PointXYZ pmax(
									share_data->object_center_world(0) + share_data->predicted_size,
									share_data->object_center_world(1) + share_data->predicted_size,
									share_data->object_center_world(2) + share_data->predicted_size
								);
								roi_min = ToKey(pmin, inflation_voxel_size);
								roi_max = ToKey(pmax, inflation_voxel_size);
							}
							// 当前点云体素集合 C
							std::unordered_set<VoxelKey, VoxelKeyHash> C;
							VoxelizeCloud(cloud_final_xyz, inflation_voxel_size, C, use_roi, roi_min, roi_max);
							// 历史参考点云体素集合 H
							std::unordered_set<VoxelKey, VoxelKeyHash> H;
							VoxelizeCloud(cloud_nricp, inflation_voxel_size, H, use_roi, roi_min, roi_max);
							// FPS采样点集合（H\C）
							std::unordered_set<VoxelKey, VoxelKeyHash> FPSKeys;
							SetDifference(H, C, FPSKeys);
							if (share_data->add_current_inflation) {
								// 膨胀
								std::unordered_set<VoxelKey, VoxelKeyHash> C_dil, H_dil;
								DilateSet(C, off_near, C_dil, use_roi, roi_min, roi_max);
								DilateSet(H, off_far, H_dil, use_roi, roi_min, roi_max);
								// 集合差：新增候选 = C_dil \ H_dil
								std::unordered_set<VoxelKey, VoxelKeyHash> AddedKeys;
								SetDifference(C_dil, H_dil, AddedKeys);
								// added_points 只包含膨胀出来的“壳”
								for (const auto& k : C) {
									auto it = AddedKeys.find(k);
									if (it != AddedKeys.end()) AddedKeys.erase(it);
								}
								// FPS采样点集合更新：加入新增候选
								FPSKeys.insert(AddedKeys.begin(), AddedKeys.end());
								// Added key -> 点（体素中心）
								pcl::PointCloud<pcl::PointXYZ>::Ptr added_points(new pcl::PointCloud<pcl::PointXYZ>);
								for (const auto& k : AddedKeys) {
									added_points->points.push_back(KeyToCenter(k, inflation_voxel_size));
								}
								added_points->width = (uint32_t)added_points->points.size();
								added_points->height = 1;
								if (!added_points->points.empty()) {
									std::cout << "add " << added_points->points.size() << " voxel-center points to NRICP reference cloud." << std::endl;
									// 加入参考点云：加新增体素点
									*cloud_nricp += *added_points; //ADD dilation shell (AddedKeys)
									pcl::io::savePCDFileBinary(share_data->save_path + "/inflation.pcd", *added_points);
								}
								else {
									std::cout << "no points added (voxelset)." << std::endl;
								}
							}
							//集合覆盖参考点云是H+C+(dilation if inflation enabled)
							*cloud_nricp += *cloud_final_xyz; //ADD C
							pcl::io::savePCDFileBinary(share_data->save_path + "/current_inflation_nricp.pcd", *cloud_nricp);
							//FPS参考点云是H\C+(dilation if inflation enabled)
							pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_fps(new pcl::PointCloud<pcl::PointXYZ>);
							for (const auto& k : FPSKeys) {
								cloud_fps->points.push_back(KeyToCenter(k, inflation_voxel_size));
							}
							cloud_fps->width = (uint32_t)cloud_fps->points.size();
							cloud_fps->height = 1;
							if (!cloud_fps->points.empty()) {
								std::cout << "FPS reference cloud has " << cloud_fps->points.size() << " points." << std::endl;
								pcl::io::savePCDFileBinary(share_data->save_path + "/fps_reference.pcd", *cloud_fps);
							}
							else {
								std::cout << "no points for FPS (voxelset)." << std::endl;
							}

							//FPS downsample
							pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ds = voxelDownsample(cloud_fps, inflation_voxel_size);
							std::cout << "Downsampled plant points: " << cloud_ds->size() << "\n";
							std::mt19937 rng(share_data->random_seed);
							std::vector<Eigen::Vector3d> targets = farthestPointSampling(cloud_ds, share_data->num_targets, rng);
							std::cout << "FPS targets: " << targets.size() << "\n";
							// round-robin assign candidate -> lookat targets
							ViewAssignResult assign = assignViewsRoundRobin(share_data->dynamic_candidate_views, targets, share_data->num_rounds, rng);
							// assign.lookat[k] 就是 used candidate 对应的 look-at
							// assign.used_idx[k] 对应 views[used_idx] 是相机位置
							//for (size_t i = 0; i < assign.used_idx.size(); ++i) {
							//	std::cout << "dynamic view " << assign.used_idx[i] << " assigned to lookat " << assign.lookat[i].transpose() << " (target id " << assign.assigned_target_idx[i] << ")\n";
							//}
							//save and update the assigned views and targets
							const size_t candidateN = share_data->dynamic_candidate_views.size();
							const size_t passiveN = share_data->passive_init_views.size();
							assert(now_view_space->views.size() == passiveN + candidateN);
							std::vector<Eigen::Vector3d> lookat_all(candidateN, Eigen::Vector3d(0, 0, 0));
							for (size_t i = 0; i < assign.used_idx.size(); ++i) {
								size_t c = (size_t)assign.used_idx[i];
								if (c < candidateN) lookat_all[c] = assign.lookat[i];
							}
							std::ofstream fout_assign(share_data->save_path + "/nricp_assign.txt");
							for (size_t c = 0; c < candidateN; ++c) {
								const auto& la = lookat_all[c];
								fout_assign << la.x() << "\t" << la.y() << "\t" << la.z() << "\n";
								now_view_space->views[passiveN + c].look_at = la;
							}
							//check if the look at points are saved to file correctly
							share_data->access_directory(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series));
							ifstream fin_check_look_at(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt");
							if (!fin_check_look_at.is_open()) {
								//copy the look at file
								ofstream fout_look_at(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt");
								for (size_t k = passiveN; k < passiveN + candidateN; ++k){
									fout_look_at << now_view_space->views[k].look_at(0) << "\t" << now_view_space->views[k].look_at(1) << "\t" << now_view_space->views[k].look_at(2) << endl;
								}
								cout << "Look at points saved to " << share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt" << endl;
							}
							else {
								cout << "Look at points already exist in " << share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt" << endl;
								//read from file to make sure we use the same look at points 
								ifstream fin_look_at(share_data->environment_path + "/gap_" + to_string(share_data->gap_between_series) + "/" + share_data->name_of_pcd + "_candidate_look_ats_" + share_data->look_at_group_str + ".txt");
								for (size_t k = passiveN; k < passiveN + candidateN; ++k) {
									double x, y, z;
									fin_look_at >> x >> y >> z;
									now_view_space->views[k].look_at = Eigen::Vector3d(x, y, z);
									if ((now_view_space->views[k].look_at - lookat_all[k - passiveN]).norm() > 1e-6) {
										cout << "Warning: look at point from file is different from assigned result for view " << k << ". Using the one from file." << endl;
									}
								}
								cout << "Look at points read from file and updated to view space." << endl;
							}

							if (share_data->debug || share_data->show) { //显示分配结果
								auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("simple viewer"));
								viewer->setBackgroundColor(255, 255, 255);
								//viewer->addCoordinateSystem(0.2);
								viewer->initCameraParameters();
								// 0) 当前的地图点云
								viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_final, "cloud_final");
								viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, "cloud_final");
								// 1) plant obb 红色线框（不是点）
								AddObbWireframe(viewer, share_data->plant_obb, "plant_obb", 1.0, 60.0 / 255.0, 60.0 / 255.0, 1);
								// 1) 原始 downsample 点云：magenta色，小点
								pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col1(cloud_ds, 255, 0, 255);
								viewer->addPointCloud<pcl::PointXYZ>(cloud_ds, col1, "cloud_ds");
								viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_ds");
								// 2) targets / lookat：红色，大点
								auto pc_targets = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
								for (size_t k = 0; k < assign.lookat.size(); ++k) {
									const Eigen::Vector3d& p = assign.lookat[k];   // 你也可以换成 targets[i]
									pc_targets->push_back(pcl::PointXYZ((float)p.x(), (float)p.y(), (float)p.z()));
								}
								pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col2(pc_targets, 255, 60, 60);
								viewer->addPointCloud<pcl::PointXYZ>(pc_targets, col2, "targets");
								viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "targets");
								// 3) views：蓝色，中点
								auto pc_views = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
								for (size_t k = 0; k < assign.used_idx.size(); ++k) {
									int vidx = assign.used_idx[k];
									Eigen::Vector3d cam = share_data->dynamic_candidate_views[vidx];
									pc_views->push_back(pcl::PointXYZ((float)cam.x(), (float)cam.y(), (float)cam.z()));
								}
								pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col3(pc_views, 80, 160, 255);
								viewer->addPointCloud<pcl::PointXYZ>(pc_views, col3, "views");
								viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "views");
								cout << "Number of assigned views: " << assign.used_idx.size() << endl;
								// 4) rays：每条匹配线段上采样若干点，绿色，小点，形成虚线箭头的效果
								auto pc_rays = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
								int N = 30;                 // 每条匹配线段采样点数（越大越密）
								for (size_t k = 0; k < assign.used_idx.size(); ++k) {
									int vidx = assign.used_idx[k];
									Eigen::Vector3d cam = share_data->dynamic_candidate_views[vidx];
									Eigen::Vector3d tgt = assign.lookat[k];
									Eigen::Vector3d d = tgt - cam;
									double len = d.norm();
									if (len < 1e-9) continue;
									for (int i = 0; i <= 30; ++i) {
										double t = double(i) / double(N);         // [0,1]
										Eigen::Vector3d p = cam + t * d;
										pc_rays->push_back(pcl::PointXYZ((float)p.x(), (float)p.y(), (float)p.z()));
									}
								}
								cout << "Number of ray points: " << pc_rays->points.size() << endl;
								pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col4(pc_rays, 60, 255, 120);
								viewer->addPointCloud<pcl::PointXYZ>(pc_rays, col4, "rays");
								viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "rays");
								while (!viewer->wasStopped()) {
									viewer->spinOnce(10);
									std::this_thread::sleep_for(std::chrono::milliseconds(10));
								}
							}

							//!!!Debug用，只生成look at candidate就结束!!!
							if (share_data->debug) {
								cout << "Debug mode: only generate look at candidates and then exit." << endl;
								status = Over;
								break;
							}

							//transfer cloud_nricp<PointXYZ> to cloud_nricp<PointXYZRGB>
							pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_nricp_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
							pcl::copyPointCloud(*cloud_nricp, *cloud_nricp_rgb);

							//create a temporary share_data for rendering
							cout << "starting perception for set covering." << endl;
							double nricp_perception_start_time = clock();
							Share_Data* share_data_temp = new Share_Data("../DefaultConfiguration.yaml", share_data->name_of_pcd, share_data->rotate_state, share_data->first_view_id, share_data->method_of_IG);
							share_data_temp->use_saved_cloud = false; //不使用保存的点云
							share_data_temp->has_table = false; //set covering时无需使用桌面，速度更快
							share_data_temp->cloud_pcd = cloud_nricp_rgb;
							share_data_temp->ground_truth_resolution = share_data->ground_truth_resolution * share_data->history_resolution_factor;
							delete share_data_temp->ground_truth_model;
							share_data_temp->ground_truth_model = new octomap::ColorOcTree(share_data_temp->ground_truth_resolution);
							//create a temporary nbv_planer
							NBV_Planner* nbv_plan_temp = new NBV_Planner(share_data_temp);
							//get forbidden views (who is dynamic_candidate and its look_at == (0,0,0))
							set<int> forbidden_views;
							for (int i = 0; i < now_view_space->views.size(); i++) {
								if (i < share_data->passive_init_views.size()) continue; //被动视点不考虑
								if ((now_view_space->views[i].look_at - Eigen::Vector3d(0, 0, 0)).norm() < 1e-6) {
									forbidden_views.insert(i);
									//cout << "forbidden view " << i << " added." << endl;
								}
							}
							//获取全部点云
							for (int i = 0; i < nbv_plan_temp->now_view_space->views.size(); i++) {
								if (forbidden_views.find(i) == forbidden_views.end()) { //正常视点，进行感知
									//cout << "Percepting the " << i << "th view." << endl;
									nbv_plan_temp->percept->precept(&nbv_plan_temp->now_view_space->views[i]);
								}
								else { //如果是禁止视点，则手动处理，无需感知，push一个占位点云进去
									pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
									//pcl::PointCloud<pcl::PointXYZRGB>::Ptr no_table(new pcl::PointCloud<pcl::PointXYZRGB>);
									pcl::PointXYZRGB s;
									s.x = s.y = s.z = 0.0f;
									s.r = s.g = s.b = 0;
									cloud->points.resize(1);
									cloud->points[0] = s;
									cloud->width = 1;
									cloud->height = 1;
									cloud->is_dense = false;
									//no_table->points.resize(1);
									//no_table->points[0] = s;
									//no_table->width = 1;
									//no_table->height = 1;
									//no_table->is_dense = false;
									share_data_temp->clouds.push_back(cloud);
									//share_data_temp->clouds_notable.push_back(no_table);
									*share_data_temp->cloud_final += *cloud;
									share_data_temp->vaild_clouds++;
								}
							}
							//计算set covering
							set<int> sc_chosen_views;
							for (int i = 0; i < now_view_space->views.size(); i++) {
								if (now_view_space->views[i].vis)	sc_chosen_views.insert(i);
							}
							unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* all_voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
							vector<unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>*> voxels;
							for (int i = 0; i < nbv_plan_temp->now_view_space->views.size(); i++) {
								unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
								for (int j = 0; j < share_data_temp->clouds[i]->points.size(); j++) {
									octomap::OcTreeKey key;
									if (!share_data_temp->ground_truth_model->coordToKeyChecked(share_data_temp->clouds[i]->points[j].x, share_data_temp->clouds[i]->points[j].y, share_data_temp->clouds[i]->points[j].z, key)) {
										cout << "Warning: point " << share_data_temp->clouds[i]->points[j].x << " " << share_data_temp->clouds[i]->points[j].y << " " << share_data_temp->clouds[i]->points[j].z << " is out of octomap bounds." << endl;
										continue;
									}
									if (voxel->find(key) == voxel->end()) {
										(*voxel)[key] = 1;
										//由于有重复，所以需要判断是否已经存在
										if (all_voxel->find(key) != all_voxel->end()) {
											(*all_voxel)[key]++;
										}
										else {
											(*all_voxel)[key] = 1;
										}
									}
								}
								voxels.push_back(voxel);
							}
							int confidence_count = share_data->history_confidence_count;
							vector<unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>*> voxels_confident;
							for (int i = 0; i < nbv_plan_temp->now_view_space->views.size(); i++) {
								unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_confident = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
								for (auto it = voxels[i]->begin(); it != voxels[i]->end(); it++) {
									octomap::OcTreeKey key = it->first;
									if ((*all_voxel)[it->first] > confidence_count) {
										//cout << "key " << key[0] << " " << key[1] << " " << key[2] << " " << (*all_voxel)[it->first] << endl;
										(*voxel_confident)[key] = 1;
									}
								}
								voxels_confident.push_back(voxel_confident);
							}
							double nricp_perception_end_time = clock();
							double nricp_perception_time = (nricp_perception_end_time - nricp_perception_start_time) / CLOCKS_PER_SEC;
							double nricp_covering_start_time = clock();
							views_voxels_LM* SCOP_solver = new views_voxels_LM(share_data_temp, nbv_plan_temp->now_view_space, sc_chosen_views, voxels_confident, forbidden_views);
							SCOP_solver->solve();
							view_set_label = SCOP_solver->get_view_id_set();
							//保险一点去掉重复的视点
							set<int> vis_view_ids;
							for (int i = 0; i < now_view_space->views.size(); i++) {
								if (now_view_space->views[i].vis)	vis_view_ids.insert(i);
							}
							for (auto it = view_set_label.begin(); it != view_set_label.end(); ) {
								if (vis_view_ids.count((*it))) {
									it = view_set_label.erase(it);
								}
								else {
									it++;
								}
							}
							//记录时间
							double nricp_covering_end_time = clock();
							double nricp_covering_time = (nricp_covering_end_time - nricp_covering_start_time) / CLOCKS_PER_SEC;
							ofstream fout_time(share_data->save_path + "/nricp_covering_time.txt");
							fout_time << nricp_covering_time << "\t" << nricp_perception_time << endl;
							fout_time.close();
							cout << "NRICP perception time is " << nricp_perception_time << " seconds." << endl;
							cout << "NRICP covering time is " << nricp_covering_time << " seconds." << endl;
							//释放内存
							for (int i = 0; i < voxels.size(); i++)
								delete voxels[i];
							voxels.clear();
							voxels.shrink_to_fit();
							for (int i = 0; i < voxels_confident.size(); i++)
								delete voxels_confident[i];
							voxels_confident.clear();
							voxels_confident.shrink_to_fit();
							delete SCOP_solver;
							delete all_voxel;
							delete nbv_plan_temp;
							delete share_data_temp;
						}
						else {
							//读取MASCVP结果
							share_data->access_directory(share_data->sc_net_path + "/log");
							ifstream ftest;
							do {
								//ftest.open(share_data->sc_net_path + "/log/ready.txt"); //串行版本
								ftest.open(share_data->sc_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_ready.txt"); //并行版本
							} while (!ftest.is_open());
							ftest.close();
							ifstream fin(share_data->sc_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + ".txt");
							int rest_view_id;
							while (fin >> rest_view_id) {
								view_set_label.push_back(rest_view_id);
							}
							//若有,则删除已访问视点
							set<int> vis_view_ids;
							for (int i = 0; i < now_view_space->views.size(); i++) {
								if (now_view_space->views[i].vis)	vis_view_ids.insert(i);
							}
							for (auto it = view_set_label.begin(); it != view_set_label.end(); ) {
								if (vis_view_ids.count((*it))) {
									it = view_set_label.erase(it);
								}
								else {
									it++;
								}
							}
						}
						//保存一下最终视点个数
						ofstream fout_all_needed_views(share_data->save_path + "/all_needed_views.txt");
						fout_all_needed_views << view_set_label.size() + 1 + share_data->num_of_nbvs_combined << endl;
						cout << "All_needed_views is " << view_set_label.size() + 1 + share_data->num_of_nbvs_combined << endl;
						//如果没有视点需要，这就直接退出
						if (view_set_label.size() == 0) {
							//更新标志文件
							this_thread::sleep_for(chrono::seconds(1));
							int removed = remove((share_data->sc_net_path + "/log/ready.txt").c_str());
							if (removed != 0) cout << "cannot remove ready.txt." << endl;
							//系统退出
							share_data->over = true;
							status = WaitMoving;
							break;
						}

						//规划全局路径
						double global_path_start_time = clock();
						vector<int> view_set_label_group1, view_set_label_group2;
						vector<int> view_id_path_group1, view_id_path_group2;
						//根据当前视点Z高度分上下两层，确保G1是当前视点层
						const double z_split = share_data->view_obb.c(2);
						if (now_best_view->init_pos(2) >= z_split) {
							//初始视点在上层，上层是G1
							for (auto& view_id : view_set_label) {
								if (now_view_space->views[view_id].init_pos(2) >= z_split) {
									view_set_label_group1.push_back(view_id);
								}
								else {
									view_set_label_group2.push_back(view_id);
								}
							}
						}
						else {
							//初始视点在下层，下层是G1
							for (auto& view_id : view_set_label) {
								if (now_view_space->views[view_id].init_pos(2) < z_split) {
									view_set_label_group1.push_back(view_id);
								}
								else {
									view_set_label_group2.push_back(view_id);
								}
							}
						}
						//先规划组1路径，起点是当前视点
						view_set_label_group1.insert(view_set_label_group1.begin(), now_best_view->id); //把当前视点放到组1开头，确保路径从当前视点开始
						if (view_set_label_group1.size() >= 2) { //2个以上视点才规划
							int start_view_id = view_set_label_group1.front();
							//保证层切换和运动顺溜，强制终点（行最远 + z接近分层平面）
							const double alpha = 0.5; //距离和Z奖励的权重，可以调整
							const Eigen::Vector3d p0 = now_view_space->views[start_view_id].init_pos;
							double best_score = -1e100;
							int end_view_id = -1;
							for (int i = 1; i < view_set_label_group1.size(); ++i) { // 从1开始，跳过起点
								int vid = view_set_label_group1[i];
								const Eigen::Vector3d p = now_view_space->views[vid].init_pos;
								double dist_norm = (p.head<2>() - p0.head<2>()).norm() / std::max(share_data->view_obb.extent(0), share_data->view_obb.extent(1));          // ~0..1
								double dz_norm = std::abs(p(2) - z_split) / (share_data->view_obb.extent(2) * 0.5); // ~0..1
								double score = dist_norm - alpha * dz_norm;   // alpha 0.2~0.5: 轻偏好；alpha 1: 强偏好
								if (score > best_score) {
									best_score = score;
									end_view_id = vid;
								}
							}
							//运行全局路径规划，强制终点
							Global_Path_Planner* global_path_planner_group1 = new Global_Path_Planner(share_data, now_view_space->views, view_set_label_group1, start_view_id, end_view_id);
							global_path_planner_group1->solve();
							view_id_path_group1 = global_path_planner_group1->get_path_id_set();
							delete global_path_planner_group1;
						}
						else if (view_set_label_group1.size() == 1) { //1个视点就不动了
							view_id_path_group1 = view_set_label_group1;
						}
						//再规划组2路径，起点是组1终点，由于组1一定至少有个当前视点，所以没问题
						view_set_label_group2.insert(view_set_label_group2.begin(), view_id_path_group1.back()); //把组1终点放到组2开头，确保组2路径从组1终点开始
						if (view_set_label_group2.size() >= 2) {
							int start_view_id = view_set_label_group2.front();
							//保证运动顺溜，强制终点（行最远）
							const Eigen::Vector3d p0 = now_view_space->views[start_view_id].init_pos;
							double best_score = -1e100;
							int end_view_id = -1;
							for (int i = 1; i < view_set_label_group2.size(); ++i) { // 从1开始，跳过起点
								int vid = view_set_label_group2[i];
								const Eigen::Vector3d p = now_view_space->views[vid].init_pos;
								double dist = (p - p0).norm(); // 距离项：越远越好
								double score = dist;
								if (score > best_score) {
									best_score = score;
									end_view_id = vid;
								}
							}
							Global_Path_Planner* global_path_planner_group2 = new Global_Path_Planner(share_data, now_view_space->views, view_set_label_group2, start_view_id, end_view_id);
							global_path_planner_group2->solve();
							view_id_path_group2 = global_path_planner_group2->get_path_id_set();
							delete global_path_planner_group2;
						}
						else if (view_set_label_group2.size() == 1) {
							view_id_path_group2.clear(); //1个视点就空置，因为组1的终点已经是它了，不需要再动了
						}
						//最终路径是两个的拼接，两个出来的结果已经去掉了起点
						vector<int> view_id_path_full = view_id_path_group1;
						for (auto& view_id : view_id_path_group2) {
							view_id_path_full.push_back(view_id);
						}
						//放置到数据区
						share_data->view_label_id = view_id_path_full; 
						//保存时间
						double global_path_end_time = clock();
						double global_path_time = (global_path_end_time - global_path_start_time) / CLOCKS_PER_SEC;
						cout << "global path time is " << global_path_time << " seconds." << endl;
						ofstream fout_global_path_time(share_data->save_path + "/global_path_time.txt");
						fout_global_path_time << global_path_time << endl;
						fout_global_path_time.close();

						//保存到单独的文件夹里，方便查看
						share_data->access_directory(share_data->save_path + "/movement/" + to_string(view_set_label.size() + 1 + share_data->num_of_nbvs_combined));
						double total_distance = share_data->passive_map_cost; //如果有passive阶段，先把passive阶段的代价加上
						ofstream fout_passive(share_data->save_path + "/movement/" + to_string(view_set_label.size() + 1 + share_data->num_of_nbvs_combined) + "/path" + to_string(-1) + ".txt");
						fout_passive << now_best_view->id << '\t' << 0.0 << '\t' << total_distance << endl; //起点
						vector<int> view_id_path_with_start = view_id_path_full;
						view_id_path_with_start.insert(view_id_path_with_start.begin(), now_best_view->id); //把起点放到路径开头
						assert(view_id_path_with_start.size() == static_cast<size_t>(view_set_label.size() + 1 + share_data->num_of_nbvs_combined - share_data->passive_init_views.size()));
						for (int i = 0; i + 1 < view_id_path_with_start.size(); i++) {
							ofstream fout(share_data->save_path + "/movement/" + to_string(view_set_label.size() + 1 + share_data->num_of_nbvs_combined) + "/path" + to_string(i) + ".txt");
							pair<int, double> local_path = get_local_path(now_view_space->views[view_id_path_with_start[i]].init_pos.eval(), now_view_space->views[view_id_path_with_start[i + 1]].init_pos.eval(), share_data->object_center_world.eval(), share_data->predicted_size * sqrt(2));
							if (local_path.first < 0) cout << "local path wrong." << endl;
							total_distance += local_path.second;
							fout << view_id_path_with_start[i + 1] << '\t' << local_path.second << '\t' << total_distance << endl;
						}
					
						//更新当前视点
						delete now_best_view;
						now_best_view = new View(now_view_space->views[share_data->view_label_id[iterations - share_data->num_of_nbvs_combined]]);
						//运动代价：视点id，当前代价，总体代价
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
						//更新标志文件
						this_thread::sleep_for(chrono::seconds(1));
						if (share_data->use_history_model_for_covering) {
							int removed = remove((share_data->nricp_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_ready.txt").c_str()); //并行版本
							if (removed != 0) cout << "cannot remove ready.txt." << endl;
						}
						else {
							//int removed = remove((share_data->sc_net_path + "/log/ready.txt").c_str()); //串行版本
							int removed = remove((share_data->sc_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_ready.txt").c_str()); //并行版本
							if (removed != 0) cout << "cannot remove ready.txt." << endl;
						}
						//clean
						view_set_label.clear();
						view_set_label.shrink_to_fit();
					}
					else {
						if (iterations == share_data->view_label_id.size() + share_data->num_of_nbvs_combined) {
							share_data->over = true;
							status = WaitMoving;
							break;
						}
						delete now_best_view;
						now_best_view = new View(now_view_space->views[share_data->view_label_id[iterations - share_data->num_of_nbvs_combined]]);
						//运动代价：视点id，当前代价，总体代价
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					}
				}
				else {//搜索算法

					if (iterations < 0 + share_data->passive_init_views.size()) {
						//如果是passive按顺序读
						int next_id = now_best_view->id;
						if (share_data->first_view_id == 0) {
							next_id++; //Order
							if (next_id >= share_data->passive_init_views.size()) {
								cout << "error: no next view in Order method. check num_of_nbvs_combined == passive_init_views.size()-1" << endl;
								next_id = 0;
							}
						}
						else if (share_data->first_view_id == share_data->passive_init_views.size() - 1) {
							next_id--; //Reverse Order
							if (next_id < 0) {
								cout << "error: no next view in Order method. check num_of_nbvs_combined == passive_init_views.size()-1" << endl;
								next_id = share_data->passive_init_views.size() - 1;
							}
						}
						else {
							cout << "error: first_view_id is not correct in Order or Reverse Order method." << endl;
							next_id = 0;
						}
						cout << "next view id is " << next_id << endl;
						now_view_space->views[next_id].vis++;
						delete now_best_view;
						now_best_view = new View(now_view_space->views[next_id]);
						//运动代价：视点id，当前代价，总体代价
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					}
					else {
						//如果是动态情况才需要根据infromation选
						for (int i = 0; i < now_view_space->views.size(); i++) {
							if ((now_view_space->views[i].look_at).norm() < 1e-6) { //如果look_at是(0,0,0)，说明这个视点没有被分配到目标，直接设置为不可达，避免后续计算
								now_view_space->views[i].can_move = false;
							}
						}
						//对视点排序
						sort(now_view_space->views.begin(), now_view_space->views.end(), view_utility_compare);
						/*if (share_data->sum_local_information == 0) {
							cout << "randomly choose a view" << endl;
							random_shuffle(now_view_space->views.begin(), now_view_space->views.end());
						}*/
						//informed_viewspace
						if (share_data->show) { //显示BBX与相机位置
							pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Iteration" + to_string(iterations)));
							viewer->setBackgroundColor(0, 0, 0);
							viewer->addCoordinateSystem(0.1);
							viewer->initCameraParameters();
							//test_viewspace
							pcl::PointCloud<pcl::PointXYZRGB>::Ptr test_viewspace(new pcl::PointCloud<pcl::PointXYZRGB>);
							test_viewspace->is_dense = false;
							test_viewspace->points.resize(now_view_space->views.size());
							auto ptr = test_viewspace->points.begin();
							int needed = 0;
							for (int i = 0; i < now_view_space->views.size(); i++) {
								(*ptr).x = now_view_space->views[i].init_pos(0);
								(*ptr).y = now_view_space->views[i].init_pos(1);
								(*ptr).z = now_view_space->views[i].init_pos(2);
								//访问过的点记录为蓝色
								if (now_view_space->views[i].vis) (*ptr).r = 0, (*ptr).g = 0, (*ptr).b = 255;
								//在网络流内的设置为黄色
								else if (now_view_space->views[i].in_coverage[iterations] && i < now_view_space->views.size() / 10) (*ptr).r = 255, (*ptr).g = 255, (*ptr).b = 0;
								//在网络流内的设置为绿色
								else if (now_view_space->views[i].in_coverage[iterations]) (*ptr).r = 255, (*ptr).g = 0, (*ptr).b = 0;
								//前10%的权重的点设置为蓝绿色
								else if (i < now_view_space->views.size() / 10) (*ptr).r = 0, (*ptr).g = 255, (*ptr).b = 255;
								//其余点白色
								else (*ptr).r = 255, (*ptr).g = 255, (*ptr).b = 255;
								ptr++;
								needed++;
							}
							test_viewspace->points.resize(needed);
							viewer->addPointCloud<pcl::PointXYZRGB>(test_viewspace, "test_viewspace");
							bool best_have = false;
							for (int i = 0; i < now_view_space->views.size(); i++) {
								if (now_view_space->views[i].vis) {
									now_view_space->views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
									Eigen::Matrix4d view_pose_world = (share_data->now_camera_pose_world * now_view_space->views[i].pose.inverse()).eval();
									Eigen::Vector4d X(0.03, 0, 0, 1);
									Eigen::Vector4d Y(0, 0.03, 0, 1);
									Eigen::Vector4d Z(0, 0, 0.03, 1);
									Eigen::Vector4d O(0, 0, 0, 1);
									X = view_pose_world * X;
									Y = view_pose_world * Y;
									Z = view_pose_world * Z;
									O = view_pose_world * O;
									viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
									viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
									viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
								}
								else if (!best_have) {
									now_view_space->views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
									Eigen::Matrix4d view_pose_world = (share_data->now_camera_pose_world * now_view_space->views[i].pose.inverse()).eval();
									Eigen::Vector4d X(0.08, 0, 0, 1);
									Eigen::Vector4d Y(0, 0.08, 0, 1);
									Eigen::Vector4d Z(0, 0, 0.08, 1);
									Eigen::Vector4d O(0, 0, 0, 1);
									X = view_pose_world * X;
									Y = view_pose_world * Y;
									Z = view_pose_world * Z;
									O = view_pose_world * O;
									viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
									viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
									viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
									best_have = true;
								}
							}
							viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_final, "cloud_now_itreation");
							while (!viewer->wasStopped())
							{
								viewer->spinOnce(100);
								boost::this_thread::sleep(boost::posix_time::microseconds(100000));
							}
						}
						double max_utility = -1;
						for (int i = 0; i < now_view_space->views.size(); i++) {
							cout << "checking view " << i << endl;
							if (now_view_space->views[i].vis) continue;

							//注意这里加了不可达点的限制
							if (!now_view_space->views[i].can_move) continue;

							delete now_best_view;
							now_best_view = new View(now_view_space->views[i]);
							max_utility = now_best_view->final_utility;
							now_view_space->views[i].vis++;
							//now_view_space->views[i].can_move = true;
							cout << "choose the " << i << "th view." << endl;
							//运动代价：视点id，当前代价，总体代价
							share_data->movement_cost += now_best_view->robot_cost;
							share_data->access_directory(share_data->save_path + "/movement");
							ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
							fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
							break;
						}
						if (max_utility == -1) {
							cout << "Can't move to any viewport.Stop." << endl;
							status = Over;
							break;
						}
					}

					cout << " next best view pos is (" << now_best_view->init_pos(0) << ", " << now_best_view->init_pos(1) << ", " << now_best_view->init_pos(2) << ")" << endl;
					cout << " next best view final_utility is " << now_best_view->final_utility << endl;
				}

				//进入运动模块
				thread next_moving(move_robot, now_best_view, now_view_space, share_data);
				next_moving.detach();
				status = WaitMoving;
			}
			break;
		case WaitMoving:
			//if the method is not (combined) one-shot and random, then use f_voxel to decide whether to stop
			if(!(share_data->Combined_on == true || share_data->method_of_IG == 7 || share_data->method_of_IG == 8)){
				//compute f_voxels
				int f_voxels_num = 0;
				for (octomap::ColorOcTree::leaf_iterator it = share_data->octo_model->begin_leafs(), end = share_data->octo_model->end_leafs(); it != end; ++it) {
					double occupancy = (*it).getOccupancy();
					if (fabs(occupancy - 0.5) < 1e-3) { // unknown
						auto coordinate = it.getCoordinate();
						if (coordinate.x() >= now_view_space->object_center_world(0) - now_view_space->predicted_size && coordinate.x() <= now_view_space->object_center_world(0) + now_view_space->predicted_size
							&& coordinate.y() >= now_view_space->object_center_world(1) - now_view_space->predicted_size && coordinate.y() <= now_view_space->object_center_world(1) + now_view_space->predicted_size
							&& coordinate.z() >= now_view_space->object_center_world(2) - now_view_space->predicted_size && coordinate.z() <= now_view_space->object_center_world(2) + now_view_space->predicted_size)
						{
							// compute the frontier voxels that is unknown and has at least one free and one occupied neighbor
							int free_cnt = 0;
							int occupied_cnt = 0;
							for (int i = -1; i <= 1; i++)
								for (int j = -1; j <= 1; j++)
									for (int k = -1; k <= 1; k++)
									{
										if (i == 0 && j == 0 && k == 0) continue;
										double x = coordinate.x() + i * share_data->octomap_resolution;
										double y = coordinate.y() + j * share_data->octomap_resolution;
										double z = coordinate.z() + k * share_data->octomap_resolution;
										octomap::point3d neighbour(x, y, z);
										octomap::OcTreeKey neighbour_key;  bool neighbour_key_have = share_data->octo_model->coordToKeyChecked(neighbour, neighbour_key);
										if (neighbour_key_have) {
											octomap::ColorOcTreeNode* neighbour_voxel = share_data->octo_model->search(neighbour_key);
											if (neighbour_voxel != NULL) {
												double neighbour_occupancy = neighbour_voxel->getOccupancy();
												free_cnt += neighbour_occupancy < 0.5 ? 1 : 0;
												occupied_cnt += neighbour_occupancy > 0.5 ? 1 : 0;
											}
										}
									}
							//edge
							if (free_cnt >= 1 && occupied_cnt >= 1) {
								f_voxels_num++;
								//cout << "f voxel: " << coordinate.x() << " " << coordinate.y() << " " << coordinate.z() << endl;
							}
						}
					}
				}
				share_data->f_voxels.push_back(f_voxels_num);

				share_data->access_directory(share_data->save_path + "/f_voxels");
				ofstream fout_f_voxels_num(share_data->save_path + "/f_voxels/f_num" + to_string(iterations) + ".txt");
				fout_f_voxels_num << f_voxels_num << endl;
			    // check if the f_voxels_num is stable
				if (share_data->f_stop_iter == -1) {
					if (share_data->f_voxels.size() > 2) {
						bool f_voxels_change = false;
						//三次扫描过程中，连续两个f变化都小于阈值就结束
						if (fabs(share_data->f_voxels[share_data->f_voxels.size() - 1] - share_data->f_voxels[share_data->f_voxels.size() - 2]) >= 32 * 32 * 32 * share_data->f_stop_threshold) {
							f_voxels_change = true;
						}
						if (fabs(share_data->f_voxels[share_data->f_voxels.size() - 2] - share_data->f_voxels[share_data->f_voxels.size() - 3]) >= 32 * 32 * 32 * share_data->f_stop_threshold) {
							f_voxels_change = true;
						}
						if (!f_voxels_change) {
							cout << "two f_voxels change smaller than threshold. Record." << endl;
							share_data->f_stop_iter = iterations;

							ofstream fout_f_stop_views(share_data->save_path + "/f_voxels/f_stop_views.txt");
							fout_f_stop_views << 1 << "\t" << share_data->f_stop_iter + 1 << endl; //1 means f_voxels stop
						}
					}
					if (share_data->over == true && share_data->f_stop_iter == -1) {
						cout << "Max iter reached. Record." << endl;
						share_data->f_stop_iter = iterations;

						ofstream fout_f_stop_views(share_data->save_path + "/f_voxels/f_stop_views.txt");
						fout_f_stop_views << 0 << "\t" << share_data->f_stop_iter + 1 << endl; //0 means over
					}
				}
				if (share_data->f_stop_iter_lenient == -1) {
					if (share_data->f_voxels.size() > 2) {
						bool f_voxels_change = false;
						//三次扫描过程中，连续两个f变化都小于阈值就结束
						if (fabs(share_data->f_voxels[share_data->f_voxels.size() - 1] - share_data->f_voxels[share_data->f_voxels.size() - 2]) >= 32 * 32 * 32 * share_data->f_stop_threshold_lenient) {
							f_voxels_change = true;
						}
						if (fabs(share_data->f_voxels[share_data->f_voxels.size() - 2] - share_data->f_voxels[share_data->f_voxels.size() - 3]) >= 32 * 32 * 32 * share_data->f_stop_threshold_lenient) {
							f_voxels_change = true;
						}
						if (!f_voxels_change) {
							cout << "two f_voxels change smaller than threshold_lenient. Record." << endl;
							share_data->f_stop_iter_lenient = iterations;

							ofstream fout_f_stop_views(share_data->save_path + "/f_voxels/f_lenient_stop_views.txt");
							fout_f_stop_views << 1 << "\t" << share_data->f_stop_iter_lenient + 1 << endl; //1 means f_voxels stop
						}
					}
					if (share_data->over == true && share_data->f_stop_iter_lenient == -1) {
						cout << "Max iter reached. Record." << endl;
						share_data->f_stop_iter_lenient = iterations;

						ofstream fout_f_stop_views(share_data->save_path + "/f_voxels/f_lenient_stop_views.txt");
						fout_f_stop_views << 0 << "\t" << share_data->f_stop_iter_lenient + 1 << endl; //0 means over
					}
				}
			}
			//virtual move
			if (share_data->over) {
				cout << "Progress over.Saving octomap and cloud." << endl;
				status = Over;
				break;
			}
			if (share_data->move_on) {
				iterations++;
				share_data->iterations = iterations;
				share_data->now_view_space_processed = false;
				share_data->now_views_infromation_processed = false;
				share_data->move_on = false;
				status = WaitData;
			}
			break;
		}
		return status;
	}

	string out_status() {
		string status_string;
		switch (status)
		{
		case Over:
			status_string = "Over";
			break;
		case WaitData:
			status_string = "WaitData";
			break;
		case WaitViewSpace:
			status_string = "WaitViewSpace";
			break;
		case WaitInformation:
			status_string = "WaitInformation";
			break;
		case WaitMoving:
			status_string = "WaitMoving";
			break;
		}
		return status_string;
	}
};

void save_cloud_mid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string name, Share_Data* share_data) {
	//保存中间的点云的线程，目前不检查是否保存完毕
	share_data->save_cloud_to_disk(cloud, "/clouds", name);
	cout << name << " saved" << endl;
	//清空点云
	cloud->points.clear();
	cloud->points.shrink_to_fit();
}

void create_view_space(View_Space** now_view_space, View* now_best_view, Share_Data* share_data, int iterations) {
	//计算关键帧相机位姿
	now_best_view->get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
	//不再使用上次的相机位姿更新当前相机位姿结合type_of_pose = 0，目前使用全局type_of_pose = 1/2
	//share_data->now_camera_pose_world = (share_data->now_camera_pose_world * now_best_view->pose.inverse()).eval();
	//处理viewspace,如果不需要评估并且是one-shot路径就不更新OctoMap
	if (share_data->evaluate_one_shot == 0 && share_data->method_of_IG == 7 && iterations > 0 + share_data->num_of_nbvs_combined);
	else (*now_view_space)->update(iterations, share_data, share_data->cloud_final, share_data->clouds[iterations], now_best_view->id);
	//保存中间迭代结果
	if(share_data->is_save)	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_mid(new pcl::PointCloud<pcl::PointXYZRGB>);
		*cloud_mid = *share_data->cloud_final;
		thread save_mid(save_cloud_mid, cloud_mid, "pointcloud" + to_string(iterations), share_data);
		save_mid.detach();
	}
	//更新标志位
	share_data->now_view_space_processed = true;
}

void create_views_information(Views_Information** now_views_infromation, View* now_best_view, View_Space* now_view_space, Share_Data* share_data, int iterations) {
	if (share_data->method_of_IG == 8) { //PriorTemporalRandom
		;
	}
	else if (share_data->method_of_IG == 11) { //PriorTemporalCovering
		;
	}
	else if (share_data->method_of_IG == 12) { //PriorPassiveRandom
		;
	}
	else if (share_data->method_of_IG == 13) { //PriorBBXRandom
		;
	}
	else if (share_data->method_of_IG == 14) { //SamplingNBV
		;
	}
	else if (share_data->method_of_IG == 6) { //NBV-NET
		//octotree
		share_data->access_directory(share_data->nbv_net_path + "/data");
		ofstream fout(share_data->nbv_net_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) +'_'+ to_string(iterations) + ".txt");
		for (int i = 0; i < 32; i++)
			for (int j = 0; j < 32; j++)
				for (int k = 0; k < 32; k++)
				{
					double x = share_data->object_center_world(0) - share_data->predicted_size + share_data->octomap_resolution * i;
					double y = share_data->object_center_world(1) - share_data->predicted_size + share_data->octomap_resolution * j;
					double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + share_data->octomap_resolution * k;
					auto node = share_data->octo_model->search(x, y, z);
					if (node == NULL) cout << "what?" << endl;
					fout << node->getOccupancy() << '\n';
				}
		fout.close();
	}
	else if (share_data->method_of_IG == 9) { //PCNBV
		share_data->access_directory(share_data->pcnbv_path + "/data");
		ofstream fout_pointcloud(share_data->pcnbv_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_pc" + to_string(iterations) + ".txt");
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);
		//随机降采样点云
		pcl::RandomSample<pcl::PointXYZRGB> ran;
		ran.setInputCloud(share_data->cloud_final);
		ran.setSample(1024); //设置下采样点云的点数
		ran.filter(*cloud_out);
		for (int i = 0; i < cloud_out->points.size(); i++){
			fout_pointcloud << cloud_out->points[i].x << ' '
				<< cloud_out->points[i].y << ' '
				<< cloud_out->points[i].z << '\n';
		}
		ofstream fout_viewstate(share_data->pcnbv_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_vs" + to_string(iterations) + ".txt");
		for (int i = 0; i < share_data->length_of_viewstate; i++) {
			if (now_view_space->views[i].vis)	fout_viewstate << 1 << '\n';
			else  fout_viewstate << 0 << '\n';
		}
		fout_pointcloud.close();
		fout_viewstate.close();
		cloud_out->points.clear();
		cloud_out->points.shrink_to_fit();
	}
	else if (share_data->method_of_IG == 7) { //SCVP，MA-SCVP，NRICP pipeline
		//如果NBV结束了，或者是直接覆盖
		if (iterations == 0 + share_data->num_of_nbvs_combined) {
			//如果使用历史模型则调用NRICP
			if (share_data->use_history_model_for_covering) {
				share_data->access_directory(share_data->nricp_path + "/data");
				//根据当前水密GT分辨率降采样
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out = voxelDownsample(share_data->cloud_final, share_data->ground_truth_resolution);
				//计算normal
				pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_out_normals = EstimateNormals_KdTree(cloud_out, share_data->view_obb.c);
				pcl::io::savePLYFile<pcl::PointXYZRGBNormal>(share_data->nricp_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_tgt.ply", *cloud_out_normals);
				//复制历史模型，从映射中读取历史模型名称
				string room_str = share_data->name_of_pcd.substr(0, share_data->name_of_pcd.find('_'));
				string current_timestamp = share_data->name_of_pcd.substr(share_data->name_of_pcd.find('_') + 1);
				string history_model_name = room_str + "_" + share_data->previous_reconstruction_mapping[current_timestamp];
				cout << "current_timestamp name is " << current_timestamp << endl;
				cout << "history model name is " << history_model_name << endl;
				//读取历史模型，并根据当前旋转情况旋转
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr history_model(new pcl::PointCloud<pcl::PointXYZRGB>);
				pcl::io::loadPCDFile<pcl::PointXYZRGB>(share_data->pcd_file_path + history_model_name + ".pcd", *history_model);
				//旋转历史模型，注意既然GT旋转了，这里读进来就也要旋转，否则容易导致NRICP初值有问题不收敛
				Eigen::Matrix3d rotation;
				rotation = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
					Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
					Eigen::AngleAxisd(45 * share_data->rotate_state * acos(-1.0) / 180.0, Eigen::Vector3d::UnitZ());
				Eigen::Matrix4d T_pose(Eigen::Matrix4d::Identity(4, 4));
				T_pose(0, 0) = rotation(0, 0); T_pose(0, 1) = rotation(0, 1); T_pose(0, 2) = rotation(0, 2); T_pose(0, 3) = 0;
				T_pose(1, 0) = rotation(1, 0); T_pose(1, 1) = rotation(1, 1); T_pose(1, 2) = rotation(1, 2); T_pose(1, 3) = 0;
				T_pose(2, 0) = rotation(2, 0); T_pose(2, 1) = rotation(2, 1); T_pose(2, 2) = rotation(2, 2); T_pose(2, 3) = 0;
				T_pose(3, 0) = 0;			   T_pose(3, 1) = 0;			  T_pose(3, 2) = 0;			     T_pose(3, 3) = 1;
				pcl::transformPointCloud(*history_model, *history_model, T_pose);
				//根据当前水密GT分辨率降采样
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr history_model_out = voxelDownsample(history_model, share_data->ground_truth_resolution);
				//计算normal
				pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr history_model_out_normals = EstimateNormals_KdTree(history_model_out, share_data->view_obb.c);
				pcl::io::savePLYFile<pcl::PointXYZRGBNormal>(share_data->nricp_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_src.ply", *history_model_out_normals);
				//写config文件
				ofstream fout_config(share_data->nricp_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_config.yaml");
				fout_config << "src_pc: \"" << "./data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_src.ply\"\n"
					<< "tgt_pc: \"" << "./data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_tgt.ply\"\n"
					<< "visualize_nricp: False\n" //是否可视化NRICP结果
					<< "nricp_result_path: \"" << "./log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_nricp.ply\"\n"
					<< "cost_time_path: \"" << "./log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_nricp_time.txt\"\n";
				fout_config.close();
			}
			//如果调用SCVP/MA-SCVP进行预测
			else { 
				//octotree
				share_data->access_directory(share_data->sc_net_path + "/data");
				//ofstream fout(share_data->sc_net_path + "/data/" + share_data->name_of_pcd + ".txt");
				ofstream fout(share_data->sc_net_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_voxel.txt");
				for (int i = 0; i < 32; i++)
					for (int j = 0; j < 32; j++)
						for (int k = 0; k < 32; k++)
						{
							double x = share_data->object_center_world(0) - share_data->predicted_size + share_data->octomap_resolution * i;
							double y = share_data->object_center_world(1) - share_data->predicted_size + share_data->octomap_resolution * j;
							double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + share_data->octomap_resolution * k;
							auto node = share_data->octo_model->search(x, y, z);
							if (node == NULL) cout << "what?" << endl;
							fout << node->getOccupancy() << '\n';
						}
				//view state
				ofstream fout_viewstate(share_data->sc_net_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_vs.txt");
				if (share_data->MA_SCVP_on) { //如果是MA-SCVP才输出视点向量
					for (int i = 0; i < share_data->length_of_viewstate; i++) {
						if (now_view_space->views[i].vis)	fout_viewstate << 1 << '\n';
						else  fout_viewstate << 0 << '\n';
					}
				}
				fout.close();
				fout_viewstate.close();
			}
		}
	}
	else { //搜索方法
		int num_of_cover = 1;
		int num_of_voxel = 0;
		//处理views_informaiton
		if (iterations == 0) (*now_views_infromation) = new Views_Information(share_data, now_view_space->voxel_information, now_view_space, iterations);
		else (*now_views_infromation)->update(share_data, now_view_space, iterations);
		if (share_data->method_of_IG == GMC) {
			//处理GMC，获取全局优化函数
			views_voxels_GMC* max_cover_solver = new views_voxels_GMC(share_data->num_of_max_flow_node, now_view_space, *now_views_infromation, now_view_space->voxel_information, share_data);
			max_cover_solver->solve();
			vector<pair<int, int>> coverage_view_id_voxelnum_set = max_cover_solver->get_view_id_voxelnum_set();
			num_of_cover = coverage_view_id_voxelnum_set.size();
			for (int i = 0; i < now_view_space->views.size(); i++)
				now_view_space->views[i].in_cover = 0;
			for (int i = 0; i < coverage_view_id_voxelnum_set.size(); i++) {
				now_view_space->views[coverage_view_id_voxelnum_set[i].first].in_cover = coverage_view_id_voxelnum_set[i].second;
				num_of_voxel += coverage_view_id_voxelnum_set[i].second;
			}
			delete max_cover_solver;
			coverage_view_id_voxelnum_set.clear();
			coverage_view_id_voxelnum_set.shrink_to_fit();
			//保证分母不为0，无实际意义
			num_of_voxel = max(num_of_voxel, 1);
		}
		else if (share_data->method_of_IG == MCMF) {
			//处理网络流，获取全局优化函数
			views_voxels_MF* set_cover_solver = new views_voxels_MF(share_data->num_of_max_flow_node, now_view_space, *now_views_infromation, now_view_space->voxel_information, share_data);
			set_cover_solver->solve();
			vector<int> coverage_view_id_set = set_cover_solver->get_view_id_set();
			for (int i = 0; i < coverage_view_id_set.size(); i++)
				now_view_space->views[coverage_view_id_set[i]].in_coverage[iterations] = 1;
			delete set_cover_solver;
			coverage_view_id_set.clear();
			coverage_view_id_set.shrink_to_fit();
		}
		//综合计算局部贪心与全局优化，产生视点信息熵
		share_data->sum_local_information = 0;
		share_data->sum_global_information = 0;
		share_data->sum_robot_cost = 0;
		for (int i = 0; i < now_view_space->views.size(); i++) {
			share_data->sum_local_information += now_view_space->views[i].information_gain;
			share_data->sum_global_information += now_view_space->views[i].get_global_information();
			share_data->sum_robot_cost += now_view_space->views[i].robot_cost;
		}
		//保证分母不为0，无实际意义
		if (share_data->sum_local_information == 0) share_data->sum_local_information = 1.0;
		if (share_data->sum_global_information == 0) share_data->sum_global_information = 1.0;
		for (int i = 0; i < now_view_space->views.size(); i++) {
			if (share_data->move_cost_on == false) {
				if (share_data->method_of_IG == MCMF) now_view_space->views[i].final_utility = (1 - share_data->cost_weight) * now_view_space->views[i].information_gain / share_data->sum_local_information + share_data->cost_weight * now_view_space->views[i].get_global_information() / share_data->sum_global_information;
				else if (share_data->method_of_IG == Kr) now_view_space->views[i].final_utility = now_view_space->views[i].information_gain / now_view_space->views[i].voxel_num;
				else if (share_data->method_of_IG == GMC) now_view_space->views[i].final_utility = (1 - share_data->cost_weight) * now_view_space->views[i].information_gain / share_data->sum_local_information + share_data->cost_weight * now_view_space->views[i].in_cover / num_of_voxel;
				else now_view_space->views[i].final_utility = now_view_space->views[i].information_gain;
			}
			else {
				if (share_data->method_of_IG == MCMF) now_view_space->views[i].final_utility = (1 - share_data->move_weight)* ((1 - share_data->cost_weight) * now_view_space->views[i].information_gain / share_data->sum_local_information + share_data->cost_weight * now_view_space->views[i].get_global_information() / share_data->sum_global_information) + share_data->move_weight * (share_data->robot_cost_negtive == true ? -1 : 1) * now_view_space->views[i].robot_cost / share_data->sum_robot_cost;
				else if (share_data->method_of_IG == Kr) now_view_space->views[i].final_utility = (1 - share_data->move_weight) * now_view_space->views[i].information_gain / now_view_space->views[i].voxel_num + share_data->move_weight * (share_data->robot_cost_negtive == true ? -1 : 1) * now_view_space->views[i].robot_cost / share_data->sum_robot_cost;
				else if (share_data->method_of_IG == GMC) now_view_space->views[i].final_utility = (1 - share_data->move_weight) * (1 - share_data->cost_weight) * now_view_space->views[i].information_gain / share_data->sum_local_information + share_data->cost_weight * now_view_space->views[i].in_cover / num_of_voxel + share_data->move_weight * (share_data->robot_cost_negtive == true ? -1 : 1) * now_view_space->views[i].robot_cost / share_data->sum_robot_cost;
				else now_view_space->views[i].final_utility = (1 - share_data->move_weight) * now_view_space->views[i].information_gain + share_data->move_weight * (share_data->robot_cost_negtive == true ? -1 : 1) * now_view_space->views[i].robot_cost / share_data->sum_robot_cost;
			}
		}
	}
	//更新标志位
	share_data->now_views_infromation_processed = true;
}

void move_robot(View* now_best_view, View_Space* now_view_space, Share_Data* share_data) {
	if (share_data->iterations + 1 == share_data->num_of_nbvs_combined) { //Combined+MASCVP切换
		share_data->method_of_IG = SCVP;
		sort(now_view_space->views.begin(), now_view_space->views.end(), view_id_compare);
	}
	if (share_data->num_of_max_iteration > 0 && share_data->iterations + 1 >= share_data->num_of_max_iteration) share_data->over = true;
	if (!share_data->move_wait) share_data->move_on = true;
}

void show_cloud(pcl::visualization::PCLVisualizer::Ptr viewer) {
	//pcl显示点云
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

//main.cpp
atomic<bool> stop = false;		//控制程序结束
Share_Data* share_data;			//共享数据区指针
NBV_Planner* nbv_plan;

void get_command()
{	//从控制台读取指令字符串
	string cmd;
	while (!stop && !share_data->over)
	{
		cout << "Input command 1.stop 2.over 3.next_itreation :" << endl;
		cin >> cmd;
		if (cmd == "1") stop = true;
		else if (cmd == "2") share_data->over = true;
		else if (cmd == "3") share_data->move_on = true;
		else cout << "Wrong command.Retry :" << endl;
	}
	cout << "get_command over." << endl;
}

void get_run()
{
	//NBV规划期初始化
	nbv_plan = new NBV_Planner(share_data);
	//主控循环
	string status="";
	//实时读取与规划
	while (!stop && nbv_plan->plan()) {
		//如果状态有变化就输出
		if (status != nbv_plan->out_status()) {
			status = nbv_plan->out_status();
			cout << "NBV_Planner's status is " << status << endl;
		}
	}
	delete nbv_plan;
}

#define DebugOne 0 //test
#define TestAll 1 //for all objects
#define TestAllRandom 2 //for random seeds
#define GetOracleVisible 3 //for get upper bound (Theory Approximation using Voxelization)
#define GetGTPoints 4 //for cache
#define CheckOracle 5 //for set diff debug, not used
#define GetOracleVisiblePoisson 6 //for get upper bound (shared same view space)
#define TestPoseNoise 7 //for pose noise

int mode = GetGTPoints;

int main()
{
	//Init
	ios::sync_with_stdio(false);
	cout << "input mode:";
	cin >> mode;

	//没有旋转
	vector<int> rotate_states;
	rotate_states.push_back(0);

	//左下0到右上14，左侧开始和右侧开始
	vector<int> first_view_ids;
	first_view_ids.push_back(0);

	int combined_test_on;
	cout << "combined on:";
	cin >> combined_test_on;
	//默认有初始值
	//combined_test_on = 1;

	////scvp，7需要先跑，其他方法读取个数，若后跑就按默认值
	//vector<int> methods;
	//methods.push_back(7);
	//methods.push_back(3);
	//methods.push_back(4);
	//methods.push_back(0);
	//methods.push_back(6);

	int method_id;
	cout << "thread for method id:";
	cin >> method_id;

	int move_test_on;
	//cout << "move test on :";
	//cin >> move_test_on;
	//默认关闭
	move_test_on = 0;

	vector<int> random_seeds;
	random_seeds.push_back(0);
	//random_seeds.push_back(1);
	//random_seeds.push_back(42);
	//random_seeds.push_back(1024);
	//random_seeds.push_back(3407);

	//测试集
	vector<string> names;
	cout << "input models:" << endl;
	string name;
	while (cin >> name) {
		if (name == "-1") break;
		names.push_back(name);
	}

	//选取模式
	if (mode == DebugOne)
	{
		//数据区初始化
		share_data = new Share_Data("../DefaultConfiguration.yaml");
		//控制台读取指令线程
		thread cmd(get_command);
		//NBV系统运行线程
		thread runner(get_run);
		//等待线程结束
		runner.join();
		cmd.join();
		delete share_data;
	}
	else if (mode == TestAll) {
		//测试所有物体、视点、方法
		for (int i = 0; i < names.size(); i++) {
			for (int j = 0; j < rotate_states.size(); j++) {
				for (int k = 0; k < first_view_ids.size(); k++) {
					srand(42);
					//数据区初始化
					share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], rotate_states[j], first_view_ids[k], method_id, move_test_on, combined_test_on);
					//NBV系统运行线程
					get_run();
					delete share_data;
				}
			}
		}
	}
	else if (mode == TestAllRandom) {
		if (method_id != 8 && method_id != 12 && method_id != 13) {
			cout << "Random methods are 8,12,13. Please input correct method id. This will run differnt random seeds." << endl;
			return 0;
		}
		first_view_ids.clear();
		first_view_ids.push_back(0); //随机方法固定初始视点为0，从14倒着走是一样的，且增加了测试量
		//测试所有物体、视点、方法
		for (int i = 0; i < names.size(); i++) {
			for (int j = 0; j < rotate_states.size(); j++) {
				for (int k = 0; k < first_view_ids.size(); k++) {
					for (int s = 0; s < random_seeds.size(); s++) {
						srand(random_seeds[s]);
						//数据区初始化
						share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], rotate_states[j], first_view_ids[k], method_id, move_test_on, combined_test_on, random_seeds[s]);
						//NBV系统运行线程
						get_run();
						delete share_data;
					}
				}
			}
		}
	}
	else if (mode == GetOracleVisible) {
		for (int i = 0; i < names.size(); i++) {
			cout << "Get Oracle visible pointcloud number of model " << names[i] << endl;
			share_data = new Share_Data("../DefaultConfiguration.yaml", names[i]);
			//第一步：使用GT点云生成GT octomap
			octomap::ColorOcTree ground_truth_model(share_data->ground_truth_resolution);
			cout << "use resolution " << share_data->ground_truth_resolution << " to generate GT octomap." << endl;
			std::vector<octomap::OcTreeKey> ground_truth_keys;
			std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> ground_truth_key_set;
			for (auto ptr = share_data->cloud_pcd->points.begin(); ptr != share_data->cloud_pcd->points.end(); ptr++)
			{
				octomap::OcTreeKey key;  bool key_have = ground_truth_model.coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key);
				if (key_have) {
					octomap::ColorOcTreeNode* voxel = ground_truth_model.search(key);
					if (voxel == NULL) {
						ground_truth_model.setNodeValue(key, ground_truth_model.getProbHitLog(), true);
						ground_truth_model.integrateNodeColor(key, (*ptr).r, (*ptr).g, (*ptr).b);
						if (ground_truth_key_set.insert(key).second) {
							ground_truth_keys.push_back(key);
						}
					}
				}
			}
			ground_truth_model.updateInnerOccupancy();
			int full_voxels = 0;
			for (octomap::ColorOcTree::leaf_iterator it = ground_truth_model.begin_leafs(), end = ground_truth_model.end_leafs(); it != end; ++it) {
				full_voxels++;
			}
			cout << "Ground truth octomap generated with " << full_voxels << " full voxels." << endl;
			// resolution
			const double res = share_data->ground_truth_resolution * share_data->view_space_octomap_resolution_factor;
			const OBB& obb = share_data->view_obb;
			std::cout << "Discretize view space with resolution " << res << std::endl;
			// 1) OBB corners -> AABB (world)
			auto cs = obb.corners();
			Eigen::Vector3d bb_min = cs[0];
			Eigen::Vector3d bb_max = cs[0];
			for (int i = 1; i < 8; ++i) {
				bb_min = bb_min.cwiseMin(cs[i]);
				bb_max = bb_max.cwiseMax(cs[i]);
			}
			// 2) build view_obb_model (for coordToKeyChecked / keyToCoord)
			octomap::ColorOcTree view_obb_model(res);
			// 3) AABB -> key range (IMPORTANT: pad a little to avoid boundary miss)
			const double pad = 2.0 * res;  // 壳厚度同量级的 padding，避免角落漏 key
			octomap::OcTreeKey kmin, kmax;
			bool ok_min = view_obb_model.coordToKeyChecked(octomap::point3d(bb_min.x() - pad, bb_min.y() - pad, bb_min.z() - pad), kmin);
			bool ok_max = view_obb_model.coordToKeyChecked(octomap::point3d(bb_max.x() + pad, bb_max.y() + pad, bb_max.z() + pad), kmax);
			if (!ok_min || !ok_max) {
				std::cout << "Warning: view OBB AABB is out of octomap bounds when converting to keys." << std::endl;
				// 你可以选择 continue; 或者 clamp；这里简单 continue
				// continue;
			}
			// 4) shell test (use voxel center p from keyToCoord, no phase issue)
			auto isShell = [&](const Eigen::Vector3d& p)->bool {
				Eigen::Vector3d q = obb.R.transpose() * (p - obb.c);  // local
				Eigen::Vector3d h = obb.half();
				const double eps = 1e-9;
				if (std::abs(q.x()) > h.x() + eps ||
					std::abs(q.y()) > h.y() + eps ||
					std::abs(q.z()) > h.z() + eps) return false;
				const double t = 2.0 * res;  // shell_thickness
				const double ax = std::abs(q.x());
				const double ay = std::abs(q.y());
				const double az = std::abs(q.z());
				bool near_x = std::abs(ax - h.x()) <= t;
				bool near_y = std::abs(ay - h.y()) <= t;
				bool near_z = std::abs(az - h.z()) <= t;
				return near_x || near_y || near_z;
			};
			auto on_far_face = [&](const Eigen::Vector3d& p)->bool {
				Eigen::Vector3d qplant = obb.R.transpose() * (share_data->plant_obb.c - obb.c); // plant center in view local
				int axis = 0;
				if (std::abs(qplant.y()) > std::abs(qplant.x())) axis = 1;
				if (std::abs(qplant.z()) > std::abs(qplant[axis])) axis = 2;
				int far_sign = (qplant[axis] >= 0.0) ? -1 : +1; // plant on +axis => remove -face
				Eigen::Vector3d h = obb.half();
				const double t = 2.0 * res;  // shell_thickness
				Eigen::Vector3d q = obb.R.transpose() * (p - obb.c);
				if (axis == 0) {
					return (far_sign < 0) ? (std::abs(q.x() + h.x()) <= t) : (std::abs(q.x() - h.x()) <= t);
				}
				else if (axis == 1) {
					return (far_sign < 0) ? (std::abs(q.y() + h.y()) <= t) : (std::abs(q.y() - h.y()) <= t);
				}
				else {
					return (far_sign < 0) ? (std::abs(q.z() + h.z()) <= t) : (std::abs(q.z() - h.z()) <= t);
				}
			};
			// 5) iterate keys
			std::vector<octomap::OcTreeKey> view_keys;
			std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> view_space_key_set;
			for (unsigned int kx = kmin[0]; kx <= kmax[0]; ++kx) {
				for (unsigned int ky = kmin[1]; ky <= kmax[1]; ++ky) {
					for (unsigned int kz = kmin[2]; kz <= kmax[2]; ++kz) {
						octomap::OcTreeKey key(kx, ky, kz);

						// voxel center from THIS tree => consistent phase
						octomap::point3d c = view_obb_model.keyToCoord(key);
						Eigen::Vector3d p(c.x(), c.y(), c.z());
						if (!obb.contains(p)) continue;
						//if (!isShell(p)) continue;
						//if (on_far_face(p)) continue;   // 需要去掉“最远平行面”就打开

						if (view_space_key_set.insert(key).second) {
							view_keys.push_back(key);
							view_obb_model.setNodeValue(key, view_obb_model.getProbHitLog(), true);
							view_obb_model.integrateNodeColor(key, 0, 255, 0);
						}
					}
				}
			}
			view_obb_model.updateInnerOccupancy();
			// 转离散化结果进PCL点云
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_view_obb(new pcl::PointCloud<pcl::PointXYZ>);
			for (const auto& key : view_keys) {
				octomap::point3d center = view_obb_model.keyToCoord(key);
				pcl::PointXYZ pt;
				pt.x = center.x();
				pt.y = center.y();
				pt.z = center.z();
				cloud_view_obb->points.push_back(pt);
			}
			// 可视化
			if (share_data->debug && share_data->show) {
				pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("ViewSpace-GT"));
				viewer->setBackgroundColor(0, 0, 0);
				// 可视化 OBB 内的体素（绿色）和 GT 模型点云（红色）
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
				for (const auto& pt : cloud_view_obb->points) {
					pcl::PointXYZRGB pt_vis;
					pt_vis.x = pt.x;
					pt_vis.y = pt.y;
					pt_vis.z = pt.z;
					pt_vis.r = 0;
					pt_vis.g = 255;
					pt_vis.b = 0;
					cloud_vis->points.push_back(pt_vis);
				}
				for (const auto& pt : share_data->cloud_pcd->points) {
					pcl::PointXYZRGB pt_vis;
					pt_vis.x = pt.x;
					pt_vis.y = pt.y;
					pt_vis.z = pt.z;
					pt_vis.r = 255;
					pt_vis.g = 0;
					pt_vis.b = 0;
					cloud_vis->points.push_back(pt_vis);
				}
				viewer->addPointCloud<pcl::PointXYZRGB>(cloud_vis, "view_space");
				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "view_space");
				show_cloud(viewer);
			}
			cout << "View space discretization done. Number of voxels in view space: " << view_keys.size() << endl;
			//第三步对于每个表面点，生成其candidate direction（使用kdtree在cloud_view_obb上搜索share_data->ray_max_dis半径内的点）
			//对于每个view candidate，使用octomap的castRay判断是否可见
			int num_visible_points = 0;
			int current_num = 0;
			// 构建 KD-Tree 以加速邻域搜索
			pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
			kdtree.setSortedResults(true);
			kdtree.setInputCloud(cloud_view_obb);
			// 遍历每个 GT 表面点
			double time_start = clock();
			std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash> visible_surface_key_map_to_view;
			for (const auto& key : ground_truth_keys) {
				if (current_num % 100 == 0) {
					cout << "Processing GT point " << current_num + 1 << " / " << ground_truth_keys.size() << " with time " << double(clock() - time_start) / CLOCKS_PER_SEC << " seconds." << endl;
				}
				current_num++;

				//如果之前有扫到这个点了（之前的点可能命中了这个点），就不再计算了，直接跳过
				if (visible_surface_key_map_to_view.find(key) != visible_surface_key_map_to_view.end()) {
					num_visible_points++;
					continue;
				}

				octomap::point3d center = ground_truth_model.keyToCoord(key);
				pcl::PointXYZ search_point;
				search_point.x = center.x();
				search_point.y = center.y();
				search_point.z = center.z();
				std::vector<int> point_idx_radius_search;
				std::vector<float> point_radius_squared_distance;
				if (kdtree.radiusSearch(search_point, share_data->ray_max_dis, point_idx_radius_search, point_radius_squared_distance) > 0) {
					//可视化当前GT表面点和搜索到的view space点
					if (share_data->debug && share_data->show) {
						pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Search"));
						viewer->setBackgroundColor(0, 0, 0);
						// 可视化 OBB 内的体素（绿色）和 GT 模型点云（红色）
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
						for (size_t j = 0; j < point_idx_radius_search.size(); ++j) {
							if (j > 100000) continue; // 显示前面的，看看是不是按照距离排序的（kdtree.radiusSearch 设置了 sorted 结果，所以越前面越近）
							const auto& vp = cloud_view_obb->points[point_idx_radius_search[j]];
							pcl::PointXYZRGB pt_vis;
							pt_vis.x = vp.x;
							pt_vis.y = vp.y;
							pt_vis.z = vp.z;
							pt_vis.r = 0;
							pt_vis.g = 255;
							pt_vis.b = 0;
							cloud_vis->points.push_back(pt_vis);
						}
						pcl::PointXYZRGB pt_vis;
						pt_vis.x = center.x();
						pt_vis.y = center.y();
						pt_vis.z = center.z();
						pt_vis.r = 255;
						pt_vis.g = 0;
						pt_vis.b = 0;
						cloud_vis->points.push_back(pt_vis);
						viewer->addPointCloud<pcl::PointXYZRGB>(cloud_vis, "view_space");
						viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "view_space");
						show_cloud(viewer);
					}

					//对于每个搜索到的点，计算方向向量
					bool visible = false;
					Eigen::Vector3d visible_view_point;
					for (size_t j = 0; j < point_idx_radius_search.size(); ++j) {
						//std::cout << j << " d2=" << point_radius_squared_distance[j] << " idx=" << point_idx_radius_search[j] << "\n";

						int idx = point_idx_radius_search[j];
						const auto& vp = cloud_view_obb->points[idx];
						octomap::point3d candidate(vp.x, vp.y, vp.z);
						octomap::point3d origin = candidate;
						octomap::point3d dir(center.x() - candidate.x(), center.y() - candidate.y(), center.z() - candidate.z());

						octomap::OcTreeKey origin_key;
						if (!ground_truth_model.coordToKeyChecked(origin, origin_key)) {
							cout << "Warning: candidate view point is out of GT octomap bounds. This may be due to incomplete map or too large ray_max_dis. Please check if the GT octomap covers the area and if ray_max_dis is set appropriately." << endl;
							continue;
						}
						if (origin_key == key) {
							cout << "Error: View in Plant space check." << endl;
							continue;
						}

						octomap::point3d hit;
						bool hit_occ = ground_truth_model.castRay(origin, dir, hit, true, share_data->ray_max_dis);
						if (!hit_occ) {
							// 如果 range 足够，应该能撞到；否则说明 range 不够或地图没包含那边
							cout << "Warning: ray from surface point to view space point did not hit anything. This may be due to insufficient ray range or incomplete map. Please check if ray_max_dis is large enough and the GT octomap covers the area." << endl;
							continue;
						}

						octomap::OcTreeKey hit_key;
						if (!ground_truth_model.coordToKeyChecked(hit, hit_key)) {
							cout << "Warning: ray hit point is out of GT octomap bounds. This may be due to incomplete map or too large ray_max_dis. Please check if the GT octomap covers the area and if ray_max_dis is set appropriately." << endl;
							continue;
						}

						if (hit_key == key) {
							// 最终命中了GT 表面点自己，说明该方向可见
							visible = true;
							visible_view_point = Eigen::Vector3d(vp.x, vp.y, vp.z);
						}
						else {
							// 只考虑命中自己太严格命中了其他点实际上说明那个点是可见的，但是这个点是不可见的
							visible = false;
							visible_surface_key_map_to_view[hit_key] = Eigen::Vector3d(vp.x, vp.y, vp.z);
						}

						if (visible) break; // 找到一个可见方向就够了
					}
					//cout << "GT point " << current_num << ": " << point_idx_radius_search.size() << " view candidates, visible=" << visible << endl;
					
					//如果至少有一个方向可见，则该点可见
					if (visible) {
						num_visible_points++;
						visible_surface_key_map_to_view[key] = visible_view_point;
						// 输出离散化结果（可选），加入一条center to visible_end_point的射线到cloud_view_obb中（蓝色）
						if (share_data->debug && share_data->show) {
							pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Ray"));
							viewer->setBackgroundColor(0, 0, 0);
							// 可视化 OBB 内的体素（绿色）和 GT 模型点云（红色）
							pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
							for (size_t j = 0; j < point_idx_radius_search.size(); ++j) {
								if (j > 100000) continue; // 显示前面的，看看是不是按照距离排序的（kdtree.radiusSearch 设置了 sorted 结果，所以越前面越近）
								const auto& vp = cloud_view_obb->points[point_idx_radius_search[j]];
								pcl::PointXYZRGB pt_vis;
								pt_vis.x = vp.x;
								pt_vis.y = vp.y;
								pt_vis.z = vp.z;
								pt_vis.r = 0;
								pt_vis.g = 255;
								pt_vis.b = 0;
								cloud_vis->points.push_back(pt_vis);
							}
							for (const auto& pt : share_data->cloud_pcd->points) {
								pcl::PointXYZRGB pt_vis;
								pt_vis.x = pt.x;
								pt_vis.y = pt.y;
								pt_vis.z = pt.z;
								pt_vis.r = 255;
								pt_vis.g = 0;
								pt_vis.b = 0;
								cloud_vis->points.push_back(pt_vis);
							}
							viewer->addPointCloud<pcl::PointXYZRGB>(cloud_vis, "view_space");
							viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "view_space");
							// 加入射线（蓝色）
							pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ray(new pcl::PointCloud<pcl::PointXYZRGB>);
							const double step = share_data->ground_truth_resolution * 0.5; // 点间距
							Eigen::Vector3d p0(center.x(), center.y(), center.z());
							Eigen::Vector3d p1(visible_view_point.x(), visible_view_point.y(), visible_view_point.z());
							Eigen::Vector3d d = p1 - p0;
							double L = d.norm();
							if (L > 1e-9) {
								d /= L;
								int n = std::max(2, (int)std::ceil(L / step));
								for (int i = 0; i <= n; ++i) {
									Eigen::Vector3d p = p0 + d * (L * (double(i) / double(n)));
									pcl::PointXYZRGB pr;
									pr.x = (float)p.x(); pr.y = (float)p.y(); pr.z = (float)p.z();
									pr.r = 0; pr.g = 0; pr.b = 255; // 蓝色
									cloud_ray->points.push_back(pr);
								}
							}
							viewer->addPointCloud<pcl::PointXYZRGB>(cloud_ray, "ray");
							viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "ray");
							show_cloud(viewer);
						}
					}
				}
			}
			cout << "Number of visible points: " << num_visible_points << endl;
			cout << "Oracle visible pointcloud percentage: " << static_cast<double>(num_visible_points) / ground_truth_keys.size() * 100.0 << "%" << endl;
			//保存结果
			share_data->access_directory(share_data->gt_path + "/GT_OracleVisible/");
			ofstream fout_GT_points_num(share_data->gt_path + "/GT_OracleVisible/" + names[i] + ".txt");
			fout_GT_points_num << num_visible_points << '\t' << ground_truth_keys.size() << endl;
			fout_GT_points_num.close();
			ofstream fout_GT_surface_viewspace(share_data->gt_path + "/GT_OracleVisible/" + names[i] + "_surface_viewspace.txt");
			for (const auto& it : visible_surface_key_map_to_view) {
				const auto& surface_key = it.first;
				const auto& view_point = it.second;
				auto surface_center = ground_truth_model.keyToCoord(surface_key);
				fout_GT_surface_viewspace << surface_center.x() << '\t' << surface_center.y() << '\t' << surface_center.z() << '\t'
					<< view_point(0) << '\t' << view_point(1) << '\t' << view_point(2) << '\n';
			}
			fout_GT_surface_viewspace.close();
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_gt_points(new pcl::PointCloud<pcl::PointXYZRGB>);
			for (const auto& it : visible_surface_key_map_to_view) {
				octomap::point3d center = ground_truth_model.keyToCoord(it.first);
				pcl::PointXYZRGB pt_vis;
				pt_vis.x = center.x();
				pt_vis.y = center.y();
				pt_vis.z = center.z();
				pt_vis.r = 255;
				pt_vis.g = 0;
				pt_vis.b = 0;
				cloud_gt_points->points.push_back(pt_vis);
			}
			pcl::io::savePCDFileBinary(share_data->gt_path + "/GT_OracleVisible/" + names[i] + "_visible_points.pcd", *cloud_gt_points);
			//可选可视化可见表面点（红色）和命中viewspace的点（绿色）
			if (share_data->show) {
				pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Visible Surface"));
				viewer->setBackgroundColor(0, 0, 0);
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ray(new pcl::PointCloud<pcl::PointXYZRGB>);
				for (const auto& it : visible_surface_key_map_to_view) {
					const auto& surface_key = it.first;
					const auto& view_point = it.second;
					auto surface_center = ground_truth_model.keyToCoord(surface_key);
					pcl::PointXYZRGB surface_vis;
					surface_vis.x = surface_center.x();
					surface_vis.y = surface_center.y();
					surface_vis.z = surface_center.z();
					surface_vis.r = 255;
					surface_vis.g = 0;
					surface_vis.b = 0;
					cloud_vis->points.push_back(surface_vis);
					pcl::PointXYZRGB viewspace_vis;
					viewspace_vis.x = view_point(0);
					viewspace_vis.y = view_point(1);
					viewspace_vis.z = view_point(2);
					viewspace_vis.r = 0;
					viewspace_vis.g = 255;
					viewspace_vis.b = 0;
					cloud_vis->points.push_back(viewspace_vis);
					const double step = share_data->ground_truth_resolution * 0.5; // 点间距
					Eigen::Vector3d p0(surface_center.x(), surface_center.y(), surface_center.z());
					Eigen::Vector3d p1(view_point(0), view_point(1), view_point(2));
					Eigen::Vector3d d = p1 - p0;
					double L = d.norm();
					if (L > 1e-9) {
						d /= L;
						int n = std::max(2, (int)std::ceil(L / step));
						for (int i = 0; i <= n; ++i) {
							Eigen::Vector3d p = p0 + d * (L * (double(i) / double(n)));
							pcl::PointXYZRGB pr;
							pr.x = (float)p.x(); pr.y = (float)p.y(); pr.z = (float)p.z();
							pr.r = 0; pr.g = 0; pr.b = 255; // 蓝色
							cloud_ray->points.push_back(pr);
						}
					}
				}
				viewer->addPointCloud<pcl::PointXYZRGB>(cloud_vis, "visible_surface");
				viewer->addPointCloud<pcl::PointXYZRGB>(cloud_ray, "ray");
				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "visible_surface");
				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ray");
				show_cloud(viewer);
			}
			//end of for each model
		}
	}
	else if (mode == GetGTPoints) {
		for (int i = 0; i < names.size(); i++) {
			cout << "Get GT visible pointcloud number of model " << names[i] << endl;
			
			//for (int rotate_state = 0; rotate_state < 8; rotate_state++) {
			for (int x = 0; x < rotate_states.size(); x++) {
				int rotate_state = rotate_states[x];
				//数据区初始化
				share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], rotate_state, first_view_ids[0], method_id, move_test_on, combined_test_on);

				//!!!!!这个只是debug用的，目的是只生成passive view的点云!!!!!
				bool only_passive = share_data->debug;

				if (!only_passive && share_data->dynamic_candidate_look_ats.size() == 0) {
					cout << "No dynamic candidate view look ats. Skip. Run deformation first!" << endl;
					continue;
				}

				//NBV规划期初始化
				nbv_plan = new NBV_Planner(share_data);

				//get forbidden views (who is dynamic_candidate and its look_at == (0,0,0))
				set<int> forbidden_views;
				for (int j = 0; j < nbv_plan->now_view_space->views.size(); j++) {
					if (j < share_data->passive_init_views.size()) continue; //被动视点不考虑
					if ((nbv_plan->now_view_space->views[j].look_at - Eigen::Vector3d(0, 0, 0)).norm() < 1e-6) {
						forbidden_views.insert(j);
						//cout << "forbidden view " << j << " added." << endl;
					}
				}

				//保存GT
				share_data->access_directory(share_data->gt_path + "/GT_points_" + share_data->look_at_group_str + "/" + names[i] + "_r" + to_string(rotate_state) + "/");
				
				//share_data->ground_truth_model->write(share_data->gt_path + "/GT_points_" + share_data->look_at_group_str + "/" + names[i] + "_r" + to_string(rotate_state) + "/gt_otcomap.ot");

				//获取全部点云，从0-31顺序
				for (int j = 0; j < nbv_plan->now_view_space->views.size(); j++) {
					//!!!!!这个只是debug用的，目的是只生成passive view的点云!!!!!
					if (only_passive && j >= share_data->passive_init_views.size()) break;

					if (forbidden_views.find(j) != forbidden_views.end()) { //如果是禁止视点，则手动处理，无需感知，push一个占位点云进去
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
						//pcl::PointCloud<pcl::PointXYZRGB>::Ptr no_table(new pcl::PointCloud<pcl::PointXYZRGB>);
						pcl::PointXYZRGB s;
						s.x = s.y = s.z = 0.0f;
						s.r = s.g = s.b = 0;
						cloud->points.resize(1);
						cloud->points[0] = s;
						cloud->width = 1;
						cloud->height = 1;
						cloud->is_dense = false;
						//no_table->points.resize(1);
						//no_table->points[0] = s;
						//no_table->width = 1;
						//no_table->height = 1;
						//no_table->is_dense = false;
						share_data->clouds.push_back(cloud);
						//share_data->clouds_notable.push_back(no_table);
						*share_data->cloud_final += *cloud;
						share_data->vaild_clouds++;
					}
					else { //正常视点，调用感知模块获取点云
						nbv_plan->percept->precept(&nbv_plan->now_view_space->views[j]);
						//如果点云为空则保存一个占位点，否则pcl会报错
						if (share_data->clouds[j]->empty()) {
							pcl::PointXYZRGB s;
							s.x = s.y = s.z = 0.0f;
							s.r = s.g = s.b = 0;
							share_data->clouds[j]->points.resize(1);
							share_data->clouds[j]->points[0] = s;
							share_data->clouds[j]->width = 1;
							share_data->clouds[j]->height = 1;
							share_data->clouds[j]->is_dense = false;
						}
						//if (share_data->clouds_notable[j]->empty()) {
						//	pcl::PointXYZRGB s;
						//	s.x = s.y = s.z = 0.0f;
						//	s.r = s.g = s.b = 0;
						//	share_data->clouds_notable[j]->points.resize(1);
						//	share_data->clouds_notable[j]->points[0] = s;
						//	share_data->clouds_notable[j]->width = 1;
						//	share_data->clouds_notable[j]->height = 1;
						//	share_data->clouds_notable[j]->is_dense = false;
						//}
					}
					//保存到磁盘
					pcl::io::savePCDFile<pcl::PointXYZRGB>(share_data->gt_path + "/GT_points_" + share_data->look_at_group_str + "/" + names[i] + "_r" + to_string(rotate_state) + "/cloud_view" + to_string(j) + ".pcd", *share_data->clouds[j]);
					//pcl::io::savePCDFile<pcl::PointXYZRGB>(share_data->gt_path + "/GT_points_" + share_data->look_at_group_str + "/" + names[i] + "_r" + to_string(rotate_state) + "/cloud_notable_view" + to_string(j) + ".pcd", *share_data->clouds_notable[j]);
				}

				//!!!!!这个只是debug用的，目的是只生成passive view的点云!!!!!
				if (only_passive) continue;

				//根据Octomap精度统计可见点个数
				double now_time = clock();
				int num = 0;
				unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
				for (int j = 0; j < share_data->cloud_final->points.size(); j++) {
					octomap::OcTreeKey key;
					if(!share_data->ground_truth_model->coordToKeyChecked(share_data->cloud_final->points[j].x, share_data->cloud_final->points[j].y, share_data->cloud_final->points[j].z, key)) {
						cout << "Warning: point " << j << " in final cloud is out of GT octomap bounds. This may be due to incomplete map or points outside the area. Please check if the GT octomap covers the area and if the point cloud is correct." << endl;
						continue;
					}
					if (voxel->find(key) == voxel->end()) {
						(*voxel)[key] = num++;
					}
				}
				cout << "GT visible pointcloud get with " << num << " points." << endl;
				cout << "GT visible pointcloud get with " << voxel->size() << " points." << endl;
				cout << "GT visible pointcloud get with total points number " << share_data->cloud_final->points.size() << endl;
				cout << "GT visible pointcloud get with executed time " << clock() - now_time << " ms." << endl;

				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_GT_points(new pcl::PointCloud<pcl::PointXYZRGB>);
				//根据Octomap的分辨率生成点云
				for (auto& kv : *voxel) {
					octomap::OcTreeKey key = kv.first;
					octomap::point3d center = share_data->ground_truth_model->keyToCoord(key);
					pcl::PointXYZRGB point;
					point.x = center.x();
					point.y = center.y();
					point.z = center.z();
					cloud_GT_points->points.push_back(point);
				}
				cloud_GT_points->width = cloud_GT_points->points.size();
				cloud_GT_points->height = 1;
				cloud_GT_points->is_dense = false;
				pcl::io::savePCDFile<pcl::PointXYZRGB>(share_data->gt_path + "/GT_points_" + share_data->look_at_group_str + "/" + names[i] + "_r" + to_string(rotate_state) + "/cloud_GT_points.pcd", *cloud_GT_points);

				ofstream fout_GT_points_num(share_data->gt_path + "/GT_points_" + share_data->look_at_group_str + "/" + names[i] + "_r" + to_string(rotate_state) + "/all_candidate_visible_num.txt");
				fout_GT_points_num << voxel->size() << endl;
				cout << "Rotate " << rotate_state << " GT_points_num is " << voxel->size() << " ,rate is " << 1.0 * voxel->size() / share_data->cloud_points_number << endl;
				delete voxel;
				
				delete nbv_plan;
				delete share_data;
				
			}
			
		}
	}
	else if (mode == CheckOracle) {
		for (int i = 0; i < names.size(); i++) {
			share_data = new Share_Data("../DefaultConfiguration.yaml", names[i]);
			//使用GT点云生成GT octomap
			octomap::ColorOcTree ground_truth_model(share_data->ground_truth_resolution);
			std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> ground_truth_key_set;
			for (auto ptr = share_data->cloud_pcd->points.begin(); ptr != share_data->cloud_pcd->points.end(); ptr++)
			{
				octomap::OcTreeKey key;  bool key_have = ground_truth_model.coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key);
				if (key_have) {
					octomap::ColorOcTreeNode* voxel = ground_truth_model.search(key);
					if (voxel == NULL) {
						ground_truth_model.setNodeValue(key, ground_truth_model.getProbHitLog(), true);
						ground_truth_model.integrateNodeColor(key, (*ptr).r, (*ptr).g, (*ptr).b);
						ground_truth_key_set.insert(key);
					}
				}
			}
			ground_truth_model.updateInnerOccupancy();

			string cloud_oracle_path = share_data->gt_path + "/GT_OracleVisible/" + names[i] + "_visible_points.pcd";
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_oracle(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::io::loadPCDFile<pcl::PointXYZ>(cloud_oracle_path, *cloud_oracle);
			unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> cloud_oracle_keys;
			for (const auto& point : cloud_oracle->points) {
				octomap::OcTreeKey key;
				if (ground_truth_model.coordToKeyChecked(point.x, point.y, point.z, key)) {
					cloud_oracle_keys.insert(key);
				}
			}

			string cloud_union_path = share_data->gt_path + "/GT_points_" + share_data->look_at_group_str + "/" + names[i] + "_r" + to_string(share_data->rotate_state) + "/cloud_GT_points.pcd";
			unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> cloud_union_keys;
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_union(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::io::loadPCDFile<pcl::PointXYZ>(cloud_union_path, *cloud_union);
			for (const auto& point : cloud_union->points) {
				octomap::OcTreeKey key;
				if (ground_truth_model.coordToKeyChecked(point.x, point.y, point.z, key)) {
					cloud_union_keys.insert(key);
				}
			}

			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
			for (const auto& point : cloud_oracle->points) {
				octomap::OcTreeKey key;
				if (!ground_truth_model.coordToKeyChecked(point.x, point.y, point.z, key)) continue;
				pcl::PointXYZRGB point_vis;
				point_vis.x = point.x;
				point_vis.y = point.y;
				point_vis.z = point.z;
				if (cloud_union_keys.count(key)) { //共同的用绿色
					point_vis.r = 0;
					point_vis.g = 255;
					point_vis.b = 0;
				}
				else {//单独用红色
					point_vis.r = 255;
					point_vis.g = 0;
					point_vis.b = 0;
				}
				if (!ground_truth_key_set.count(key)) { //如果oracle点不在GT里，说明是oracle独有的点，用黑色显示（理论上不应该有）
					point_vis.r = 0;
					point_vis.g = 0;
					point_vis.b = 0;
				}
				cloud_vis->points.push_back(point_vis);
			}
			for (const auto& point : cloud_union->points) {
				octomap::OcTreeKey key;
				if (!ground_truth_model.coordToKeyChecked(point.x, point.y, point.z, key)) continue;
				pcl::PointXYZRGB point_vis;
				point_vis.x = point.x;
				point_vis.y = point.y;
				point_vis.z = point.z;
				if (cloud_oracle_keys.count(key)) { //共同的用绿色
					point_vis.r = 0;
					point_vis.g = 255;
					point_vis.b = 0;
				}
				else {//单独用蓝色
					point_vis.r = 0;
					point_vis.g = 0;
					point_vis.b = 255;
				}
				if (!ground_truth_key_set.count(key)) { //如果oracle点不在GT里，说明是oracle独有的点，用黑色显示（理论上不应该有）
					point_vis.r = 0;
					point_vis.g = 0;
					point_vis.b = 0;
				}
				cloud_vis->points.push_back(point_vis);
			}
			cloud_vis->width = cloud_vis->points.size();
			cloud_vis->height = 1;
			cloud_vis->is_dense = false;
			pcl::io::savePCDFile<pcl::PointXYZRGB>(share_data->gt_path + "/set_diff.pcd", *cloud_vis);
		}
	}
	else if (mode == GetOracleVisiblePoisson) {
		for (int i = 0; i < names.size(); i++) {
			cout << "Get Oracle visible pointcloud number of model " << names[i] << endl;
			share_data = new Share_Data("../DefaultConfiguration.yaml", names[i]);
			//第一步：使用GT点云生成GT octomap
			octomap::ColorOcTree ground_truth_model(share_data->ground_truth_resolution);
			cout << "use resolution " << share_data->ground_truth_resolution << " to generate GT octomap." << endl;
			std::vector<octomap::OcTreeKey> ground_truth_keys;
			std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> ground_truth_key_set;
			for (auto ptr = share_data->cloud_pcd->points.begin(); ptr != share_data->cloud_pcd->points.end(); ptr++)
			{
				octomap::OcTreeKey key;  bool key_have = ground_truth_model.coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key);
				if (key_have) {
					octomap::ColorOcTreeNode* voxel = ground_truth_model.search(key);
					if (voxel == NULL) {
						ground_truth_model.setNodeValue(key, ground_truth_model.getProbHitLog(), true);
						ground_truth_model.integrateNodeColor(key, (*ptr).r, (*ptr).g, (*ptr).b);
						if (ground_truth_key_set.insert(key).second) {
							ground_truth_keys.push_back(key);
						}
					}
				}
			}
			ground_truth_model.updateInnerOccupancy();
			int full_voxels = 0;
			for (octomap::ColorOcTree::leaf_iterator it = ground_truth_model.begin_leafs(), end = ground_truth_model.end_leafs(); it != end; ++it) {
				full_voxels++;
			}
			cout << "Ground truth octomap generated with " << full_voxels << " full voxels." << endl;
			// 第二步：读取泊松采样的视点
			vector<Eigen::Vector3d> oracle_views;
			string oracle_views_path = share_data->environment_path + "/" + share_data->room_str + "_candidate_views.txt";
			ifstream fin_oracle_views(oracle_views_path);
			if (fin_oracle_views.is_open()) {
				while (!fin_oracle_views.eof()) {
					double x, y, z;
					fin_oracle_views >> x >> y >> z;
					if (fin_oracle_views.eof()) break;
					oracle_views.push_back(Eigen::Vector3d(x, y, z));
					if (!share_data->view_obb.contains(oracle_views.back())) {
						cout << "views out of BBX. check." << endl;
					}
				}
				fin_oracle_views.close();
			}
			else {
				cout << "No oracle views file found." << endl;
			}
			cout << "oracle_views size: " << oracle_views.size() << endl;
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_oracle_view(new pcl::PointCloud<pcl::PointXYZ>);
			for (const auto& view : oracle_views) {
				pcl::PointXYZ pt;
				pt.x = view.x();
				pt.y = view.y();
				pt.z = view.z();
				cloud_oracle_view->points.push_back(pt);
			}
			// 可视化
			if (share_data->debug && share_data->show) {
				pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("ViewSpace-GT"));
				viewer->setBackgroundColor(0, 0, 0);
				// 可视化 OBB 内的泊松采样（绿色）和 GT 模型点云（红色）
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
				for (const auto& pt : cloud_oracle_view->points) {
					pcl::PointXYZRGB pt_vis;
					pt_vis.x = pt.x;
					pt_vis.y = pt.y;
					pt_vis.z = pt.z;
					pt_vis.r = 0;
					pt_vis.g = 255;
					pt_vis.b = 0;
					cloud_vis->points.push_back(pt_vis);
				}
				for (const auto& pt : share_data->cloud_pcd->points) {
					pcl::PointXYZRGB pt_vis;
					pt_vis.x = pt.x;
					pt_vis.y = pt.y;
					pt_vis.z = pt.z;
					pt_vis.r = 255;
					pt_vis.g = 0;
					pt_vis.b = 0;
					cloud_vis->points.push_back(pt_vis);
				}
				viewer->addPointCloud<pcl::PointXYZRGB>(cloud_vis, "view_space");
				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "view_space");
				show_cloud(viewer);
			}
			//第三步对于每个表面点，只找邻域share_data->oracle_knn个
			//对于每个view candidate，使用octomap的castRay判断是否可见
			int num_visible_points = 0;
			int current_num = 0;
			// 构建 KD-Tree 以加速邻域搜索
			pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
			kdtree.setInputCloud(cloud_oracle_view);
			// 遍历每个 GT 表面点
			double time_start = clock();
			std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash> visible_surface_key_map_to_view;
			for (const auto& key : ground_truth_keys) {
				if (current_num % 100 == 0) {
					cout << "Processing GT point " << current_num + 1 << " / " << ground_truth_keys.size() << " with time " << double(clock() - time_start) / CLOCKS_PER_SEC << " seconds." << endl;
				}
				current_num++;

				//如果之前有扫到这个点了（之前的点可能命中了这个点），就不再计算了，直接跳过
				if (visible_surface_key_map_to_view.find(key) != visible_surface_key_map_to_view.end()) {
					num_visible_points++;
					continue;
				}

				octomap::point3d center = ground_truth_model.keyToCoord(key);
				pcl::PointXYZ search_point;
				search_point.x = center.x();
				search_point.y = center.y();
				search_point.z = center.z();
				std::vector<int> point_idx_knn_search;
				std::vector<float> point_knn_squared_distance;
				if (kdtree.nearestKSearch(search_point, share_data->oracle_knn, point_idx_knn_search, point_knn_squared_distance) > 0) {
					//可视化当前GT表面点和搜索到的view space点
					if (share_data->debug && share_data->show) {
						pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Search"));
						viewer->setBackgroundColor(0, 0, 0);
						// 可视化 OBB 内的体素（绿色）和 GT 模型点云（红色）
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
						for (size_t j = 0; j < point_idx_knn_search.size(); ++j) {
							const auto& vp = cloud_oracle_view->points[point_idx_knn_search[j]];
							pcl::PointXYZRGB pt_vis;
							pt_vis.x = vp.x;
							pt_vis.y = vp.y;
							pt_vis.z = vp.z;
							pt_vis.r = 0;
							pt_vis.g = 255;
							pt_vis.b = 0;
							cloud_vis->points.push_back(pt_vis);
						}
						pcl::PointXYZRGB pt_vis;
						pt_vis.x = center.x();
						pt_vis.y = center.y();
						pt_vis.z = center.z();
						pt_vis.r = 255;
						pt_vis.g = 0;
						pt_vis.b = 0;
						cloud_vis->points.push_back(pt_vis);
						viewer->addPointCloud<pcl::PointXYZRGB>(cloud_vis, "view_space");
						viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "view_space");
						show_cloud(viewer);
					}

					//对于每个搜索到的点，计算方向向量
					bool visible = false;
					Eigen::Vector3d visible_view_point;
					for (size_t j = 0; j < point_idx_knn_search.size(); ++j) {

						int idx = point_idx_knn_search[j];
						const auto& vp = cloud_oracle_view->points[idx];
						octomap::point3d candidate(vp.x, vp.y, vp.z);
						octomap::point3d origin = candidate;
						octomap::point3d dir(center.x() - candidate.x(), center.y() - candidate.y(), center.z() - candidate.z());

						octomap::OcTreeKey origin_key;
						if (!ground_truth_model.coordToKeyChecked(origin, origin_key)) {
							cout << "Warning: candidate view point is out of GT octomap bounds. This may be due to incomplete map or too large ray_max_dis. Please check if the GT octomap covers the area and if ray_max_dis is set appropriately." << endl;
							continue;
						}
						if (origin_key == key) {
							cout << "Error: View in Plant space check." << endl;
							continue;
						}

						octomap::point3d hit;
						bool hit_occ = ground_truth_model.castRay(origin, dir, hit, true, share_data->ray_max_dis);
						if (!hit_occ) {
							//cout << "Warning: ray from surface point to view space point did not hit anything. This may be due to insufficient ray range or incomplete map. Please check if ray_max_dis is large enough and the GT octomap covers the area." << endl;
							continue;
						}

						octomap::OcTreeKey hit_key;
						if (!ground_truth_model.coordToKeyChecked(hit, hit_key)) {
							cout << "Warning: ray hit point is out of GT octomap bounds. This may be due to incomplete map or too large ray_max_dis. Please check if the GT octomap covers the area and if ray_max_dis is set appropriately." << endl;
							continue;
						}

						if (hit_key == key) {
							// 最终命中了GT 表面点自己，说明该方向可见
							visible = true;
							visible_view_point = Eigen::Vector3d(vp.x, vp.y, vp.z);
						}
						else {
							// 只考虑命中自己太严格命中了其他点实际上说明那个点是可见的，但是这个点是不可见的
							visible = false;
							visible_surface_key_map_to_view[hit_key] = Eigen::Vector3d(vp.x, vp.y, vp.z);
						}

						if (visible) break; // 找到一个可见方向就够了
					}
					//cout << "GT point " << current_num << ": " << point_idx_radius_search.size() << " view candidates, visible=" << visible << endl;

					//如果至少有一个方向可见，则该点可见
					if (visible) {
						num_visible_points++;
						visible_surface_key_map_to_view[key] = visible_view_point;
						// 输出离散化结果（可选），加入一条center to visible_end_point的射线到cloud_view_obb中（蓝色）
						if (share_data->debug && share_data->show) {
							pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Ray"));
							viewer->setBackgroundColor(0, 0, 0);
							// 可视化 OBB 内的体素（绿色）和 GT 模型点云（红色）
							pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
							for (size_t j = 0; j < point_idx_knn_search.size(); ++j) {
								const auto& vp =  cloud_oracle_view->points[point_idx_knn_search[j]];
								pcl::PointXYZRGB pt_vis;
								pt_vis.x = vp.x;
								pt_vis.y = vp.y;
								pt_vis.z = vp.z;
								pt_vis.r = 0;
								pt_vis.g = 255;
								pt_vis.b = 0;
								cloud_vis->points.push_back(pt_vis);
							}
							for (const auto& pt : share_data->cloud_pcd->points) {
								pcl::PointXYZRGB pt_vis;
								pt_vis.x = pt.x;
								pt_vis.y = pt.y;
								pt_vis.z = pt.z;
								pt_vis.r = 255;
								pt_vis.g = 0;
								pt_vis.b = 0;
								cloud_vis->points.push_back(pt_vis);
							}
							viewer->addPointCloud<pcl::PointXYZRGB>(cloud_vis, "view_space");
							viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "view_space");
							// 加入射线（蓝色）
							pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ray(new pcl::PointCloud<pcl::PointXYZRGB>);
							const double step = share_data->ground_truth_resolution * 0.5; // 点间距
							Eigen::Vector3d p0(center.x(), center.y(), center.z());
							Eigen::Vector3d p1(visible_view_point.x(), visible_view_point.y(), visible_view_point.z());
							Eigen::Vector3d d = p1 - p0;
							double L = d.norm();
							if (L > 1e-9) {
								d /= L;
								int n = std::max(2, (int)std::ceil(L / step));
								for (int i = 0; i <= n; ++i) {
									Eigen::Vector3d p = p0 + d * (L * (double(i) / double(n)));
									pcl::PointXYZRGB pr;
									pr.x = (float)p.x(); pr.y = (float)p.y(); pr.z = (float)p.z();
									pr.r = 0; pr.g = 0; pr.b = 255; // 蓝色
									cloud_ray->points.push_back(pr);
								}
							}
							viewer->addPointCloud<pcl::PointXYZRGB>(cloud_ray, "ray");
							viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "ray");
							show_cloud(viewer);
						}
					}
				}
			}
			cout << "Number of visible points: " << num_visible_points << endl;
			cout << "Oracle visible pointcloud percentage: " << static_cast<double>(num_visible_points) / ground_truth_keys.size() * 100.0 << "%" << endl;
			//保存结果
			share_data->access_directory(share_data->gt_path + "/GT_OracleVisiblePoisson/");
			ofstream fout_GT_points_num(share_data->gt_path + "/GT_OracleVisiblePoisson/" + names[i] + ".txt");
			fout_GT_points_num << num_visible_points << '\t' << ground_truth_keys.size() << endl;
			fout_GT_points_num.close();
			ofstream fout_GT_surface_viewspace(share_data->gt_path + "/GT_OracleVisiblePoisson/" + names[i] + "_surface_viewspace.txt");
			for (const auto &it : visible_surface_key_map_to_view) {
				const auto& surface_key = it.first;
				const auto& view_point = it.second;
				auto surface_center = ground_truth_model.keyToCoord(surface_key);
				fout_GT_surface_viewspace << surface_center.x() << '\t' << surface_center.y() << '\t' << surface_center.z() << '\t'
					<< view_point(0) << '\t' << view_point(1) << '\t' << view_point(2) << '\n';
			}
			fout_GT_surface_viewspace.close();
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_gt_points(new pcl::PointCloud<pcl::PointXYZRGB>);
			for (const auto& it : visible_surface_key_map_to_view) {
				octomap::point3d center = ground_truth_model.keyToCoord(it.first);
				pcl::PointXYZRGB pt_vis;
				pt_vis.x = center.x();
				pt_vis.y = center.y();
				pt_vis.z = center.z();
				pt_vis.r = 255;
				pt_vis.g = 0;
				pt_vis.b = 0;
				cloud_gt_points->points.push_back(pt_vis);
			}
			pcl::io::savePCDFileBinary(share_data->gt_path + "/GT_OracleVisiblePoisson/" + names[i] + "_visible_points.pcd", *cloud_gt_points);
			//可选可视化可见表面点（红色）和命中viewspace的点（绿色）
			if (share_data->show) {
				pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Visible Surface"));
				viewer->setBackgroundColor(0, 0, 0);
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ray(new pcl::PointCloud<pcl::PointXYZRGB>);
				for (const auto& it : visible_surface_key_map_to_view) {
					const auto& surface_key = it.first;
					const auto& view_point = it.second;
					auto surface_center = ground_truth_model.keyToCoord(surface_key);
					pcl::PointXYZRGB surface_vis;
					surface_vis.x = surface_center.x();
					surface_vis.y = surface_center.y();
					surface_vis.z = surface_center.z();
					surface_vis.r = 255;
					surface_vis.g = 0;
					surface_vis.b = 0;
					cloud_vis->points.push_back(surface_vis);
					pcl::PointXYZRGB viewspace_vis;
					viewspace_vis.x = view_point(0);
					viewspace_vis.y = view_point(1);
					viewspace_vis.z = view_point(2);
					viewspace_vis.r = 0;
					viewspace_vis.g = 255;
					viewspace_vis.b = 0;
					cloud_vis->points.push_back(viewspace_vis);
					const double step = share_data->ground_truth_resolution * 0.5; // 点间距
					Eigen::Vector3d p0(surface_center.x(), surface_center.y(), surface_center.z());
					Eigen::Vector3d p1(view_point(0), view_point(1), view_point(2));
					Eigen::Vector3d d = p1 - p0;
					double L = d.norm();
					if (L > 1e-9) {
						d /= L;
						int n = std::max(2, (int)std::ceil(L / step));
						for (int i = 0; i <= n; ++i) {
							Eigen::Vector3d p = p0 + d * (L * (double(i) / double(n)));
							pcl::PointXYZRGB pr;
							pr.x = (float)p.x(); pr.y = (float)p.y(); pr.z = (float)p.z();
							pr.r = 0; pr.g = 0; pr.b = 255; // 蓝色
							cloud_ray->points.push_back(pr);
						}
					}
				}
				viewer->addPointCloud<pcl::PointXYZRGB>(cloud_vis, "visible_surface");
				viewer->addPointCloud<pcl::PointXYZRGB>(cloud_ray, "ray");
				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "visible_surface");
				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ray");
				show_cloud(viewer);
			}
			//end of for each model
		}
		// end of GetOracleVisiblePoisson
	}
	else if (mode == TestPoseNoise) {
		for (int i = 0; i < names.size(); i++) {
			cout << "Test Pose Noise of " << names[i] << endl;
			share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], 0, 0, method_id, move_test_on, combined_test_on);
			nbv_plan = new NBV_Planner(share_data);

			//读选取的视点
			vector<int> chosen_views = { share_data->first_view_id };
			int num_of_chosen_views;
			ifstream fin_num_of_chosen_views(share_data->save_path + "/all_needed_views.txt");
			if (fin_num_of_chosen_views.is_open()) {
				fin_num_of_chosen_views >> num_of_chosen_views;
				fin_num_of_chosen_views.close();
			}
			else {
				cout << "No all_needed_views file found." << endl;
				continue;
			}
			for (int j = 0; j <= num_of_chosen_views - 2; j++) {
				string view_id_file = share_data->save_path + "/movement/path" + to_string(j) + ".txt";
				ifstream fin_view_id(view_id_file);
				int view_id;
				fin_view_id >> view_id;
				chosen_views.push_back(view_id);
			}
			cout << "chosen_views num = " << chosen_views.size() << endl;

			bool check_view_id = true;
			for (int view_id_idx = 0; view_id_idx < share_data->passive_init_views.size(); view_id_idx++) {
				int view_id = chosen_views[view_id_idx];
				if (view_id >= share_data->passive_init_views.size()) {
					cout << "Error: view_id " << view_id << " out of passive init views range. First " << share_data->passive_init_views.size() << " views are passive init views." << endl;
					check_view_id = false;
				}
			}
			if (!check_view_id) {
				cout << "Error: check view id failed. Please check the chosen views." << endl;
				continue;
			}
			
			//注意这里假设有GT，因此成像是直接读取
			for (int j = 0; j < nbv_plan->now_view_space->views.size(); j++) {
				nbv_plan->percept->precept(&nbv_plan->now_view_space->views[j]);
			}

			vector<double> sigmas_t = { 0.0, 0.002, 0.005, 0.01 }; // 2mm, 5mm, 10mm
			vector<double> sigmas_r = { 0.0, 0.5, 1.0 , 2.0}; // 0.5 deg, 1 deg, 2 deg
			vector<bool> icp_ons = { false, true }; // 是否使用ICP后处理

			//vector<double> sigmas_t = { 0.0, 0.01 }; // 2mm, 5mm, 10mm
			//vector<double> sigmas_r = { 0.0, 2.0 }; // 0.5 deg, 1 deg, 2 deg
			//vector<bool> icp_ons = { false, true }; // 是否使用ICP后处理
			
			for (int noise_level_idx = 0; noise_level_idx < sigmas_t.size(); noise_level_idx++) {
				double sigma_t = sigmas_t[noise_level_idx];
				double sigma_r = sigmas_r[noise_level_idx] * M_PI / 180.0; // deg -> rad
				cout << "Testing noise level " << noise_level_idx << ": sigma_t = " << sigma_t << " m, sigma_r = " << sigma_r * 180.0 / M_PI << " deg" << endl;

				for (bool icp_on : icp_ons) {
					if (noise_level_idx == 0 && icp_on == true) continue;

					cout << "ICP " << (icp_on ? "ON" : "OFF") << endl;

					pcl::PointCloud<pcl::PointXYZRGB>::Ptr noisy_final_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

					//遍历选取的视点，添加噪声，计算噪声对可见点云的影响
					for (int view_id : chosen_views) {
						std::mt19937 rng_view(42 + view_id * 10007 + noise_level_idx * 100000);

						//如果是被动初始化的视点，直接使用原始点云
						if (view_id < share_data->passive_init_views.size()) {
							*noisy_final_cloud += *share_data->clouds[view_id];
						}
						else {
							// get pose
							nbv_plan->now_view_space->views[view_id].get_next_camera_pos();
							Eigen::Matrix4d T_wc_pose_mat = nbv_plan->now_view_space->views[view_id].pose.inverse().eval();
							Eigen::Isometry3d T_GT(T_wc_pose_mat);
							// add noise
							Eigen::Isometry3d dT = Eigen::Isometry3d::Identity();
							if (sigma_t > 0 || sigma_r > 0) dT = se3_noise::SampleDeltaT(rng_view, sigma_t, sigma_r);
							Eigen::Isometry3d dT_w = se3_noise::WorldRelTransformFromRightNoise(T_GT, dT);

							pcl::PointCloud<pcl::PointXYZRGB>::Ptr noisy_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
							for (const auto& pt : share_data->clouds[view_id]->points) {
								Eigen::Vector3d p_world(pt.x, pt.y, pt.z);
								// perturb a world point
								Eigen::Vector3d p_noisy = dT_w * p_world;
								pcl::PointXYZRGB pt_noisy;
								pt_noisy.x = p_noisy(0);
								pt_noisy.y = p_noisy(1);
								pt_noisy.z = p_noisy(2);
								pt_noisy.r = pt.r;
								pt_noisy.g = pt.g;
								pt_noisy.b = pt.b;
								noisy_cloud->points.push_back(pt_noisy);
							}
							noisy_cloud->width = noisy_cloud->points.size();
							noisy_cloud->height = 1;
							noisy_cloud->is_dense = false;

							if (icp_on) {
								// 使用ICP将有噪声的点云对齐到GT点云
								pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp_pcl;
								icp_pcl.setMaxCorrespondenceDistance(5.0 * share_data->ground_truth_resolution); // 设置最大对应点距离
								icp_pcl.setMaximumIterations(50);
								icp_pcl.setTransformationEpsilon(1e-7);
								icp_pcl.setEuclideanFitnessEpsilon(1e-6);

								pcl::VoxelGrid<pcl::PointXYZRGB> vg;
								vg.setLeafSize(2.0 * share_data->ground_truth_resolution, 2.0 * share_data->ground_truth_resolution, 2.0 * share_data->ground_truth_resolution);
								//icp_pcl.setInputSource(noisy_cloud);
								pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_ds(new pcl::PointCloud<pcl::PointXYZRGB>);
								vg.setInputCloud(noisy_cloud);
								vg.filter(*source_ds);
								icp_pcl.setInputSource(source_ds);
								///icp_pcl.setInputTarget(noisy_final_cloud);
								pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_ds(new pcl::PointCloud<pcl::PointXYZRGB>);
								vg.setInputCloud(noisy_final_cloud);
								vg.filter(*target_ds);
								icp_pcl.setInputTarget(target_ds);

								double max_mse = (2 * share_data->ground_truth_resolution) * (2 * share_data->ground_truth_resolution);
								pcl::PointCloud<pcl::PointXYZRGB> aligned_cloud;
								icp_pcl.align(aligned_cloud);

								if (icp_pcl.hasConverged() && icp_pcl.getFitnessScore() < max_mse) {
									cout << "ICP converged for view " << view_id << " with score " << icp_pcl.getFitnessScore() << endl;
									Eigen::Matrix4f T = icp_pcl.getFinalTransformation();
									pcl::PointCloud<pcl::PointXYZRGB> aligned_full;
									pcl::transformPointCloud(*noisy_cloud, aligned_full, T);
									*noisy_final_cloud += aligned_full;
								}
								else {
									cout << "ICP did not converge for view " << view_id << ". Using noisy cloud without alignment." << endl;
									*noisy_final_cloud += *noisy_cloud;
								}
							}
							else {
								cout << "ICP OFF, directly use noisy cloud for view " << view_id << endl;
								*noisy_final_cloud += *noisy_cloud;
							}
						}
					}

					// --- standardize sampling density for evaluation (does NOT change the map, only evaluation) ---
					pcl::VoxelGrid<pcl::PointXYZRGB> vg_eval;
					double leaf = share_data->ground_truth_resolution;  // or fixed 0.01 for 1cm
					vg_eval.setLeafSize(leaf, leaf, leaf);

					// downsample pred
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr pred_ds(new pcl::PointCloud<pcl::PointXYZRGB>);
					vg_eval.setInputCloud(noisy_final_cloud);
					vg_eval.filter(*pred_ds);

					// downsample GT
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr gt_ds(new pcl::PointCloud<pcl::PointXYZRGB>);
					vg_eval.setInputCloud(share_data->cloud_ground_truth);
					vg_eval.filter(*gt_ds);

					// build kd-trees on ds clouds
					pcl::KdTreeFLANN<pcl::PointXYZRGB> pred_kdtree;
					pred_kdtree.setInputCloud(pred_ds);

					pcl::KdTreeFLANN<pcl::PointXYZRGB> gt_kdtree_eval;
					gt_kdtree_eval.setInputCloud(gt_ds);

					// thresholds (meters)
					vector<double> taus = { 1.0, 2.0 };
					vector<double> precisions, recalls, f1s;

					for (double tau_id : taus) {
						double tau = tau_id * share_data->ground_truth_resolution;
						const double tau2_sq = tau * tau;

						// Precision: pred_ds -> gt_ds
						int tp_pred = 0;
						for (const auto& q : pred_ds->points) {
							std::vector<int> nn_idx(1);
							std::vector<float> nn_sqdist(1);
							if (gt_kdtree_eval.nearestKSearch(q, 1, nn_idx, nn_sqdist) > 0) {
								if (nn_sqdist[0] <= tau2_sq) tp_pred++;
							}
						}
						int pred_n = static_cast<int>(pred_ds->size());
						double precision = (pred_n > 0) ? double(tp_pred) / double(pred_n) : 0.0;

						// Recall: gt_ds -> pred_ds
						int tp_gt = 0;
						for (const auto& p : gt_ds->points) {
							std::vector<int> nn_idx(1);
							std::vector<float> nn_sqdist(1);
							if (pred_kdtree.nearestKSearch(p, 1, nn_idx, nn_sqdist) > 0) {
								if (nn_sqdist[0] <= tau2_sq) tp_gt++;
							}
						}
						int gt_n = static_cast<int>(gt_ds->size());
						double recall = (gt_n > 0) ? double(tp_gt) / double(gt_n) : 0.0;

						double f1 = (precision + recall > 0.0) ? (2.0 * precision * recall) / (precision + recall) : 0.0;

						precisions.push_back(precision);
						recalls.push_back(recall);
						f1s.push_back(f1);
					}

					double chamfer_distance = chamferDistance(noisy_final_cloud, share_data->cloud_ground_truth, share_data->ground_truth_resolution, share_data->ray_max_dis);
					cout << "Noise level " << noise_level_idx << (icp_on ? " with ICP" : " without ICP") << " chamfer distance " << chamfer_distance << endl;
					for (size_t tau_id = 0; tau_id < taus.size(); ++tau_id) {
						cout << "tau " << taus[tau_id] << ": precision = " << precisions[tau_id] << ", recall = " << recalls[tau_id] << ", F1 = " << f1s[tau_id] << endl;
					}
					// 保存结果
					share_data->access_directory(share_data->save_path + "/pose_noise/");
					ofstream fout_pose_noise(share_data->save_path + "/pose_noise/" + "sigma_t_" + to_string(sigma_t) + "_sigma_r_" + to_string(sigma_r * 180.0 / M_PI) + (icp_on ? "_with_icp" : "_without_icp") + ".txt");
					fout_pose_noise << chamfer_distance << '\n';
					for (size_t tau_id = 0; tau_id < taus.size(); ++tau_id) {
						fout_pose_noise << precisions[tau_id] << '\t' << recalls[tau_id] << '\t' << f1s[tau_id] << '\t';
					}

					//可选可视化
					if (share_data->show) {
						pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Surface"));
						viewer->setBackgroundColor(255, 255, 255);
						viewer->addPointCloud<pcl::PointXYZRGB>(noisy_final_cloud, "noisy_final_cloud");
						viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "noisy_final_cloud");
						show_cloud(viewer);
					}

					// end of icp options
				}
				// end of noise level options
			}
			//end of for each model
		}
		//end of TestPoseNoise
	}
	cout << "System over." << endl;
	return 0;
}

//gap1
/*
room1_20250918
room1_20250922
room1_20250925
room1_20250929
room4_20250918
room4_20250922
room4_20250925
room4_20250929
*/

//gap2
/*
room1_20250922
room1_20250925
room1_20250929
room4_20250922
room4_20250925
room4_20250929
*/

//gap3
/*
room1_20250925
room1_20250929
room4_20250925
room4_20250929
*/

//gap4
/*
room1_20250929
room4_20250929
*/