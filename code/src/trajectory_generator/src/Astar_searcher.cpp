#include "Astar_searcher.h"

using namespace std;
using namespace Eigen;

void Astarpath::begin_grid_map(double _resolution, Vector3d global_xyz_l,
                                  Vector3d global_xyz_u, int max_x_id,
                                  int max_y_id, int max_z_id) {
  gl_xl = global_xyz_l(0);
  gl_yl = global_xyz_l(1);
  gl_zl = global_xyz_l(2);

  gl_xu = global_xyz_u(0);
  gl_yu = global_xyz_u(1);
  gl_zu = global_xyz_u(2);

  GRID_X_SIZE = max_x_id;
  GRID_Y_SIZE = max_y_id;
  GRID_Z_SIZE = max_z_id;
  GLYZ_SIZE = GRID_Y_SIZE * GRID_Z_SIZE;
  GLXYZ_SIZE = GRID_X_SIZE * GLYZ_SIZE;

  resolution = _resolution;
  inv_resolution = 1.0 / _resolution;

  // 初始化 Z 轴权重参数（lambda > 1 使得 Z 轴移动代价更高）
  z_weight_lambda = 2.0;  // 可根据需要调整

  data = new uint8_t[GLXYZ_SIZE];
  memset(data, 0, GLXYZ_SIZE * sizeof(uint8_t));

  data_raw = new uint8_t[GLXYZ_SIZE];
  memset(data_raw, 0, GLXYZ_SIZE * sizeof(uint8_t));

  Map_Node = new MappingNodePtr **[GRID_X_SIZE];
  for (int i = 0; i < GRID_X_SIZE; i++) {
    Map_Node[i] = new MappingNodePtr *[GRID_Y_SIZE];
    for (int j = 0; j < GRID_Y_SIZE; j++) {
      Map_Node[i][j] = new MappingNodePtr[GRID_Z_SIZE];
      for (int k = 0; k < GRID_Z_SIZE; k++) {
        Vector3i tmpIdx(i, j, k);
        Vector3d pos = gridIndex2coord(tmpIdx);
        Map_Node[i][j][k] = new MappingNode(tmpIdx, pos);
      }
    }
  }
}

void Astarpath::resetGrid(MappingNodePtr ptr) {
  ptr->id = 0;
  ptr->Father = NULL;
  ptr->g_score = inf;
  ptr->f_score = inf;
}

void Astarpath::resetUsedGrids() {
  for (int i = 0; i < GRID_X_SIZE; i++)
    for (int j = 0; j < GRID_Y_SIZE; j++)
      for (int k = 0; k < GRID_Z_SIZE; k++)
        resetGrid(Map_Node[i][j][k]);
}

void Astarpath::set_barrier(const double coord_x, const double coord_y,
                             const double coord_z) {
  if (coord_x < gl_xl || coord_y < gl_yl || coord_z < gl_zl ||
      coord_x >= gl_xu || coord_y >= gl_yu || coord_z >= gl_zu)
    return;

  int idx_x = static_cast<int>((coord_x - gl_xl) * inv_resolution);
  int idx_y = static_cast<int>((coord_y - gl_yl) * inv_resolution);
  int idx_z = static_cast<int>((coord_z - gl_zl) * inv_resolution);

  data_raw[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] = 1;

  if (idx_x == 0 || idx_y == 0 || idx_z == GRID_Z_SIZE || idx_x == GRID_X_SIZE ||
      idx_y == GRID_Y_SIZE)
    data[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] = 1;
  else {
    data[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x + 1) * GLYZ_SIZE + (idx_y + 1) * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x + 1) * GLYZ_SIZE + (idx_y - 1) * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x - 1) * GLYZ_SIZE + (idx_y + 1) * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x - 1) * GLYZ_SIZE + (idx_y - 1) * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x)*GLYZ_SIZE + (idx_y + 1) * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x)*GLYZ_SIZE + (idx_y - 1) * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x + 1) * GLYZ_SIZE + (idx_y)*GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x - 1) * GLYZ_SIZE + (idx_y)*GRID_Z_SIZE + idx_z] = 1;
  }
}

vector<Vector3d> Astarpath::getVisitedNodes() {
  vector<Vector3d> visited_nodes;
  for (int i = 0; i < GRID_X_SIZE; i++)
    for (int j = 0; j < GRID_Y_SIZE; j++)
      for (int k = 0; k < GRID_Z_SIZE; k++) {
        // if(Map_Node[i][j][k]->id != 0) // visualize all nodes in open and
        // close list
        if (Map_Node[i][j][k]->id ==
            -1) // visualize nodes in close list only
          visited_nodes.push_back(Map_Node[i][j][k]->coord);
      }

  ROS_WARN("visited_nodes size : %d", visited_nodes.size());
  return visited_nodes;
}

Vector3d Astarpath::gridIndex2coord(const Vector3i &index) {
  Vector3d pt;

  pt(0) = ((double)index(0) + 0.5) * resolution + gl_xl;
  pt(1) = ((double)index(1) + 0.5) * resolution + gl_yl;
  pt(2) = ((double)index(2) + 0.5) * resolution + gl_zl;

  return pt;
}

Vector3i Astarpath::coord2gridIndex(const Vector3d &pt) {
  Vector3i idx;
  idx << min(max(int((pt(0) - gl_xl) * inv_resolution), 0), GRID_X_SIZE - 1),
      min(max(int((pt(1) - gl_yl) * inv_resolution), 0), GRID_Y_SIZE - 1),
      min(max(int((pt(2) - gl_zl) * inv_resolution), 0), GRID_Z_SIZE - 1);

  return idx;
}

Vector3i Astarpath::c2i(const Vector3d &pt) {
  Vector3i idx;
  idx << min(max(int((pt(0) - gl_xl) * inv_resolution), 0), GRID_X_SIZE - 1),
      min(max(int((pt(1) - gl_yl) * inv_resolution), 0), GRID_Y_SIZE - 1),
      min(max(int((pt(2) - gl_zl) * inv_resolution), 0), GRID_Z_SIZE - 1);

  return idx;
}

Eigen::Vector3d Astarpath::coordRounding(const Eigen::Vector3d &coord) {
  return gridIndex2coord(coord2gridIndex(coord));
}

inline bool Astarpath::isOccupied(const Eigen::Vector3i &index) const {
  return isOccupied(index(0), index(1), index(2));
}

bool Astarpath::is_occupy(const Eigen::Vector3i &index) {
  return isOccupied(index(0), index(1), index(2));
}

bool Astarpath::is_occupy_raw(const Eigen::Vector3i &index) {
  int idx_x = index(0);
  int idx_y = index(1);
  int idx_z = index(2);
  return (idx_x >= 0 && idx_x < GRID_X_SIZE && idx_y >= 0 && idx_y < GRID_Y_SIZE &&
          idx_z >= 0 && idx_z < GRID_Z_SIZE &&
          (data_raw[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] == 1));
}

inline bool Astarpath::isFree(const Eigen::Vector3i &index) const {
  return isFree(index(0), index(1), index(2));
}

inline bool Astarpath::isOccupied(const int &idx_x, const int &idx_y,
                                        const int &idx_z) const {
  return (idx_x >= 0 && idx_x < GRID_X_SIZE && idx_y >= 0 && idx_y < GRID_Y_SIZE &&
          idx_z >= 0 && idx_z < GRID_Z_SIZE &&
          (data[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] == 1));
}

inline bool Astarpath::isFree(const int &idx_x, const int &idx_y,
                                    const int &idx_z) const {
  return (idx_x >= 0 && idx_x < GRID_X_SIZE && idx_y >= 0 && idx_y < GRID_Y_SIZE &&
          idx_z >= 0 && idx_z < GRID_Z_SIZE &&
          (data[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] < 1));
}

inline void Astarpath::AstarGetSucc(MappingNodePtr currentPtr,
                                          vector<MappingNodePtr> &neighborPtrSets,
                                          vector<double> &edgeCostSets) {
  neighborPtrSets.clear();
  edgeCostSets.clear();
  Vector3i Idx_neighbor;
  for (int dx = -1; dx < 2; dx++) {
    for (int dy = -1; dy < 2; dy++) {
      for (int dz = -1; dz < 2; dz++) {

        if (dx == 0 && dy == 0 && dz == 0)
          continue;

        Idx_neighbor(0) = (currentPtr->index)(0) + dx;
        Idx_neighbor(1) = (currentPtr->index)(1) + dy;
        Idx_neighbor(2) = (currentPtr->index)(2) + dz;

        if (Idx_neighbor(0) < 0 || Idx_neighbor(0) >= GRID_X_SIZE ||
            Idx_neighbor(1) < 0 || Idx_neighbor(1) >= GRID_Y_SIZE ||
            Idx_neighbor(2) < 0 || Idx_neighbor(2) >= GRID_Z_SIZE) {
          continue;
        }

        neighborPtrSets.push_back(
            Map_Node[Idx_neighbor(0)][Idx_neighbor(1)][Idx_neighbor(2)]);
        
        // 修改边权重：对 Z 轴分量进行加权
        double weighted_dz = z_weight_lambda * dz;
        edgeCostSets.push_back(sqrt(dx * dx + dy * dy + weighted_dz * weighted_dz));
      }
    }
  }
}

double Astarpath::getHeu(MappingNodePtr node1, MappingNodePtr node2) {
  
  // 使用欧几里得距离作为启发式函数
  double dx = node1->coord(0) - node2->coord(0);
  double dy = node1->coord(1) - node2->coord(1);
  double dz = node1->coord(2) - node2->coord(2);
  
  // 修改启发式：对 Z 轴距离差进行加权
  double weighted_dz = z_weight_lambda * dz;
  double heu = sqrt(dx*dx + dy*dy + weighted_dz*weighted_dz);
  
  // Tiebreaker: 略微放大启发式值，打破平局
  double tie_breaker = 1.0 + 1.0 / 10000.0;
  heu = heu * tie_breaker;
  
  return heu;
}


bool Astarpath::AstarSearch(Vector3d start_pt, Vector3d end_pt) {
  ros::Time time_1 = ros::Time::now();

  // start_point 和 end_point 索引
  Vector3i start_idx = coord2gridIndex(start_pt);
  Vector3i end_idx = coord2gridIndex(end_pt);
  goalIdx = end_idx;

  //start_point 和 end_point 的位置
  start_pt = gridIndex2coord(start_idx);
  end_pt = gridIndex2coord(end_idx);

  // 初始化 struct MappingNode 的指针，分别代表 start node 和 goal node
  // 
  MappingNodePtr startPtr = new MappingNode(start_idx, start_pt);
  MappingNodePtr endPtr = new MappingNode(end_idx, end_pt);

  // Openset 是通过 STL 库中的 multimap 实现的open_list
  Openset.clear();
  // currentPtr 表示 open_list 中 f（n） 最低的节点
  MappingNodePtr currentPtr = NULL;
  MappingNodePtr neighborPtr = NULL;

  // 将 Start 节点放在 Open Set 中
  startPtr->g_score = 0;
  /**
   *
   * STEP 1.1:  完成 Astarpath::getHeu
   *
   * **/
  startPtr->f_score = getHeu(startPtr, endPtr);

  

  startPtr->id = 1;
  startPtr->coord = start_pt;
  startPtr -> Father = NULL;
  Openset.insert(make_pair(startPtr->f_score, startPtr));


  double tentative_g_score;
  vector<MappingNodePtr> neighborPtrSets;
  vector<double> edgeCostSets;

  /**
   *
   * STEP 1.2:  完成循环
   *
   * **/

  while (!Openset.empty()) {
    // 1. 弹出 g+h 最小的节点
    currentPtr = Openset.begin()->second;
    Openset.erase(Openset.begin());
    currentPtr->id = -1;  // 移入 Closed List
    
    // 2. 判断是否是终点
    if (currentPtr->index == goalIdx) {
      terminatePtr = currentPtr;
      ROS_WARN("[A*] Goal reached!");
      return true;
    }
    
    // 3. 扩展当前节点，获取邻居
    AstarGetSucc(currentPtr, neighborPtrSets, edgeCostSets);
    
    for(unsigned int i=0;i<neighborPtrSets.size();i++)
    {
      
      if(neighborPtrSets[i]->id==-1)
      {
         continue;
      }
      tentative_g_score=currentPtr->g_score+edgeCostSets[i];
      neighborPtr=neighborPtrSets[i];
      if(isOccupied(neighborPtr->index))
      continue;
      if(neighborPtr->id==0)
      {
        // 4. 新节点：填写信息，加入 Open List
        neighborPtr->g_score = tentative_g_score;
        neighborPtr->f_score = tentative_g_score + getHeu(neighborPtr, endPtr);
        neighborPtr->Father = currentPtr;
        neighborPtr->id = 1;
        Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
        continue;
      }
      else if(neighborPtr->id==1)
      {
        // 已在 Open List：检查是否需要更新
        if(neighborPtr->g_score > tentative_g_score){
          neighborPtr->g_score = tentative_g_score;
          neighborPtr->Father = currentPtr;
          neighborPtr->f_score = tentative_g_score + getHeu(neighborPtr, endPtr);
          Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
        }
      continue;
      }
    }
  }

  ros::Time time_2 = ros::Time::now();
  if ((time_2 - time_1).toSec() > 0.1)
    ROS_WARN("Time consume in Astar path finding is %f",
             (time_2 - time_1).toSec());
  return false;
}


vector<Vector3d> Astarpath::getPath() {
  vector<Vector3d> path;
  vector<MappingNodePtr> front_path;
do
{
terminatePtr->coord=gridIndex2coord(terminatePtr->index);
front_path.push_back(terminatePtr);
terminatePtr=terminatePtr->Father;
}while(terminatePtr->Father!=NULL);
  /**
   *
   * STEP 1.3:  追溯找到的路径
   *
   * **/

  // 将起点也加入路径
  terminatePtr->coord = gridIndex2coord(terminatePtr->index);
  front_path.push_back(terminatePtr);
  
  // 反转路径（从终点到起点 -> 从起点到终点）
  reverse(front_path.begin(), front_path.end());
  
  // 提取坐标向量
  for(const auto& node : front_path) {
    path.push_back(node->coord);
  }

  return path;
}


// 检查线段是否与障碍物碰撞
bool Astarpath::segmentCollisionFree(const Eigen::Vector3d &start, const Eigen::Vector3d &end) {
  Eigen::Vector3d direction = end - start;
  double distance = direction.norm();
  
  if (distance < 1e-6) return true;
  
  // 采样步长（比栅格分辨率更小以确保精度）
  double step = resolution * 0.5;
  int num_steps = static_cast<int>(distance / step) + 1;
  
  Eigen::Vector3d unit_dir = direction.normalized();
  
  for (int i = 0; i <= num_steps; i++) {
    Eigen::Vector3d point = start + unit_dir * (i * step);
    Eigen::Vector3i idx = coord2gridIndex(point);
    
    if (isOccupied(idx)) {
      return false;  // 有障碍物
    }
  }
  return true;  // 无障碍物
}

std::vector<Vector3d> Astarpath::pathSimplify(const vector<Vector3d> &path,
                                               double path_resolution) {
  if (path.size() <= 2) {
    return path;
  }

  // 限制最大跳跃距离，防止多项式轨迹偏离过大
  const double MAX_SEGMENT_LENGTH = 2.0;  // 最大线段长度（米）
  const size_t MAX_LOOK_AHEAD = 15;       // 每次最多跳过的点数

  // First pass: line-of-sight pruning with distance constraint
  std::vector<Vector3d> los_path;
  los_path.reserve(path.size());
  los_path.push_back(path.front());

  size_t anchor = 0;
  while (anchor + 1 < path.size()) {
    size_t candidate = anchor + 1;
    size_t max_idx = std::min(anchor + MAX_LOOK_AHEAD, path.size() - 1);
    
    for (size_t i = anchor + 1; i <= max_idx; ++i) {
      double dist = (path[i] - path[anchor]).norm();
      // 只有在距离限制内且无碰撞时才延长
      if (dist <= MAX_SEGMENT_LENGTH && segmentCollisionFree(path[anchor], path[i])) {
        candidate = i;
      } else if (dist > MAX_SEGMENT_LENGTH) {
        // 超过最大距离，停止搜索
        break;
      } else {
        // 碰撞，停止搜索
        break;
      }
    }

    los_path.push_back(path[candidate]);
    anchor = candidate;
  }

  // Second pass: remove nearly collinear points while respecting obstacle check
  std::vector<Vector3d> resultPath;
  resultPath.reserve(los_path.size());
  resultPath.push_back(los_path.front());

  for (size_t i = 1; i + 1 < los_path.size(); ++i) {
    double d = perpendicularDistance(los_path[i], los_path[i - 1], los_path[i + 1]);
    double segment_len = (los_path[i + 1] - los_path[i - 1]).norm();
    
    // 保留条件：垂直距离大于阈值，或者跳过会碰撞，或者线段太长
    if (d > path_resolution || 
        segment_len > MAX_SEGMENT_LENGTH ||
        !segmentCollisionFree(los_path[i - 1], los_path[i + 1])) {
      resultPath.push_back(los_path[i]);
    }
  }

  resultPath.push_back(los_path.back());
  
  ROS_INFO("[Path Simplify] Original: %lu -> LOS: %lu -> Final: %lu points", 
           path.size(), los_path.size(), resultPath.size());

  return resultPath;
}

double Astarpath::perpendicularDistance(const Eigen::Vector3d point_insert,const Eigen:: Vector3d point_st,const Eigen::Vector3d point_end)
{
  Vector3d line1=point_end-point_st;
  Vector3d line2=point_insert-point_st;
  return double(line2.cross(line1).norm()/line1.norm());
}

Vector3d Astarpath::getPosPoly(MatrixXd polyCoeff, int k, double t) {
  Vector3d ret;
  int _poly_num1D = (int)polyCoeff.cols() / 3;
  for (int dim = 0; dim < 3; dim++) {
    VectorXd coeff = (polyCoeff.row(k)).segment(dim * _poly_num1D, _poly_num1D);
    VectorXd time = VectorXd::Zero(_poly_num1D);

    for (int j = 0; j < _poly_num1D; j++)
      if (j == 0)
        time(j) = 1.0;
      else
        time(j) = pow(t, j);

    ret(dim) = coeff.dot(time);
    // cout << "dim:" << dim << " coeff:" << coeff << endl;
  }

  return ret;
}


int Astarpath::safeCheck(MatrixXd polyCoeff, VectorXd time) {
  int unsafe_segment = -1; //-1 -> the whole trajectory is safe

  double delta_t=resolution/1.0;//conservative advance step size;
  double t = delta_t;
  Vector3d advancePos;
  for(int i=0;i<polyCoeff.rows();i++)
  {
    while(t<time(i)){
     advancePos=getPosPoly(polyCoeff,i,t) ;
     if(isOccupied(coord2gridIndex(advancePos))){
       unsafe_segment=i;
       break;
     }   
     t+=delta_t;
    }
    if(unsafe_segment!=-1){

      break;
    }else{
      t=delta_t;
    }
  }
  return unsafe_segment;
}

void Astarpath::resetOccupy(){
  for (int i = 0; i < GRID_X_SIZE; i++)
for (int j = 0; j < GRID_Y_SIZE; j++)
  for (int k = 0; k < GRID_Z_SIZE; k++) {
    data[i * GLYZ_SIZE + j * GRID_Z_SIZE + k] = 0;
    data_raw[i * GLYZ_SIZE + j * GRID_Z_SIZE + k] = 0;
  }
}