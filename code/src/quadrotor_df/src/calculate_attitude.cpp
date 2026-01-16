#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

// 定义重力加速度
const double g = 9.81;

// 计算给定时间t的位置
Vector3d getPosition(double t) {
    double sin_t = std::sin(t);
    double cos_t = std::cos(t);
    double sin2_t = sin_t * sin_t;
    double denom = 1.0 + sin2_t;
    
    Vector3d pos;
    pos(0) = 10.0 * cos_t / denom;  // x
    pos(1) = 10.0 * sin_t * cos_t / denom;  // y
    pos(2) = 10.0;  // z
    
    return pos;
}

// 计算速度（位置的一阶导数）
Vector3d getVelocity(double t) {
    double sin_t = std::sin(t);
    double cos_t = std::cos(t);
    double sin2_t = sin_t * sin_t;
    double denom = 1.0 + sin2_t;
    double denom2 = denom * denom;
    
    Vector3d vel;
    // dx/dt = d/dt[10*cos(t)/(1+sin²(t))]
    // 使用商法则: (f/g)' = (f'g - fg')/g²
    // f = 10*cos(t), f' = -10*sin(t)
    // g = 1+sin²(t), g' = 2*sin(t)*cos(t)
    double numerator_x = -10.0 * sin_t * denom - 10.0 * cos_t * 2.0 * sin_t * cos_t;
    vel(0) = numerator_x / denom2;
    
    // dy/dt = d/dt[10*sin(t)*cos(t)/(1+sin²(t))]
    // f = 10*sin(t)*cos(t), f' = 10*(cos²(t) - sin²(t)) = 10*cos(2t)
    // g = 1+sin²(t), g' = 2*sin(t)*cos(t) = sin(2t)
    double f_y = 10.0 * sin_t * cos_t;
    double f_y_prime = 10.0 * (cos_t * cos_t - sin_t * sin_t);
    double g_prime = 2.0 * sin_t * cos_t;
    double numerator_y = f_y_prime * denom - f_y * g_prime;
    vel(1) = numerator_y / denom2;
    
    // dz/dt = 0
    vel(2) = 0.0;
    
    return vel;
}

// 计算加速度（位置的二阶导数）
Vector3d getAcceleration(double t) {
    double sin_t = std::sin(t);
    double cos_t = std::cos(t);
    double sin2_t = sin_t * sin_t;
    double cos2_t = cos_t * cos_t;
    double denom = 1.0 + sin2_t;
    
    // 使用数值微分计算加速度
    double dt = 1e-6;
    Vector3d vel_plus = getVelocity(t + dt);
    Vector3d vel_minus = getVelocity(t - dt);
    Vector3d acc = (vel_plus - vel_minus) / (2.0 * dt);
    
    return acc;
}

// 将旋转矩阵转换为四元数
Quaterniond rotationMatrixToQuaternion(const Matrix3d& R) {
    Quaterniond q(R);
    
    // 确保 qw >= 0
    if (q.w() < 0) {
        q.w() = -q.w();
        q.x() = -q.x();
        q.y() = -q.y();
        q.z() = -q.z();
    }
    
    // 归一化
    q.normalize();
    
    return q;
}

// 根据微分平坦性计算姿态
Quaterniond calculateAttitude(double t) {
    Vector3d pos = getPosition(t);
    Vector3d vel = getVelocity(t);
    Vector3d acc = getAcceleration(t);
    
    // 总加速度 = 加速度 + 重力
    Vector3d acc_total = acc + Vector3d(0, 0, g);
    
    // 机体z轴（向上）与总加速度方向一致
    Vector3d z_body = acc_total.normalized();
    
    // 偏航角与速度方向对齐
    // 在水平面内，x轴（前方）指向速度方向的投影
    Vector3d vel_horizontal(vel(0), vel(1), 0);
    
    if (vel_horizontal.norm() < 1e-6) {
        // 如果水平速度为0，使用默认方向
        vel_horizontal = Vector3d(1, 0, 0);
    }
    
    // x轴候选方向（在水平面内）
    Vector3d x_candidate = vel_horizontal.normalized();
    
    // y轴 = z轴 × x轴候选
    Vector3d y_body = z_body.cross(x_candidate);
    
    if (y_body.norm() < 1e-6) {
        // 如果z轴和x候选方向平行，重新选择
        x_candidate = Vector3d(1, 0, 0);
        if (std::abs(z_body.dot(x_candidate)) > 0.9) {
            x_candidate = Vector3d(0, 1, 0);
        }
        y_body = z_body.cross(x_candidate);
    }
    
    y_body.normalize();
    
    // x轴 = y轴 × z轴
    Vector3d x_body = y_body.cross(z_body);
    x_body.normalize();
    
    // 构造旋转矩阵 [x_body, y_body, z_body]
    Matrix3d R;
    R.col(0) = x_body;
    R.col(1) = y_body;
    R.col(2) = z_body;
    
    // 转换为四元数
    return rotationMatrixToQuaternion(R);
}

int main(int argc, char** argv) {
    std::cout << "计算四旋翼微分平坦性姿态..." << std::endl;
    
    // 时间范围和步长
    double t_start = 0.0;
    double t_end = 2.0 * M_PI;
    double dt = 0.02;
    
    // 存储结果
    std::vector<std::pair<double, Quaterniond>> results;
    
    // 计算每个时刻的姿态
    for (double t = t_start; t <= t_end + 1e-6; t += dt) {
        Quaterniond q = calculateAttitude(t);
        results.push_back(std::make_pair(t, q));
    }
    
    // 输出到CSV文件
    std::string output_path = "/home/bonnie/File/noetic_ws/MRPC-2025-homework/solutions/df_quaternion.csv";
    std::ofstream outfile(output_path);
    
    if (!outfile.is_open()) {
        std::cerr << "无法打开输出文件: " << output_path << std::endl;
        return 1;
    }
    
    // 写入表头
    outfile << "t,x,y,z,w" << std::endl;
    
    // 写入数据
    for (const auto& result : results) {
        double t = result.first;
        const Quaterniond& q = result.second;
        
        outfile << std::fixed << std::setprecision(2) << t << ","
                << std::fixed << std::setprecision(7) << q.x() << ","
                << std::fixed << std::setprecision(7) << q.y() << ","
                << std::fixed << std::setprecision(7) << q.z() << ","
                << std::fixed << std::setprecision(7) << q.w() << std::endl;
    }
    
    outfile.close();
    
    std::cout << "计算完成！结果已保存到: " << output_path << std::endl;
    std::cout << "共计算了 " << results.size() << " 个时间点" << std::endl;
    
    return 0;
}
