import random

from cached_env.cached_task import Task
from cached_env.cached_vehicles import Vehicles

# 生成车辆所用参数
RSU_coverage = 500  # RSU覆盖范围（单位：米）
road_length = 1000  # 道路长度（单位：米）
num_lanes = 2  # 车道数
lane_width = 3  # 车道宽度（单位：米）
vehicle_num_lane = 10  # 单个车道的车辆数 5-15
vehicle_num = num_lanes * vehicle_num_lane  # 车辆数 20
rsu_num = 3  # RSU的数量
min_vehicle_speed = 10  # 最小车速（单位：米/秒）
max_vehicle_speed = 20  # 最大车速（单位：米/秒）

# 数据传输速率
# rsu的数据传输速率，每一段不同，1000m道路分成7段，每段142.85 142.85/16.6
R2V_rates = [1, 2, 5.5, 11, 5.5, 2, 1]  # Mbps RSU和车辆之间的传输速率 [1, 2, 5.5, 11, 5.5, 2, 1]
R2R_rate = 20  # rsu之间的传输速率 Mbps
R2C_rate = 5  # 云到RSU或车辆的传输速率 Mbps 初始5
V2V_rate = 10  # v2v直接的传输速率 Mbps
F_MEC = 15  # MEC服务器计算能力 Ghz 初始15
F_V = 2  # 车辆计算能力 Ghz 初始 2

content_num = 30  # 内容数量 初始 30
content_types = [1, 2, 3, 4, 5, 6]  # 内容类型

# MEC服务器和车辆的存储容量
S_MEC = 60  # MB  初始60
S_V = 30  # MB 初始30


# 动作空间
"""
卸载决策 DQN
[0,1]
内容访问方式和缓存决策 SAC
[[0,0],[0,1],[1,0],[1,1]]
[0,      1,    2,    3]
"""
action_space_offloading = [0, 1]
action_space_cached = [0, 1, 2, 3]

punish = -4  # 惩罚值


class VecCachedEnv:
    def __init__(self, num_T, T):
        """
        :type num_T: 时隙个数
        :type T:一个时隙大小
        """
        super(VecCachedEnv, self).__init__()
        # 初始化车辆类
        self.vehicle = Vehicles(vehicle_num_lane, road_length, num_lanes, lane_width, min_vehicle_speed
                                , max_vehicle_speed)

        self.vehicle_num = vehicle_num  # 车辆数
        self.vehicles = []
        # 初始化任务
        self.task = Task(RSU_coverage)
        self.task_generate_times = []
        # 时隙
        self.num_T = num_T  # 时隙个数
        self.T = T  # 一个时隙大小

        self.state_space_offloading = []  # 卸载决策算法的状态空间
        self.state_space_cached = []  # 内容缓存算法的状态空间
        self.state_space_cached_conv = []  # 内容缓存算法的状态空间 conv
        self.action_space_offloading = action_space_offloading  # 动作空间,卸载决策
        self.action_space_cached = action_space_cached  # 动作空间,缓存决策
        self.task_messages = []  # 临时存储任务信息
        self.content_messages = []
        self.vehicle_count = 0  # TV个数计数

        self.S_MECs = []  # 所有MEC服务器的存储容量
        self.S_Vs = []  # 存储所有车辆的当前存储容量
        self.cached = []  # 存储所有内容的缓存情况 0：未缓存 1：已缓存
        self.node_content_ids = []  # 每一个节点上的存储的内容的id

        self.content_messages = self.task.content_message(content_num, content_types)
        self.con_num = content_num
        self.F_MEC = F_MEC

    """
    状态空间
    卸载决策算法状态
    s_o = {任务信息, 车辆计算能力, MEC服务器计算能力,c_R1,c_R2,c_R3(mec服务器上的缓存情况)}
        [   0           1           2               3]
    [任务大小,内容id,内容大小,内容流行度,最大可容忍时间, 内容类型]
    [0,         1,      2,      3,          4]
    s_gc = {任务信息, 车辆存储能力, MEC服务器存储能力, 存储情况}
    """

    def reset(self):
        self.state_space_offloading.clear()  # 重置卸载算法状态空间
        self.state_space_cached.clear()  # 重置缓存算法状态空间
        self.state_space_cached_conv.clear()  # 重置缓存算法状态空间 conv
        self.task.task_queue_mec.clear()  # 重置在MEC队列
        self.task.queue_time = 0  # 重置MEC队列时延
        self.task_generate_times = self.task.task_generate_time(
            vehicle_num * num_lanes * self.num_T, self.num_T, self.T).tolist()  # 任务生成时间
        self.vehicle.vehicles.clear()  # 重置车辆集合
        self.vehicles = self.vehicle.initialization_vehicle()  # 初始化车辆
        self.vehicle_count = 0
        self.S_MECs = [S_MEC for _ in range(rsu_num)]  # 重置MEC服务器存储容量

        # 直接初始化所有车辆的存储容量
        self.S_Vs = [S_V for _ in range(vehicle_num)]

        # 直接初始化所有节点的内容
        self.node_content_ids = [[] for _ in range(vehicle_num + rsu_num)]

        self.cached.clear()  # 重置内容缓存情况
        for i in range(content_num):
            cached_temp = []  # 创建新的列表对象 不能使用cached_temp.clear()
            for j in range(vehicle_num + rsu_num):
                cached_temp.append(0)
            self.cached.append(cached_temp)

        # 构建状态空间s_o
        # 生成任务内容
        # [[content_id, content_size, content_probability, content_type],[],...]
        # 选取一个任务内容[content_id, content_size, content_probability]
        content_message = self.task.select_content(self.content_messages)
        self.task_messages = self.task.task_generate(content_message[0], content_message[1],
                                                     content_message[2], content_message[3],
                                                     )
        self.state_space_offloading.append(self.task_messages)  # 0
        self.state_space_offloading.append([F_V])  # 1
        self.state_space_offloading.append([F_MEC])  # 2
        self.state_space_offloading.append([0, 0, 0])  # 3

        # 将任务内容按流行度从高到低排序，假设row[2]为流行度
        sorted_content_messages = sorted(self.content_messages, key=lambda row: row[2], reverse=True)
        # 将任务内容存储到当前RSU
        for row in sorted_content_messages:
            if self.S_MECs[0] >= row[1]:
                self.cached[row[0]][vehicle_num] = 1  # 更改内容缓存情况
                self.node_content_ids[vehicle_num].append(row[0])  # 更改节点缓存情况
                self.S_MECs[0] -= row[1]  # 更新剩余存储容量

        # 相邻RSU随机存储
        for rsu_index in range(1, rsu_num):
            random_integers = random.sample(range(0, content_num), content_num)
            for row in random_integers:
                if self.S_MECs[rsu_index] >= self.content_messages[row][1]:
                    self.cached[row][vehicle_num + rsu_index] = 1  # 更改内容缓存情况
                    self.node_content_ids[vehicle_num + rsu_index].append(row)  # 更改节点缓存情况
                    self.S_MECs[rsu_index] -= self.content_messages[row][1]  # 更新剩余存储容量
        # 构建状态空间s_gc
        """
        s_gc = {任务信息, 车辆存储能力, MEC服务器存储能力, 请求的任务内容的存储情况}
                [0,         1,          2,               3]
                [任务大小,内容id,内容大小,内容流行度,最大可容忍时间,内容类型]
                []                                      [[],[],[]]
        """
        self.state_space_cached.append(self.task_messages)  # 0
        self.state_space_cached.append([S_V])  # 1
        self.state_space_cached.append([S_MEC])  # 2
        self.state_space_cached.append(self.cached[self.task_messages[1]])  # 3

        state_space_offloading = sum(self.state_space_offloading, [])
        state_space_cached = sum(self.state_space_cached, [])
        state_space_cached_conv = self.task.construct_similarity_matrix(self.content_messages,
                                                                        content_num)
        return state_space_offloading, state_space_cached, state_space_cached_conv

    """
    # 动作空间
    action = [卸载决策,内容访问方式和缓存决策]
    卸载决策 DQN
    [0,1]  0:本地计算 1:卸载计算
    内容访问方式和缓存决策 PPO
    [[0,0],[0,1],[1,0],[1,1]]  1:从MEC服务器获取内容 0:从周围车辆获取内容
    [0,      1,    2,    3]
    """

    # action[action_off, action_cac]
    def step(self, action):
        reward = 0
        done = False
        s1 = self.state_space_offloading
        s1_ = s1  # 存储下一个状态
        s2 = self.state_space_cached
        s2_ = s2  # 存储下一个状态
        a_j_o = action[0]  # 卸载决策
        action_env = [[0, 0], [0, 1], [1, 0], [1, 1]]
        if a_j_o == 0 and self.cached[self.task_messages[1]][self.vehicle_count] == 0:
            a_j_g = action_env[action[1]][0]  # 内容访问方式决策
            a_j_c = action_env[action[1]][1]  # 缓存决策
        else:
            a_j_g = -1
            a_j_c = -1
        # [position_x, position_y, speed]
        # 查找车辆在哪个区域
        k = (int)(self.vehicles[self.vehicle_count][0] / (road_length / len(R2V_rates)))
        if k >= len(R2V_rates):
            k = k - 1
        R2V_rate = R2V_rates[k]
        task_size = self.task_messages[0]  # 任务大小
        content_id = self.task_messages[1]  # 内容id
        content_size = self.task_messages[2]  # 内容大小
        content_popularity = self.task_messages[3]  # 内容流行度
        max_time = self.task_messages[4]  # 最大可容忍时延
        cached_list = self.cached[content_id]
        # 计算时延
        time_total, isHit, isLocalCompute = self.task.cal_time(a_j_o, a_j_g, F_V, F_MEC, task_size,
                                                               R2V_rate, R2R_rate, R2C_rate, V2V_rate, content_size,
                                                               self.task_generate_times[0], cached_list,
                                                               self.vehicle_count,
                                                               self.vehicles)
        reward = -time_total
        isSuccess = 0
        if time_total <= max_time:
            isSuccess = 1
        if a_j_c == 1:  # 缓存内容
            if self.S_Vs[self.vehicle_count] >= content_size:  # 容量充足
                self.cached[content_id][self.vehicle_count] = 1  # 更改内容缓存情况
                self.node_content_ids[self.vehicle_count].append(content_id)  # 添加车辆上存储的内容的id
                self.S_Vs[self.vehicle_count] = self.S_Vs[self.vehicle_count] - content_size  # 更新剩余存储容量
            else:
                tempPro = 1
                min_id = 0
                # 获取流行度最低的内容id
                for index in range(len(self.node_content_ids[self.vehicle_count])):
                    temp_con_id = self.node_content_ids[self.vehicle_count][index]
                    if self.content_messages[temp_con_id][2] < tempPro:
                        tempPro = self.content_messages[temp_con_id][2]
                        min_id = temp_con_id
                # 移除内容
                self.node_content_ids[self.vehicle_count].remove(min_id)  # 移除流行度最低的内容
                self.cached[min_id][self.vehicle_count] = 0  # 更改内容缓存情况
                self.S_Vs[self.vehicle_count] += self.content_messages[min_id][1]  # 更新容量
                # 存储内容
                self.node_content_ids[self.vehicle_count].append(content_id)  # 添加车辆上存储的内容的id
                self.cached[content_id][self.vehicle_count] = 1  # 更改内容缓存情况
                self.S_Vs[self.vehicle_count] -= content_size  # 更新剩余存储容量

        self.vehicle_count += 1
        # 下一个时隙
        if self.vehicle_count >= vehicle_num:
            self.vehicle_count = 0
            self.vehicles = self.vehicle.vehicles_move(self.T)  # 车辆移动
            done = True
        # 更新任务到达时间
        self.task_generate_times.remove(self.task_generate_times[0])
        # 更新任务信息
        content_message = self.task.select_content(self.content_messages)
        self.task_messages = self.task.task_generate(content_message[0], content_message[1],
                                                     content_message[2], content_message[3],
                                                     )
        # 更新状态s_o
        s1_[0] = self.task_messages  # 0
        # 得到某一项内容的缓存情况
        content_cached = self.cached[content_message[0]]
        # 得到某一项内容在rsus上的缓存情况
        content_rsu = content_cached[vehicle_num:vehicle_num + rsu_num]
        s1_[3] = content_rsu  # 3
        # 更新状态s_gc
        s2_[0] = self.task_messages  # 0
        s2_[3] = self.cached[self.task_messages[1]]  # 3

        # 更新环境状态
        self.state_space_offloading = s1_
        self.state_space_cached = s2_

        s1_ = sum(s1_, [])
        s2_ = sum(s2_, [])
        next_state_conv = self.task.construct_similarity_matrix(self.content_messages,
                                                                content_num)
        return s1_, s2_, next_state_conv, reward, done, time_total, isHit, isLocalCompute, isSuccess


