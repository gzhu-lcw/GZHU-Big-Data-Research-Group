import random
import numpy as np
import math
from cached_env import forcast

from sklearn.metrics.pairwise import cosine_similarity


class Task:
    def __init__(self, RSU_coverage):
        super(Task, self).__init__()
        self.task_queue_mec = []
        self.queue_time = 0
        self.RSU_coverage = RSU_coverage

    # 生成任务生成时间
    def task_generate_time(self, time_num, num_T, T):
        """
        :param time_num: 时隙总数
        :param num_T: 时隙个数
        :param T: 一个时隙大小
        :return: 生成任务时间的numpy列表
        """
        lam = 10
        # 生成泊松分布的到达时间
        arrival_times = np.random.exponential(1 / lam, time_num + 1)
        arrival_times = np.cumsum(arrival_times)

        # 将生成时间缩放到0-1秒之间
        max_time = arrival_times[-1]
        arrival_times = arrival_times / max_time

        # 缩放到0-x秒之间
        task_generate_times = 0 + arrival_times * (T * num_T - 0)
        return task_generate_times

    # 生成任务内容
    def content_message(self,content_num, content_types):
        """
        :param content_types: 应用类型 [1,2,3,4,5...]
        :param epsilon: zipf参数
        :param content_num: 内容数量
        :return:[[content_id, content_size, content_probability, content_type],[],...] 所有内容信息
        """
        content_messages = []
        probabilitys = forcast.probability_forecast(content_num)
        for n in range(1, content_num + 1):
            # 计算概率
            probability = probabilitys[n-1]
            content_type = random.choice(content_types)  # 选取类型
            if content_type <= 3:
                content_size = 3
            elif 3 < content_type <= 5:
                content_size = 4
            else:
                content_size = 5
            content_messages.append([n - 1, content_size, probability, content_type])
        return content_messages

    # 按照概率取任务内容
    def select_content(self, content_messages):
        """
        :param content_messages: 所有内容信息 [[]]
        :return: 一个内容信息[2, 4.4, 0.15711776493591284]
        """
        # 提取概率列
        probabilities = [row[2] for row in content_messages]
        # 计算累积概率分布
        cumulative_probabilities = np.cumsum(probabilities)
        # 生成一个0到1之间的随机数
        random_number = np.random.rand()
        # 根据随机数找到对应的区间
        selected_index = np.searchsorted(cumulative_probabilities, random_number)
        # 选择对应的列
        selected_content = content_messages[selected_index]
        return selected_content

    # 生成任务[任务的输入数据大小,内容id，需要的存储空间,流行度, 最大可容忍时延]
    def task_generate(self, content_id, content_size, content_probability, content_type):
        """
        :param content_type: 内容类型 （1,2,3,4,5）
        :param content_id: 内容id
        :param content_size: 内容大小
        :param content_probability: 内容流行度
        :return:[任务大小,内容id,内容大小,内容流行度,最大可容忍时间, 内容类型]
        """
        task_input_size = round(random.uniform(4, 5), 1)  # 任务数据大小 初始2-3M
        task_max_time = random.choice([1, 2, 3, 4])  # 任务最大可容忍时延
        # 任务内容e_j的流行度
        task_messages = [task_input_size, content_id,
                         content_size, content_probability, task_max_time, content_type]
        return task_messages

    # 计算排队时间
    def cal_queue_time(self, task_queue):
        """
        计算队列时间
        :param task_queue: 任务队列[[任务到达时间,执行时间]]
        :return: 队列时延
        """
        exe_sum_time = 0
        length = len(task_queue)
        if length == 0:
            return 0
        for row in range(length - 1):
            exe_sum_time += task_queue[row][1]
        queue_time = exe_sum_time - task_queue[length - 1][0]
        if queue_time < 0:
            queue_time = 0
        return queue_time

    # 计算任务执行时间
    def cal_time(self, a_j_o, a_j_g, F_V, F_MEC, task_input_size,
                 R2V_rate, R2R_rate, R2C_rate, V2V_rate, task_content_size, generate_time, cached_list, vehicle_id,
                 vehicles):
        """
        :param a_j_o: 卸载决策
        :param a_j_g: 内容访问方式决策
        :param F_V:车辆计算能力
        :param F_MEC:MEC服务器计算能力
        :param task_input_size: 任务输入数据大小
        :param R2V_rate: 数据传输速率rsu[1, 2, 5.5, 11, 5.5, 2, 1]
        1000m道路分成7段，每段142.85 142.85/16.6
        :param V2V_rate: 车辆之间数据传输速率
        :param R2C_rate: RSU和云服务器之间数据传输速率
        :param R2R_rate: RSU之间数据传输速率
        :param task_content_size: 任务内容大小
        :param generate_time: 任务生成时间
        :param cached_list: 内容的缓存情况
        :param vehicle_id: 车辆的编号
        :param vehicles: 车辆集合
        :return:
        相邻RSU的传输速率 20 M/s
        """
        global time_trans
        vehicle_num = len(vehicles)
        rsu_num = 3
        cycle = 500  # 500 cycle/byte
        isCached = 0  # 统计命中率
        isLocalCompute = 0  # 是否本地计算
        task_cycle = task_input_size * 1e6 * cycle * 1e-9  # 任务所需的cpu周期数1-4GHz
        if a_j_o == 0:  # 本地计算
            isLocalCompute = 1
            time_loc_exe = task_cycle / F_V  # 本地计算时间
            time_mec_v = task_content_size / R2V_rate  # 从MEC服务器获取内容时间
            if cached_list[vehicle_id] == 0:  # 本地没有缓存
                delta_x = -1
                if a_j_g == 1:  # 从MEC服务器获取内容
                    for i in range(rsu_num):  # 查找缓存在哪个RSU
                        if cached_list[vehicle_num + i] == 1:
                            delta_x = i
                            break
                    if delta_x != -1:  # 边缘池缓存了任务内容
                        time_trans = time_mec_v + delta_x * (task_content_size / R2R_rate)
                        isCached = 1
                    else:  # 边缘池没有缓存任务内容,从云服务器获取
                        time_trans = time_mec_v + (task_content_size / R2C_rate)
                    time_total = time_loc_exe + time_trans
                else:  # 从周围车辆获取内容
                    coverage_vehicles = self.get_coverage_vehicles(vehicle_id, vehicles)
                    if len(coverage_vehicles) != 0:  # 周围有车辆
                        flag = True   # 判断周围是否有车辆
                        for v_id in coverage_vehicles:
                            if cached_list[v_id] == 1:  # 所选车辆缓存了内容
                                time_trans = task_content_size / V2V_rate
                                isCached = 1
                                flag = False
                                break
                        if flag:  # 周围车辆上没有缓存内容, 从云服务器获取
                            time_trans = task_content_size / R2C_rate
                    else:
                        time_trans = task_content_size / R2C_rate
                    time_total = time_loc_exe + time_trans
            else:  # 本地缓存了内容
                time_total = time_loc_exe
                isCached = 1
        else:  # MEC服务器计算
            time_v_mec = task_input_size / R2V_rate  # 传输到MEC服务器的时间
            arrival_time = generate_time + time_v_mec  # 任务到达时间
            time_mec_exe = task_cycle / F_MEC  # 任务执行时间
            self.task_queue_mec.append([arrival_time, time_mec_exe])  # 将执行的任务加入MEC队列
            self.queue_time = self.cal_queue_time(self.task_queue_mec)  # 任务排队时间
            delta_x = -1
            for i in range(rsu_num):  # 查找缓存在哪个RSU
                if cached_list[vehicle_num + i] == 1:
                    delta_x = i
                    break
            if delta_x != -1:  # 边缘池缓存了任务内容
                time_trans = delta_x * (task_content_size / R2R_rate)
                # isCached = 1
            else:  # 边缘池没有缓存任务内容,从云服务器获取
                time_trans = task_content_size / R2C_rate
            time_total = time_v_mec + time_mec_exe + self.queue_time + time_trans
        return time_total, isCached, isLocalCompute

    # 得到车辆i的周围车辆
    def get_coverage_vehicles(self, vehicle_id, vehicles):
        coverage_vehicles = []
        x_i = vehicles[vehicle_id][0]
        y_i = vehicles[vehicle_id][1]
        for i in range(len(vehicles)):
            distance_i_g = math.sqrt((x_i - vehicles[i][0]) ** 2
                                     + (y_i - vehicles[i][1]) ** 2)  # 计算车辆i和其它车辆的距离
            if distance_i_g <= self.RSU_coverage / 2:  # 车辆i周围车辆
                coverage_vehicles.append(i)
        return coverage_vehicles

    # 构建任务内容之间的相似矩阵
    """
    [[content_id, content_size, content_probability, content_type, task_max_time],
    [], ...] 所有内容信息
    """

    def construct_similarity_matrix(self, content_messages, content_num):
        task_contents = {}
        for i in range(0, content_num):
            task_contents.update({'{}'.format(i + 1): np.array([content_messages[i][1],
                                                                content_messages[i][2],
                                                                content_messages[i][3]
                                                                ])})
        # 计算任务内容之间的相似性矩阵
        content_vectors = np.array(list(task_contents.values()))
        similarity_matrix = cosine_similarity(content_vectors).tolist()
        return similarity_matrix


