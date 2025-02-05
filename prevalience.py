import pandas as pd

# 读取OTU表
otu_table1 = pd.read_excel(r'.\data1\crucial germs cirrhosis.xlsx', header=None)


def calculate_prevalence(otu_name, otu_table):
    # 筛选特定OTU的数据
    status_row = otu_table.iloc[0, 1:]  # 样本状态（跳过第一列）
    otu_data = otu_table.iloc[1:, 1:]  # OTU丰度数据（跳过第一行和第一列）
    otu_data.index = otu_table.iloc[1:, 0]  # 将第一列设置为菌名索引
    if otu_name not in otu_data.index:
        raise ValueError(f"OTU名称 '{otu_name}' 不在OTU表中。")

        # OTU 的存在性（丰度大于 0 表示存在）
    otu_presence = otu_data.loc[otu_name] > 0.001

    # 计算患病组和健康组的流行度
    disease_prevalence = otu_presence[status_row == 't2d'].mean()
    healthy_prevalence = otu_presence[status_row == 'n'].mean()

    return disease_prevalence, healthy_prevalence


def calculate_prevalence_differences(otu_table):
    # 提取样本状态和物种数据
    status_row = otu_table.iloc[0, 1:]  # 样本状态
    otu_data = otu_table.iloc[1:, 1:]  # 物种丰度数据
    # print(otu_data)
    otu_data.index = otu_table.iloc[1:, 0]  # 将第一列设置为物种名称索引
    otu_data = otu_data.apply(pd.to_numeric, errors='coerce')
    # 计算物种的存在性（丰度大于阈值）
    otu_presence = otu_data > 0
    # 分组计算流行度
    health_status = status_row == 'n'
    disease_status = status_row == 'cirrhosis'
    total_health_samples = health_status.sum()  # 健康样本总数
    total_disease_samples = disease_status.sum()  # 非健康样本总数
    print(total_health_samples)
    weighted_prevalence_h = (otu_presence.loc[:, health_status] * otu_data.loc[:, health_status]).sum(
        axis=1) / total_health_samples
    weighted_prevalence_nh = (otu_presence.loc[:, disease_status] * otu_data.loc[:, disease_status]).sum(
        axis=1) /  total_disease_samples

    # 计算流行度差异
    prevalence_difference = abs(weighted_prevalence_h - weighted_prevalence_nh)

    # 构建流行度表
    prevalence_df = pd.DataFrame({
        'Health_Prevalence': weighted_prevalence_h,
        'NonHealth_Prevalence': weighted_prevalence_nh,
        'Prevalence_Difference': prevalence_difference
    })

    return prevalence_df


# OTU的流行度
otu_name1 = 'Clostridium_nexile'  # d:0.39，n:0.43
otu_name2 = 'Clostridium_citroniae'  # d:0.78,n:0.60
otu_name3 = 'Clostridium_bolteae'  # d:0.91, n:0.82
otu_name4 = 'Clostridium_hathewayi'  # d:0.82, n:0.64
otu_name5 = 'Clostridium_asparagiforme'  # d:0.64, n:0.54
otu_name6 = 'Erysipelotrichaceae_bacterium_2_2_44A'  # d:0.39,n:0.17
otu_name7 = 'Lachnospiraceae_bacterium_3_1_57FAA_CT1'  # d:0.40, n:0.24
otu_name8 = 'Lachnospiraceae_bacterium_1_4_56FAA'  # d:0.42, n:0.35
# disease_prevalence, healthy_prevalence = calculate_prevalence(otu_name8, otu_table)
# print(f"流行度 - 患病组: {disease_prevalence:.2f}, 非患病组: {healthy_prevalence:.2f}")
otu_table1 = pd.DataFrame(otu_table1)
# 计算流行度及其差异
prevalence_df = calculate_prevalence_differences(otu_table1)
prevalence_df.to_csv('prevalence_cirrhosis.csv')
# 打印结果
print(prevalence_df)
