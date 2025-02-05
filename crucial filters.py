import pandas as pd


def filter_rows_by_keywords(file_path_1, sheet_name_1, file_path_2, sheet_name_2):
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)
    keywords = df2.iloc[:, 0].tolist()
    filtered_df = df1[df1.iloc[:, 0].isin(keywords)]
    return filtered_df


# 使用示例
file_path_1 = r'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\新CG\cirrhosis.csv'  # 第一个Excel文件路径
sheet_name_1 = 'Sheet1'
file_path_2 = r'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\新CG\table2.csv'  # 第二个Excel文件路径
sheet_name_2 = 'Sheet1'

filtered_rows = filter_rows_by_keywords(file_path_1, sheet_name_1, file_path_2, sheet_name_2)

# 打印筛选后的行
print(filtered_rows)

# 如果需要将筛选结果保存到新的Excel文件
filtered_rows.to_csv("crucial_germs_cirrhosis.csv")
