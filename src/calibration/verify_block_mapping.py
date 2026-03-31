#!/usr/bin/env python3
import pandas as pd

# 读取 BC081116d_specs.xlsx - 跳过元数据行，从row 9开始（Position行）
excel_file = r'BC081116d_specs.xlsx'
df = pd.read_excel(excel_file, header=9)  # row 9是真正的header（Position这一行）

print("BC081116d_specs.xlsx 数据来源 - 前20行:")
print("="*100)
print(df.head(20).to_string())

print("\n\n查找8个block的对应数据:")
print("="*100)
print(f"DataFrame columns: {list(df.columns)}\n")

blocks_we_have = ['A1', 'A8', 'D1', 'E10', 'G1', 'H2', 'H10', 'J10']

for block in blocks_we_have:
    # 查找Position等于block name的行
    try:
        match = df[df['Position'] == block]
        if len(match) > 0:
            row = match.iloc[0]
            print(f"\n{block}: Sample #{row[' No.']}")
            print(f"  HER2: {row['HER2']} | ER: {row['ER']} | PR: {row['PR']} | Ki67: {row['Ki67']}")
        else:
            print(f"\n{block}: 未找到")
    except Exception as e:
        print(f"\n{block}: 错误 - {e}")

