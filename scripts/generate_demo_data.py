import random
import pandas as pd
import numpy as np

# 固定随机种子
random.seed(42)
np.random.seed(42)

# 模拟一个 450aa 的蛋白序列
aa = list("ACDEFGHIKLMNPQRSTVWY")
wt_seq = "".join(random.choices(aa, k=450))

# 假设这些位点是真正影响活性的“功能位点”
important_sites = [50, 129, 208, 310, 375]

data = []

for i in range(1000):
    seq_list = list(wt_seq)

    # 随机选择一个突变位点
    pos = random.randint(0, 449)
    original = seq_list[pos]
    new_aa = random.choice([a for a in aa if a != original])

    seq_list[pos] = new_aa
    mutated_seq = "".join(seq_list)

    # 构造“有规律”的活性函数
    activity = 1.0

    if pos in important_sites:
        activity += np.random.normal(1.0, 0.2)  # 有利突变
    else:
        activity += np.random.normal(0.0, 0.2)  # 随机波动

    activity = max(0.1, activity)

    data.append({
        "variant": f"{original}{pos+1}{new_aa}",
        "sequence": mutated_seq,
        "activity": round(activity, 3)
    })

df = pd.DataFrame(data)
df.to_csv("../data/mutants.csv", index=False)

print("Demo dataset generated: 1000 mutations")
print(df.head())