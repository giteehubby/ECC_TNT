from idlelib.iomenu import encoding

from blonde import BLONDE

# 初始化 BlonDe
blonde = BLONDE()
with open('output_doubao.txt','r',encoding='utf-8') as f:
    sys=f.readlines()

with open('data/0.ref.txt','r',encoding='utf-8') as f:
    ref=f.readlines()
# 计算分数
score = blonde.corpus_score([sys], [ref])

print(score)