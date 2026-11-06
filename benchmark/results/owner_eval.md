# 主人识别评测（eres2net）

- 注册 9 人：deniro、emma、freeman、halle、jensen、jlaw、leijun、musk、tom
- 测试 24 条：dev 10 条（调阈值）/ validation 14 条（切片 + EER + FAR/FRR 曲线）
- 反向验证（--shuffle 打乱注册向量）：关

## dev 集（整条，调阈值用）

| 文件 | 真实 | 最高相似度 | 判定（阈值 0.41） |
|---|---|---|---|
| daniel_test_16384b62.wav | daniel | 0.377 | ✅ 陌生人 |
| daniel_test_2214a420.wav | daniel | 0.467 | ❌ deniro |
| deniro_test_9fb04fdc.wav | deniro | 0.802 | ✅ deniro |
| deniro_test_dec25731.wav | deniro | 0.795 | ✅ deniro |
| freeman_test_04c077cd.wav | freeman | 0.712 | ✅ freeman |
| jensen_test_fd2308ef.wav | jensen | 0.629 | ✅ jensen |
| jlaw_test_d776b812.wav | jlaw | 0.797 | ✅ jlaw |
| musk_test_9527fd28.wav | musk | 0.590 | ✅ musk |
| tom_test_9a6a3cb2.wav | tom | 0.683 | ✅ tom |
| weaver_test_158c8f0e.wav | weaver | 0.412 | ❌ halle |

## validation 集

| 文件 | 时长 | 真实 | 最高相似度 |
|---|---|---|---|
| daniel_test_04549813.wav | 整条 | daniel | 0.365 |
| daniel_test_23e2c4ac.wav | 整条 | daniel | 0.346 |
| deniro_test_ad074fb3.wav | 整条 | deniro | 0.759 |
| emma_test_31e1b798.wav | 整条 | emma | 0.684 |
| freeman_test_25d06cbd.wav | 整条 | freeman | 0.730 |
| halle_test_b94704a6.wav | 整条 | halle | 0.879 |
| halle_test_ed90461b.wav | 整条 | halle | 0.850 |
| jlaw_test_73f85b8c.wav | 整条 | jlaw | 0.672 |
| jlaw_test_8bf7cb09.wav | 整条 | jlaw | 0.813 |
| leijun_test_82c4b8c6.wav | 整条 | leijun | 0.860 |
| tom_test_8ce2dcf9.wav | 整条 | tom | 0.705 |
| tom_test_a25a57d8.wav | 整条 | tom | 0.847 |
| weaver_test_156ad1ed.wav | 整条 | weaver | 0.243 |
| weaver_test_68977e64.wav | 整条 | weaver | 0.231 |

### 整条识别（validation 集，阈值 0.41）

| 文件 | 真实 | 最高相似度 | 判定 |
|---|---|---|---|
| daniel_test_04549813.wav | daniel | 0.365 | ✅ 陌生人 |
| daniel_test_23e2c4ac.wav | daniel | 0.346 | ✅ 陌生人 |
| deniro_test_ad074fb3.wav | deniro | 0.759 | ✅ deniro |
| emma_test_31e1b798.wav | emma | 0.684 | ✅ emma |
| freeman_test_25d06cbd.wav | freeman | 0.730 | ✅ freeman |
| halle_test_b94704a6.wav | halle | 0.879 | ✅ halle |
| halle_test_ed90461b.wav | halle | 0.850 | ✅ halle |
| jlaw_test_73f85b8c.wav | jlaw | 0.672 | ✅ jlaw |
| jlaw_test_8bf7cb09.wav | jlaw | 0.813 | ✅ jlaw |
| leijun_test_82c4b8c6.wav | leijun | 0.860 | ✅ leijun |
| tom_test_8ce2dcf9.wav | tom | 0.705 | ✅ tom |
| tom_test_a25a57d8.wav | tom | 0.847 | ✅ tom |
| weaver_test_156ad1ed.wav | weaver | 0.243 | ✅ 陌生人 |
| weaver_test_68977e64.wav | weaver | 0.231 | ✅ 陌生人 |

整条准确率（阈值 0.41）：**14/14**

### 时长鲁棒性（validation 集，切片逐段识别）

| 片段时长 | 片段数 | 准确率 | 陌生人误认率 |
|---|---|---|---|
| 1s | 151 | 71% | 100%（36 段） |
| 2s | 72 | 76% | 100%（17 段） |
| 3s | 43 | 77% | 100%（10 段） |
| 5s | 25 | 76% | 100%（6 段） |

### 阈值（validation 集 3s 片段）

同一人相似度均值 0.664，不同人均值 0.141；**EER 0.7%，对应阈值 0.42**

| 阈值 | FAR | FRR |
|---|---|---|
| 0.30 | 9.0% | 0.0% |
| 0.35 | 5.6% | 0.0% |
| 0.40 | 3.1% | 0.0% |
| 0.45 | 1.4% | 0.0% |
| 0.50 | 0.6% | 3.0% |
| 0.55 | 0.3% | 12.1% |
| 0.60 | 0.0% | 18.2% |


