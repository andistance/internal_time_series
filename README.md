# 推理轨迹数据构造

使用完整的 **40,315 条题库**，先比较 Qwen3-8B thinking 的 `0.08_800` 与 `0.12_500` 两种 phase 划分，再根据效果决定是否扩充数据。文件名中的样本范围从 1 开始编号；不设置 `--limit`。训练步骤另行安排。

## 1. 安装、运行与使用说明

### 1.1 配置环境并下载代码

在 **CUDA 12.2、已安装 Conda 的 Linux 服务器**上，将项目根目录的 `requirements.txt` 放到服务器当前工作目录，然后依次执行：

```bash
# 创建 Python 3.12.8 环境，其他库全部通过 pip 安装。
conda create -n internal_time_series python=3.12.8 -y
conda activate internal_time_series
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# 下载代码，最后停在仓库上一级目录，后续命令都在这里执行。
git clone https://github.com/andistance/internal_time_series.git
cd internal_time_series
cd ..
```

清单固定了 `torch==2.6.0+cu118`，额外索引用于下载该构建，见 [PyTorch 安装说明](https://docs.pytorch.org/get-started/previous-versions/)。CUDA 12.2 描述的是服务器环境，不会自动改变清单中的包版本；PyTorch 与 vLLM 的 CUDA 构建仍需匹配。这里未实际安装或验证目标服务器环境。

### 1.2 手动修改路径和 GPU 配置，再运行实验

**先修改模型路径。** 用编辑器打开要运行的 `scripts/build_data_*.sh`，将每条生成命令和每条 Monte Carlo 标注命令中的 `--model_path` 改成这台服务器的模型目录：

| 脚本里的原值 | 手动替换为，例如 |
| --- | --- |
| `/mnt/wangchenglong/models/Qwen3-8B` | `/srv/models/Qwen3-8B` |
| `/mnt/wangchenglong/models/DeepSeek-R1-Distill-Qwen-1.5B` | `/srv/models/DeepSeek-R1-Distill-Qwen-1.5B` |

填写保存模型配置、tokenizer 和权重的**目录**，不是某个权重文件。生成与标注两处都要改；sharded 清单里的五条标注命令也都要改。当前脚本中依赖原服务器的绝对路径就是这两种模型路径。

其他路径以 `internal_time_series/` 开头，只要从仓库上一级目录运行，可以保持不变。若将数据或日志另放到其他磁盘，手动同步修改各步骤的输入、输出和 `--work_dir`，让下一步读取上一步实际生成的文件。代码、题库与模型都需要在服务器上准备好。

**接着设置 GPU：先理解 DP 和 TP。**

- **DP（数据并行）**：每组保存一份完整模型，不同组处理不同题目或标注任务。DP 越多，能同时处理的独立任务越多。
- **TP（张量并行）**：同一份模型拆到多张卡上协同运行，用于单卡放不下模型或当前运行预算的情况。
- **DP+TP**：同时运行多组模型，每组内部再用多张卡做 TP。

本项目用分组字符串表示它们：**分号 `;` 分开 DP 组，逗号 `,` 连接同组 TP 卡**。生成命令使用 `--gpus`，标注命令使用 `--gpu_groups`，每组卡数必须与 `--tensor_parallel_size` 一致；DP 数量就是组数，不另传 DP 数量参数。例如同一台 6 卡服务器：

| 运行方式 | 生成命令中填写 | 标注命令中填写 | 实际含义 |
| --- | --- | --- | --- |
| 纯 DP | `--tensor_parallel_size 1 --gpus "0;1;2;3;4;5"` | `--tensor_parallel_size 1 --gpu_groups "0;1;2;3;4;5"` | DP=6、TP=1，每张卡各处理自己的任务 |
| 纯 TP | `--tensor_parallel_size 2 --gpus "0,1"` | `--tensor_parallel_size 2 --gpu_groups "0,1"` | DP=1、TP=2，只用卡 0、1 共同运行一份模型 |
| DP+TP | `--tensor_parallel_size 2 --gpus "0,1;2,3;4,5"` | `--tensor_parallel_size 2 --gpu_groups "0,1;2,3;4,5"` | DP=3、TP=2，三组同时处理任务，每组用两张卡 |

如果是 8 卡纯 DP，将分组字符串写为 `"0;1;2;3;4;5;6;7"`；8 卡做 DP=4、TP=2，则写为 `"0,1;2,3;4,5;6,7"`。引号要保留，卡号只代表本机 GPU。

**本实验使用 H100 时，建议优先 TP=1、每张 GPU 一个 DP。Qwen3.8-27B已经设置了BF16，理论上单卡也能容纳下一个模型推理，无需设置TP。** 单卡显存能容纳 Qwen3-8B 、DS-1.5B 或 Qwen3.8-27B 及当前上下文、并发预算时，独立副本可以同时处理更多任务，并省去 TP 通信，通常更适合提高整批数据的吞吐，参见 [vLLM 并行策略说明](https://docs.vllm.ai/en/stable/configuration/optimization/#parallelism-strategies)。

<h4 style="color: gray;">1.2.1 0.12_500和0.08_800的对比实验（已做完）</h4>
**然后运行两个对比实验。** 普通 Bash 会依次完成生成与 phase 划分、判题、整份 Monte Carlo 标注：

```bash
# 对比 A：第 1～504 题，Qwen3-8B thinking，0.08_800。
bash -e internal_time_series/scripts/build_data_1-504_samples_0.08_800_Qwen3-8B.sh

# 对比 B：相同的第 1～504 题，Qwen3-8B thinking，0.12_500。
bash -e internal_time_series/scripts/build_data_1-504_samples_0.12_500_Qwen3-8B.sh
```

`bash -e` 会在脚本中的命令失败后停止。两组题目相同，但回答独立采样，不保证推理文本完全相同。

两组结果分别保存到：

```text
internal_time_series/data/train/deepscaler_phases_0.08_800_16384/Qwen3-8B/shard0_total80/
internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard0_total80/
```

**标注时，手动打开对应日志查看进度。** 在服务器的文件管理器或编辑器中，找到该标注命令 `--work_dir` 指向的目录。两个对比实验分别是：

```text
internal_time_series/log/mc_parallel_work_Qwen3-8B_0.08_800_shard0_total80/
internal_time_series/log/mc_parallel_work_Qwen3-8B_0.12_500_shard0_total80/
```

打开其中的 `worker_shard0.log`、`worker_shard1.log` 等文件，查看末尾的任务进度、保存信息或报错；运行过程中刷新文件即可。**一个 DP worker 对应一个日志**：DP=3 时是 0～2，按 H100 建议改为 DP=6 后是 0～5。

主终端也会打印 `[launcher] Log: ...`，可以直接按该位置打开文件；出现 `[launcher] All done.` 表示本轮标注及内部合并完成。其他批次或 part 的日志同样在它自己的 `--work_dir` 中，不按原始题库的分片编号查找。

**先比较训练效果。只有决定沿用 `0.12_500` 后，再运行下面的扩充：**

#### 1.2.2 采用0.12_500，DS-1.5B、Qwen3-8B和Qwen3.8-27B轮流构造数据

**在运行之前，还是要根据1.2的提示先修改模型路径和GPU配置**，之后再运行下面的命令，先补齐Qwen3-8B的剩下500条，之后再三个模型以1000条为单位轮流构造数据：

```bash
# 补充第 505～1008 题：新增 504 条，Qwen3-8B 累计 1008 条。
bash -e internal_time_series/scripts/build_data_505-1008_samples_0.12_500_Qwen3-8B.sh

# DS-1.5B 生成第 1009-2016 条。
bash -e internal_time_series/scripts/build_data_1009-2016_samples_0.12_500_DS-1.5B.sh

# Qwen3.8-27B 生成第 2017-3024 条。
bash -e internal_time_series/scripts/build_data_2017-3024_samples_0.12_500_Qwen3.8-27B.sh

# Qwen3-8B 生成第 3025-4032 条。
bash -e internal_time_series/scripts/build_data_3025-4032_samples_0.12_500_Qwen3-8B.sh

# Qwen3.8-27B 生成第 4033-5040 条。
bash -e internal_time_series/scripts/build_data_4033-5040_samples_0.12_500_Qwen3.8-27B.sh

# DS-1.5B 生成第 5041～6048 条。
bash -e internal_time_series/scripts/build_data_5041-6048_samples_0.12_500_DS-1.5B.sh
```

所有扩充数据属于 `0.12_500`：Qwen3-8B 前两批是 `Qwen3-8B/shard0_total80`、`Qwen3-8B/shard1_total80`，后一批是`Qwen3-8B/shard3_total40`；DS-1.5B 的两批是 `DS-1.5B/shard1_total40`、`DS-1.5B/shard5_total40`；Qwen3.8-27B 是 `Qwen3.8-27B/shard2_total40`、`Qwen3.8-27B/shard4_total40`。`0.08_800` 不再使用。

---

> **实验运行顺序到此结束。以下内容用于按需理解、修改或恢复命令，不是需要接着执行的任务。**

---

### 1.3 命令修改与断点续跑

#### Bash的三条命令各做什么

一个Bash包含三条命令：先生成回答，判断每个回答的正确性，再对推理前缀做 Monte Carlo 标注。恢复前先分清自己停在哪一条命令；修改实验时，也只改对应的那条命令。普通 Bash 的三步如下，拆分与合并在 1.5 介绍。

| 顺序 | Python 文件 | 作用和产物 |
| --- | --- | --- |
| 1 | `1_generate_internal_time_series_with_phases_vllm[_qwen].py` | 每题生成一条完整回答，再计算 token 熵、step 置信度和 phase 边界，写出 summary/steps。方括号表示按模型选择后缀，不是实际文件名。 |
| 2 | `2_validate_answer_correctness.py` | 提取完整回答中 boxed 最终答案并判题，将样本级 `is_correct` 写回 summary/steps，不删除答错样本。优先使用 `</think>` 后的回答。 |
| 3 | `4_build_phase_prospective_correctness_label_via_monte_carlo[_qwen].py` | 在选中的 phase 终点续写并判题，输出 labeled steps 与详情。 |

step 置信度 `token_avg_confidence` 是该 step 内实际生成 token 概率 `exp(logprob)` 的算术平均。Monte Carlo 的 `step_prospective_correctness` 则是**答对的续写数 ÷ 实际续写数**，例如 8 条中答对 6 条为 0.75；它与原始回答的 `is_correct` 是不同标签。

标注只选择 `phase_frontier == 2` 的终点，默认相邻选中位置至少间隔 128 token；包含 `</think>` 的 step 及后续 step 不标注。`phase_frontier` 的 0/1/2 通常表示起点/内部/终点，单 step phase 可能优先保留起点标记。最终仍保留完整轨迹，不是每个 step 或每个 phase 都必须有 Monte Carlo 标签。生成时的 `phase_ranges` 覆盖到含首次 `</think>` 的 step，范围里的 step 编号从 1 开始；后续答案仍保留在文件中。

#### 生成命令：想改 phase、长度或题目范围时看这里

这一步决定用哪批题、由哪个模型生成，以及如何划分 phase。下面列出脚本默认值，按需求修改对应参数，并同步更新后面判题和标注的路径。

| 参数 | 当前值 | 含义及怎么改 |
| --- | --- | --- |
| `--input_json` | `.../deepscaler_random_shuffled.json` | JSON 数组题库，每题包含 `problem`、`answer`。换路径或文件时保持顺序与版本一致，否则同一 shard 会变成不同题目。 |
| `--model_path` | 本机模型绝对路径 | 生成模型和 tokenizer 来源，标注阶段也须改为同一模型；跨系列切换见 1.4。 |
| `--summary_jsonl` | 本批 `summary.jsonl` | 完整回答、长度、step 置信度序列和 phase 范围的输出路径。 |
| `--step_jsonl` | 本批 `steps.jsonl` | 全部 steps、置信度、token 起点和 phase 标记的输出路径；后续使用 `--steps_path` 读取，注意单复数。 |
| `--max_tokens` | `16384` | 原始回答最多新增 token 数，不强制生成到上限。 |
| `--max_model_len` | `16384` | prompt 与输出合计的上下文上限，不是额外生成长度。 |
| `--tensor_parallel_size` | `1` | 每个模型副本内部的 TP 卡数，不是总 GPU 数。 |
| `--gpus` | `"0;1;2;3;4;5"` | 分号分隔独立 worker，逗号分隔同组 TP 卡；组内卡数与 TP 一致，字符串需加引号。 |
| `--gpu_memory_utilization` | `0.98` | vLLM 显存预算比例；影响显存分配，不是算力利用率，也不设置 Transformers 熵计算的显存比例。 |
| `--num_input_parts` | `80` 或 `40` | 把整个题库等分的份数，不是本次题目数量。 |
| `--input_part_id` | 对应分片编号 | 从 0 开始选择一份，必须小于总份数；实际范围见下文的样本范围表。 |
| `--phase_partition_algorithm` | `adaptive` | 自适应划分；另一选项 `original` 使用固定段数与熵阈值，当前实验不使用。 |
| `--high_entropy_ratio` | `0.08` 或 `0.12` | 高熵 token 的候选比例，范围为大于 0、至多 1，不是熵阈值，也不是标注 step 的比例。 |
| `--target_phase_tokens` | `800` 或 `500` | 期望平均 phase 长度，必须为正。算法按轨迹长度选择段数并选边界，不保证每段等长。 |
| `--entropy_chunk_size` | `16` | 精确熵前向的 token 分块大小，影响显存和速度；不是 phase 长度、样本数或生成上限。 |

#### 判题命令：读取生成结果并写入正确性

这一步给原始完整回答判题，不生成 Monte Carlo 标签。通常只需让下面两个输入路径跟随本批生成结果。

| 参数 | 含义 |
| --- | --- |
| `--summary_path` | 第一条生成命令的 summary 输出，用于读取完整回答。 |
| `--steps_path` | 同一批 steps 输出，将对应的样本级判题结果写回其中。 |
| `--no_backup` | 无后续值的开关：本次判题不生成 `.bak`。去掉它会先备份；不要写成 `--no_backup false`。 |

#### 标注命令：想改采样次数或批大小时看这里

这一步从已生成的推理前缀继续采样，估计最终答对的概率。修改 K 就是修改每个选中终点的续写次数；修改 batch_size 只是调整同时处理多少个前缀。表中的 GPU 配置是仓库默认值，H100 可按 1.2 改成每卡一个 DP。

| 参数 | 当前值/含义及怎么改 |
| --- | --- |
| `--steps_path` | 已生成并判题的完整 steps 或指定 part；不能混用其他模型、phase 配置或 shard。 |
| `--output_steps_path` | labeled 轨迹输出；未开启可选 `--inplace` 时不覆盖原输入。 |
| `--output_details_path` | 终点级续写、判题、正确次数和实际采样数等详情输出，父目录须存在。 |
| `--model_path` | 同一生成模型的本机路径；生成、标注两处都要修改。 |
| `--tensor_parallel_size`、`--gpu_groups` | 每 worker 的 TP 和本机 GPU 分组。Qwen 整份版是 TP=2、`"0,1;2,3;4,5"`；其余标注是 TP=1、`"0;1;2;3;4;5"`。注意参数名不是生成的 `--gpus`。 |
| `--work_dir` | worker 输入副本、已有标签、详情和日志的状态目录。每个实验/part 独立，恢复原任务才复用。 |
| `--K` | 当前为 8，表示每个选中终点续写次数；Python 默认值是 10，所以应显式传入。 |
| `--batch_size` | 当前为 8，表示每个 worker 一批处理多少前缀任务；K=8 时最多产生 64 条续写，不表示 8 道原题。 |
| `--save_every` | 当前为 30，累计至少完成这么多个前缀任务后保存；按批检查，batch=8 时通常到 32 个任务保存。 |
| `--max_tokens` | 每条续写新增 token 上限，当前 16384；独立于原始生成阶段的同名设置。 |
| `--max_model_len` | 题目、已有前缀与续写合计上限，当前 16384；已有前缀越长，剩余续写空间越少。 |
| `--gpu_memory_utilization` | 当前为 0.98，标注 vLLM 的显存预算；增加 K、批大小和前缀长度都会影响显存。 |
| `--vllm_enforce_eager` | 无值开关，禁用 CUDA Graph，减少图捕获相关显存开销，可能影响速度。现有 Qwen 整份版开启，其余标注未开启。 |

生成和标注默认采样温度为 `--temperature 0.7`、`--top_p 0.95`，可分别显式修改；默认未设置 `--seed`。固定 seed 有助于复现，但不同 GPU 分组或软件版本仍可能得到不同输出。Qwen thinking 默认 true，可显式传 `--enable_thinking true`。

标注的可选参数 `--min_label_token_gap` 默认 128，设为 0 可关闭间隔过滤，仍保留 phase 和 think-end 筛选。需要在详情额外保存前缀、prompt、拼接后的回答时，分别加 `--save_prefix_in_details`、`--save_prompt_in_details`、`--save_full_output_in_details`。

**区分同名 `--K`：生成脚本里它是 original 模式固定段数的别名，标注脚本里才是 Monte Carlo 次数。** 当前 adaptive 的段数由长度决定。

#### 修改样本范围和其他设置

当你想生成下一批题目时，先确定题库位置，再根据下面的表修改生成命令的两个分片参数。这样不会重复生成前面的题目。

完整题库的分片对应关系：

| 题目范围 | 数量 | `num_input_parts` | `input_part_id` | shard 目录 |
| --- | ---: | ---: | ---: | --- |
| 1～504 | 504 | 80 | 0 | `shard0_total80` |
| 505～1008 | 504 | 80 | 1 | `shard1_total80` |
| 1009～2016 | 1008 | 40 | 1 | `shard1_total40` |
| 2017～3024 | 1008 | 40 | 2 | `shard2_total40` |
| 3025～4032 | 1008 | 40 | 3 | `shard3_total40` |
| 4033～5040 | 1008 | 40 | 4 | `shard4_total40` |
| 5041～6048 | 1008 | 40 | 5 | `shard5_total40` |

对于题目总数 N、总份数 P 和 0-based 分片 i，代码的计算方式是：

```text
q, r = divmod(N, P)
start = i*q + min(i, r)
size = q + (1 if i < r else 0)
选取 data[start:start+size]，对应人类编号 start+1 ～ start+size
```

前 r 份各多一题。40315 条题库在 80 等分时前 75 份各 504 条、后 5 份各 503 条；40 等分时前 35 份各 1008 条、后 5 份各 1007 条。先完成两个 504 条分片，再从 40 等分的第 1 份继续，恰好衔接为 1009～2016。

| 需求 | 怎么修改命令 |
| --- | --- |
| 换成 `0.08_800` | 生成改 `--high_entropy_ratio 0.08 --target_phase_tokens 800`，同时替换所有上下游结果路径和独立工作目录中的配置标识。仅改目录名或重跑标注不会重新划分已有 steps。 |
| 换到下一批 505～1008 | 生成改 `--num_input_parts 80 --input_part_id 1`，所有上下游路径指向 `shard1_total80`，标注换新工作目录。 |
| 原始回答上限改 8192 | 生成改 `--max_tokens 8192`。总上下文 `--max_model_len` 是否同改，由实验决定；例如仍设 16384 就是输出上限 8192、总上下文上限 16384。 |
| 续写上限改 8192 | 在标注命令改 `--max_tokens 8192`；不会改变原始回答。两阶段都要改长度时，需要分别修改。 |
| 每终点采样改 4 次 | 只改标注的 `--K 4`，不是 `--batch_size 4`；新配置使用独立输出和工作目录。 |
| 降低批量并发 | 改标注 `--batch_size`，不改变 K、题目数或 phase 参数。 |

所有参数都要改在对应 Python 命令中；目录名不控制程序行为。路径应保持同一条数据链：

```text
生成 --summary_jsonl → 判题 --summary_path
生成 --step_jsonl → 判题 --steps_path → 标注 --steps_path
标注输出 → 本实验自己的 labeled/details 目录
标注 --work_dir → 本实验自己的 worker 状态与日志目录
```

当前接口不提供任意“起点 + 条数”参数，不能添加未实现的 `--start_idx` 或 `--num_samples`；不能由等分表达的范围，需要另存精确输入切片或明确修改 Python 接口。本流程使用全部 40315 条，不截断输入。更换题库或顺序后重新计算范围和文件名。

#### 最后会得到哪些文件

- `summary.jsonl`：每题完整回答、长度、step 置信度序列、phase 范围及原始回答的判题结果。
- `steps.jsonl`：全部 steps、置信度、token 起点、phase 标记及样本级判题结果。
- `steps_labeled.jsonl`：保留完整轨迹，在部分选中终点增加 `step_prospective_correctness`。
- Monte Carlo details：保留续写文本、判题、采样数量等标签来源；整份版一个文件，sharded 版五个 part。

选中题目数量不等于成功产出数量，错误信息可从结果和日志定位。当前脚本不执行人工筛选，也不调用 `6_filter_llm_judged_bad_labeled_samples.py`。不同生成 shard 的 `sample_idx` 会重新从局部位置编号，跨批汇总时保留模型、配置和生成 shard 身份，不能仅用它当全局唯一 ID。

交给 Codex 改命令时，说明目标 Bash、服务器路径、模型、题目范围、phase 参数、两阶段长度、K、GPU 分组和标注方式；要求联动更新上下游路径与独立工作目录即可。

#### 断点续跑

一个Bash包含三条命令：先生成回答，判断每个回答的正确性，再对推理前缀做 Monte Carlo 标注。第一条和第三条命令支持断点续跑。**如果中途停止，你需要先确定是Bash中的哪条命令中途停止了，通常只需要继续没有完成的那个命令，不必重跑全部流程**。 本节先说明怎样继续，再解释每条命令的参数，方便你调整实验需求。

| 情况 | 怎么操作 |
| --- | --- |
| 原始生成中断 | 在原来的生成 Python 命令中加 `--resume`，保持输入、分片、模型、phase 配置和输出路径不变。它复用生成中间记录，并跳过 summary/steps 中已成对完成的样本。 |
| Monte Carlo 标注中断 | 重跑原来的标注 Python 命令，保留同一输入、GPU worker 分组和 `--work_dir`。默认会复用状态并跳过已有标签，不需要加 `--resume`。 |
| 想从头生成 | 使用新输出目录；确认要删除本次生成文件时才加生成的 `--overwrite`。不能和 `--resume` 同用。 |
| 改了模型、phase、K、续写长度或分片 | 使用新结果目录和新 `--work_dir`，不要将旧标签当成新配置继续使用。 |

标注的 `--restart` 会重建 worker 中间状态，标注的 `--overwrite` 则要求重算符合条件的已有标签，二者与生成的同名开关含义不同。改变标注 worker 分组时也应重新准备状态，避免复用旧的任务分配。

恢复哪一步，就单独执行哪条 Python 命令。现有 Bash 不接收额外 CLI 参数，`bash 文件.sh --resume` 或 `bash 文件.sh --K 4` 不会修改内部命令。不加恢复选项就重跑整个 Bash，可能重新生成并追加重复记录。

### 1.4 更换模型系列

当你想让另一个系列的模型生成轨迹并为自己的轨迹标注时，需要同时切换生成和标注两条命令的模型及提示词模板。在这个项目中，生成阶段和 Monte Carlo标注阶段支持两种prompt模板，并且每个阶段各配套了两种不同的python文件： **Qwen系列的用带_qwen后缀的python文件，用Qwen的提示词模板，DeepSeek系列的用不带后缀的通用python文件，用通用的提示词模板**。下面列出两条路线的对应文件，再说明需要一起修改哪些地方。

| 项目 | Qwen3-8B thinking | DeepSeek-R1-Distill-Qwen-1.5B |
| --- | --- | --- |
| 生成 Python 文件 | `1_generate_internal_time_series_with_phases_vllm_qwen.py` | `1_generate_internal_time_series_with_phases_vllm.py` |
| 标注 Python 文件 | `4_build_phase_prospective_correctness_label_via_monte_carlo_qwen.py` | `4_build_phase_prospective_correctness_label_via_monte_carlo.py` |
| 提示词 | Qwen3 chat template | 通用题目 + `PROMPT_SUFFIX`，续写时再接推理前缀 |
| thinking 参数 | 默认 true，可显式传 `--enable_thinking true` | 通用脚本不接受该参数，需要删除 |
| 结果目录别名 | `Qwen3-8B` | `DS-1.5B` |

换模型时一并修改：**两阶段 Python 文件、两处 `--model_path`、所有结果路径中的模型别名、独立工作目录**。判题、拆分和合并脚本共用，不需要更换。若另外传过 `--entropy_model_path`，也要同步处理；当前不传时熵模型跟随生成模型。

DeepSeek 名称中虽然带 Qwen，本项目仍使用不带 `_qwen` 的通用脚本。其他 Qwen 型号也要确认 tokenizer 支持现有 Qwen3 thinking 模板，不能只根据名字选文件。

题目分片独立于模型：DeepSeek 这批用 `--num_input_parts 40 --input_part_id 5` 是实验计划的选择，换模型不会自动选中这批题目。

### 1.5 sharded：人工把标注分给多台服务器

如果一批数据由一台服务器标注太慢，可以先把已经生成的题目分成 5 个文件，让五台服务器同时各标一份，最后收集结果再合并。这是在分配不同题目的标注工作，不是在用 TP 把一份模型拆到多台机器。你需要人工启动服务器，并为每台准备它负责的那条命令。

仓库提供的两份完整命令清单是：

```text
scripts/build_data_1009-2016_samples_0.12_500_sharded_Qwen3-8B.sh
scripts/build_data_5041-6048_samples_0.12_500_sharded_DS-1.5B.sh
```

相同范围的整份版与 sharded 版选一种即可；已经生成并判题过的数据可以直接使用，不必重新生成。

**第一步：生成和判题只做一次，再拆分 `steps.jsonl`。** 以 Qwen 第 1009～2016 题为例，先执行清单前两条生成、判题命令，然后：

```bash
RUN_DIR=internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40
python internal_time_series/build_data/3_split_step_jsonl_to_shards.py \
  --input_jsonl "$RUN_DIR/steps.jsonl" \
  --output_folder "$RUN_DIR/steps_shards"
```

`--input_jsonl` 是完整输入，`--output_folder` 是自动创建的拆分目录。每行仍是一道题的全部 steps。当前 main 固定拆 5 份，1008 条依次分为 **202、202、202、201、201** 条，文件名为 `part_01.jsonl`～`part_05.jsonl`。没有 `--num_parts` 参数；要改份数，需修改 Python 的 `num_parts=5` 或另行增加接口，并增减 part 标注命令。

**第二步：人工登录各台服务器，分别运行自己的 part。** 将输入 part 复制到负责它的服务器。可以从清单复制对应的一条标注命令，另存为 `label_part_01.sh` 等脚本；例如服务器 A 的脚本内容如下：

```bash
#!/usr/bin/env bash
MODEL_PATH=/srv/models/Qwen3-8B
RUN_DIR=internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40
PART=part_01
WORK_DIR="internal_time_series/log/mc_parallel_work_shard1_total40_${PART}"
mkdir -p "$RUN_DIR/steps_labeled_shards" "$RUN_DIR/step_monte_carlo_details_shards"

python internal_time_series/build_data/4_build_phase_prospective_correctness_label_via_monte_carlo_qwen.py \
  --steps_path "$RUN_DIR/steps_shards/${PART}.jsonl" \
  --output_steps_path "$RUN_DIR/steps_labeled_shards/${PART}.jsonl" \
  --output_details_path "$RUN_DIR/step_monte_carlo_details_shards/${PART}.jsonl" \
  --model_path "$MODEL_PATH" \
  --tensor_parallel_size 1 --gpu_groups "0;1;2;3;4;5" \
  --work_dir "$WORK_DIR" --K 8 --batch_size 8 --save_every 30 \
  --max_tokens 16384 --max_model_len 16384 --gpu_memory_utilization 0.98
```

服务器 A 从仓库上一级目录运行 `bash -e label_part_01.sh`；服务器 B 改 `PART=part_02`，另存并运行 `label_part_02.sh`，其余同理。这些单 part 脚本是人工另存的，不是仓库附带的额外文件。

每台按本机修改模型路径、GPU 分组；各 part 保持相同模型版本、K 和采样设置，并使用独立工作目录。日志同样在该 `WORK_DIR/worker_shard*.log`。标注最终文件的父目录不会全部自动创建，所以保留示例中的 `mkdir -p`。同一台机器若同时运行多个 part，需要人工分配不重叠的 GPU。

DeepSeek 按相同方式分发，但改用通用标注 Python、DeepSeek 模型路径、`DS-1.5B/shard5_total40` 数据目录和独立的 DeepSeek 工作目录；不传 `--enable_thinking`。不要让每台服务器都执行整份清单。

**第三步：收齐所有 labeled part，再合并一次。** 在汇总服务器将五个文件放进同一个 `steps_labeled_shards/` 后执行：

```bash
RUN_DIR=internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40
python internal_time_series/build_data/5_merge_step_labeled_jsonls.py \
  --folder "$RUN_DIR/steps_labeled_shards" \
  --output "$RUN_DIR/steps_labeled.jsonl"
```

`--folder` 必须是已标注的目录，程序按 `part_数字.jsonl` 的数字顺序拼接；`--output` 是最终文件，已有同名结果会被覆盖，父目录须存在。合并前确认五份均完成且属于同一实验；程序不会自动判断缺份、标签完成情况或去重。

详情分别保留在 `step_monte_carlo_details_shards/part_0x.jsonl`，现有清单最后只合并 labeled steps。详情中的 `source_line_idx` 是 part 局部行号，汇总时保留 part 来源。
