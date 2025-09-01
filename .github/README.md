## 项目介绍

SGLang 是一个用于大型语言模型和视觉语言模型的快速服务框架。通过协同设计后端运行时和前端语言，它使您与模型的交互更快、更可控。核心功能包括：

- **快速后端运行时**：通过使用 RadixAttention 进行前缀缓存、零开销 CPU 调度器、PD分离、推测解码、连续批处理、分页注意力、张量/流水线/专家/数据并行、结构化输出、分块预填充、量化（FP4/FP8/INT4/AWQ/GPTQ）和多 lora 批处理来提供高效服务。
- **灵活的前端语言**：提供用于编程 LLM 应用程序的直观接口，包括链式生成调用、高级提示、控制流、多模态输入、并行性以及外部交互。
- **广泛的模型支持**：支持广泛的生成模型（Llama、Qwen、DeepSeek、Kimi、GPT、Gemma、Mistral 等）、嵌入模型（e5-mistral、gte、mcdse）和奖励模型（Skywork），并且易于扩展以集成新模型。
- **活跃的社区**：SGLang 是开源的，并拥有一个活跃的社区以及工业界的广泛应用。

SGLang社区当前已对不同厂商、多个类型的硬件做到原生支持且性能表现优异，在华为公司及其他社区贡献者的共同努力下，当前SGLang已完成对昇腾NPU硬件基础功能的支持（含大EP方案），但相比vLLM、MindIE等其他推理框架，特性完备度及性能表现均存在差距，还需要持续在集群管理、模型适配、模型调优、推理引擎、加速库（含三方）、算子等不同层次做进一步的能力补齐和性能优化，使SGLang+昇腾NPU的组合释放出最佳的性能表现。

考虑到在近期处于高频进行特性叠加和性能优化的阶段，将会存在大规模的代码变更和十分频繁的PR合入，为了避免对社区主仓的功能及性能基线造成破坏，特设置该内源项目，旨在汇聚公司内各领域的专家力量，在SGLang社区对NPU支持的总体架构和节奏规划之下，快速进行迭代开发，并合入到该内源仓进行功能验证，确保不会对主社区产生破坏后再提交PR合入至主社区。

## 代码仓及用途

**本代码仓仅用于公司内各团队在合入主社区前的方案评审、联调验证，不会用于发布版本、issue处理等，该类事务一律在[主社区](https://github.com/sgl-project/sglang)跟踪和处理。**

针对SGLang使能昇腾NPU硬件，所有涉及的代码仓清单及内源贡献方式汇总如下：

| 序号 | 社区主仓             | 用途            | 内源贡献方式            | 内源社区运营人员                    |
| ---- | ------------------ | --------------- |--------------- |-----------------------------|
| 1    | **[sglang](https://github.com/sgl-project/sglang)** | 社区主仓，包括集群管理、模型适配、模型调优、推理引擎等核心功能，是主要的贡献代码仓 | 内源仓库：[Ascend/sglang](https://github.com/Ascend/sglang)<br> 公司内各领域团队，涉及该仓库修改的，**全部先合入到内源仓库**，并在此进行统一评审、联调验证，再由内源committer统一cherry-pick并提交PR至主社区<br>**严禁直接将代码提交至社区主仓，避免引起修改引入、方案冲突等问题** | @ping1jing2, @iforgetmyname |
| 2    | [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) | NPU特殊算子库及加速能力，是主要的贡献代码仓 | **直接贡献代码至社区主仓**，相关PR需经过SGLang内源PMC及committer的评审 | @RuixuanZhang06             |
| 3 | [vLLM](https://github.com/vllm-project/vllm) | SGLang社区原生的编译依赖，一般不涉及修改 | 直接贡献代码至社区主仓，与SGLang相关的PR需经过SGLang内源PMC及committer的评审 | @xiaomingbao008             |
| 4 | [Triton-Ascend](https://github.com/Ascend/triton-ascend) | 提供Triton on Ascend和inductor相关能力，可对接至SGLang中，由其他横队主导 | 直接贡献代码至社区主仓，与SGLang相关的PR需经过SGLang内源PMC及committer的评审 | @iforgetmyname              |
| 5 | [kvcache](https://gitee.com/ascend/memfabric_hybrid) | 我司开发的兼容mooncake接口的加速库及底层内存编制能力，当前托管在昇腾社区 | 当前代码在昇腾社区，但还未完成开源，内部可见，单独添加权限后，可直接贡献代码至昇腾社区，与SGLang相关的PR需经过SGLang内源PMC及committer的评审 | @xiaomingbao008             |
| 6 | CANN | 公司内组件，暂未开源，通过support网站获取商发/POC版本 | 由对应团队完成开发，也可以直接贡献至对应团队的仓库，相关MR由责任田的committer做评审，涉及方案分歧上升至横队技术TMG决策 | @realray808                 |
| 7 | PTA | 公司内组件，暂未开源，通过support网站获取商发/POC版本 | 由对应团队完成开发，也可以直接贡献至对应团队的仓库，相关MR由责任田的committer做评审，涉及方案分歧上升至横队技术TMG决策 | @realray808                 |

## 贡献方式

为方便各部门聚焦做特性开发和代码贡献，本项目PMC成员将安排统一的集成构建、自动化验证等活动，并在验证通过后，由社区committer统一cherry-pick并0提交PR至社区。相关的开发及提交流程如下：
![image](./assets/development_workflow.png)
内源中心仓地址：https://github.com/Ascend/sglang
详细流程步骤说明：

1. 内源仓fork自社区主仓，并定期同步，main分支和主干保持一致，用于向社区提交PR，拉出develop分支用于内部开发联调及防护网验证（需修改CI相关代码）；
2. 所有内源贡献者在蓝区开发，并提交PR至内源中心仓，由内源PMC成员及committer审核后合并；
3. 每天都会在内源中心仓上进行日构建和CI自动化验证，确保合入代码质量；
4. 验证完毕后，由对接社区的committer（@ping1jing2、@iforgetmyname等）统一cherry-pick至main分支，并提交PR至主社区，联系社区committer合入；
5. 在社区主仓的日构建出镜像，并提交给测试进行测试验证，若涉及CANN等内部组件尚未正式发布的版本配套，则单独替换从内部获取的闭源run包；
6. 跟随社区的发布节奏发布版本，发布release（预计2周一个）或rc/post版本

*在公司内黄区网络开发代码，推荐使用黄蓝协同工具进行开发，详细使用说明可在内网搜索安全空间指南/黄蓝协同等资料*

## 业务范围及committer任命

结合SGLang社区软件架构，识别其支持NPU，涉及的核心修改点及特性如下：

内源社区的committer基于实际的贡献做推举，按季度进行更新。综合考虑前一个周期和项目总周期内的贡献度、参与度，由PMC成员或现有committer提名候选人，并在PMC例会中做表决。

当前，综合考虑前期在SGLang社区独立贡献的情况和PMC成员提名，结合SGLang社区软件架构，特对NPU使能所涉及的关键领域首批内源committer任命如下：

| 子领域      | Committer                                             | 使命                                                                                               |
| -------- |-------------------------------------------------------| ------------------------------------------------------------------------------------------------ |
| 集群管理     | 计算：@xiaomingbao008, @yuzhiyou2009 2012：@hsliuustc0106 | ①极简易用部署，一键自动化等②集群故障检测及恢复，实例、Token级③系统级调度优化，见调度标签页                                      |
| 模型适配     | 计算：@ping1jing2 图灵：@ZhengdQin                          | ①优先兼容主流生态模型，开箱即用DS、K2、Qwen ②横向扩展模型支持度100%与社区匹配，具体模型见模型标签页                                   |
| 模型调优     | 计算：@iremembermyname, @iforgetmyname 图灵：@ZhengdQin     | ①聚焦主流模型DS、Qwen开箱即优 ②新增模型开箱即优                                                                |
| 推理引擎     | 计算：@boxwh-s 公开：@wtxjn 2012：@dragon835 图灵：@ZhengdQin   | ①社区基本功能对接，PD分离、PrefixCache、aclgraph、MTP、量化、并行优化、负载均衡等 ②100%基础功能支持，详见基础功能标签页 ③新发布特性开箱支持 |
| 加速库（含三方） | 计算：@xiaomingbao008 2012：@jerry-lilijun                | ①北向兼容支持主流加速库，框架侧零适配修改对接，如deepep、mooncake ②其他推理引擎能力复用                                        |
| 算子       | 计算：@RuixuanZhang06 图灵：@ZhengdQin                      | ①融合算子泛化，性能开箱即优 ②自定义算子贡献，发展开发者生态                                                             |

## PMC及运营阵型

为更好地发挥一盘棋作用，确保统筹规划推理引擎的生态发展和能力共享，SGLang项目横队的PMC组织将会联合vllm、mindie一起运作，具体的运作规则及使命，待跨横队对齐后再统一公示。

运作规则：
1、轻量化运作：每月初和版本例会同步召开，同步当月进展和规划，每双周召开一次生态运营例会，同步进展和风险

2、紧急诉求或建议：生态及社区接口人：@huxiaoman7 ；PMC接口人：@xueyao00625566、@Right2Left

内源社区生态建设及社区运营整体责任人：@huxiaoman7，具体关键角色及职责如下：

| 子领域    | 责任人         | 责任                                                                               |
| ------ | ----------- | -------------------------------------------------------------------------------- |
| 基础设施运营 | @jlcoo  | ①社区基础设施工程搭建及接入、门禁搭建<br>②发布工程搭建及接入 <br>③公司内部推理基础设施部署试运行及推广（Hidev、AI for coding） |
| 版本发布 | @Right2Left | ①按照内部对齐基线，发布RoadMap至社区<br> ②Docker镜像每日Dev可用，发布节奏与社区匹配                           |
| 项目拓展及支持 | @raintBN-91 | ①头部客户/ISV联合解决方案规划、设计、开发、验证；重大市场项目支持（拓展、PoC）|
| 社区运营 | @zoezhangMS  | ①项目运营推广、开发者拓展等<br>②社区流程运作支撑<br>③三方社区合作，如verl等         |
| 高校合作及赋能   | @raintBN-91  |①高校合作、与相关高校教授联合开展课题<br> ②课程赋能：设计课程赋能资料   |
| 社区运维   | @zhaoming   | ①制定SLA响应矩阵，保证24h响应<br> ②收集github、gitcode等社区的问题，处理社区日常issue，快速响应，解决客户问题   |
| 资料运营   | @shentong   | ①内源主页、社区主页宣传运营<br> ②部署、配套资料社区发布、运营    |


## 社区roadmap

当前基于业务范围，已完成H2重点规划的优化特性及任务，可以持续关注主社区的roadmap说明：

[[Roadmap\] Supporting Ascend NPU on 2025 H2 · Issue #8004 · sgl-project/sglang](https://github.com/sgl-project/sglang/issues/8004)

## 学习资料

[SGLang Documentation](https://docs.sglang.ai/)

## 如何反馈

如果你在检视、体验、使用、参与sglang的过程中，存在任何困惑或问题，欢迎直接基于主社区提交 Issue，并标识Issue类型。

- [我要提交问题、需求](https://github.com/sgl-project/sglang/issues)
- [我要参与内源社区公开会议]()

## CI

SGLang的CI基于[github action](https://docs.github.com/en/actions)搭建，所有的CI，每日测试和版本构建发布都在[workflows](./workflows)目录中

### CI是如何运行的

请参考github action [文档](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#on)，CI通过在github action的yaml中指定特定的事件实现任务的触发

#### 如何选择CI运行的环境

当前SGLang CI提供了如下几种[self hosted runner](https://docs.github.com/en/actions/reference/runners/self-hosted-runners)

可以通过指定`workflows目录里的yaml`里的 `runs-on`字段来明确CI的运行环境

| runs-on的值  |  环境描述 |
| ------ | ----------- |
| linux-aarch64-a3-1 | 单卡的A3环境 |
| linux-aarch64-a3-2 | 2卡的A3环境 |
| linux-aarch64-a3-4 | 4卡的A3环境 |
| linux-aarch64-a3-4 | 8卡的A3环境 |
| linux-aarch64-a2b4-1 | 1卡的910B4环境(32G显存) |
| linux-aarch64-a2b4-2 | 2卡的910B4环境(32G显存) |
| linux-aarch64-a2b4-3 | 3卡的910B4环境(32G显存) |
| linux-aarch64-a2b4-4 | 4卡的910B4环境(32G显存) |
| self-hosted-x86 | 一台8C16G的x86设备，主要用于构建docker image |
| self-hosted-arm64 | 一台8C16G的arm64设备，主要用于构建docker image |

### CI运行过程中，模型是存储在哪里的

SGLang所有带NPU的环境，都在 `/root/.cache` 目录下挂载了共享存储，用于存放从[modelscope](https://modelscope.cn)下载的模型。由于目录结构和modelscope的默认缓存目录一致，因此在设置了`SGLANG_USE_MODELSCOPE`时，可以自动使用这里已经缓存的模型


### CI的模型/缓存目录怎么新增模型

本仓库通过 一个定期下载模型和数据集的[workflow](./workflows/sync-models-datasets.yml) 来实现模型和数据集的自动下载

可以通过修改[pre-downloaded-datasets](./workflows/pre-downloaded-datasets.ini)和[pre-downloaded-models](./workflows/pre-downloaded-models.ini)

当前的触发节奏是6小时，当修改对应的ini文件的PR合入时，也会触发

### CI构建发布的镜像从哪里可以下载

每日构建的镜像会发布到华为云SWR香港和贵阳站点：
```bash
# 贵阳节点，适合国内用户
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/sglang:main

# 香港站点，适合海外用户
docker pull swr.ap-southeast-1.myhuaweicloud.com/base_image/ascend-ci/sglang:main
```

### 镜像的tag原则

当前docker 镜像的tag的原则可以参考[这里](./workflows/release-docker-ascend-nightly-innersource.yml)

每天会发布a3和910b的镜像

### 内源仓workflow编写规则

- 为了减少本仓库和[sglang上游](https://github.com/sgl-project/sglang)的冲突，建议本仓库的workflow不要覆盖已经存在的非上游`workflow`
- 新增的内源workflow以`-innersource.yaml`作为结尾
- 内源workflow中，建议统一增加`if: github.repository == 'Ascend/sglang' `的判断，避免在您fork的仓库里运行

### CI任务常见异常

Q：期望增加新的硬件类型
A：可以联系基础设施接口人

Q：CI一直处于pending（黄色未转圈）状态
A：可能是NPU资源满载了，如果超过1小时还未启动，可以联系基础设施接口人

Q：CI测试运行失败
A：可以联系对应的开发责任人定位

Q：docker image构建失败
A：可以联系社区运维处理
