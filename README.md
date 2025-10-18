# AgentLightning 示例项目集合

这个仓库包含了多个使用 AgentLightning 框架构建的示例项目，展示了如何利用该框架解决不同类型的任务，包括简单的数学问题求解、复杂的 SQL 查询生成和执行，以及实验跟踪等。

## 项目概览

AgentLightning 是一个用于构建和训练基于大型语言模型（LLM）的智能代理的框架。本仓库提供了几个示例，演示如何使用该框架处理不同类型的挑战：

1. **Simple Math Agent** - 展示如何使用强化学习训练一个能够解决基本算术问题的代理
2. **Spider SQL Agent** - 实现了一个可以与 SQL 数据库交互、生成和执行 SQL 查询的智能代理
3. **Weights & Biases 集成示例** - 演示如何在 AgentLightning 项目中集成实验跟踪工具

## 技术栈与架构

### 主要依赖

- **AgentLightning** - 核心框架，用于构建和训练智能代理
- **LangChain/LangGraph** - 用于构建复杂的语言模型应用和代理工作流（在 Spider 示例中使用）
- **OpenAI API** - 作为基础语言模型提供商
- **Weights & Biases / Weave** - 用于实验跟踪和模型评估
- **SQLite/SQLDatabase** - 在 Spider 示例中用于数据库交互

### 项目结构

```
agentlightning-demo/
├── simple_math/           # 简单数学问题求解代理
│   ├── README.md          # 数学代理详细文档
│   ├── simple_math_agent.py  # 核心实现代理逻辑
│   ├── generate_train_data.py # 训练数据生成
│   └── train_simple_math.sh   # 训练脚本
├── spider/                # SQL 查询生成和执行代理
│   ├── README.md          # Spider 示例文档
│   ├── sql_agent.py       # 核心实现代理逻辑
│   ├── spider_eval/       # 评估工具
│   ├── train.sh           # 训练脚本
│   ├── train_ci.sh        # CI 训练脚本
│   └── train_llama.sh     # Llama 模型训练脚本
└── wandb/                 # 实验跟踪示例
    ├── wandb_demo.py      # Weights & Biases 基础示例
    └── wandb_demo2.py     # Weave 高级示例
```

## 快速上手指南

### 前提条件

- Python 3.8 或更高版本
- OpenAI API 密钥（对于需要 LLM 的示例）
- 对于 Spider 示例：至少 40GB 内存的 GPU

### 安装步骤

1. 克隆仓库：
```bash
git clone <repository-url>
cd agentlightning-demo
```

2. 安装 AgentLightning 框架（根据官方文档）

3. 设置环境变量：
```bash
export OPENAI_API_KEY=your_openai_api_key
```

### 配置说明

根据不同示例的需求，可能需要设置不同的环境变量：

- `OPENAI_API_KEY` - OpenAI API 密钥
- `OPENAI_API_BASE` - 可选，自定义 API 端点
- `VERL_API_BASE` - 用于 Spider 示例的 VERL API 端点
- `WANDB_API_KEY` - Weights & Biases API 密钥（如果使用实验跟踪）

## 核心功能与特性

### 1. Simple Math Agent（简单数学代理）

一个轻量级的数学问题求解代理，展示了强化学习与 LLM 结合的基本概念：

```python
agent = SimpleMathAgent(
    trained_agents="math_solver",
    max_attempts=3,              # 每个问题的最大尝试次数
    optimization_algorithm="ppo", # 优化算法 ("ppo" 或 "grpo")
    reward_function="accuracy"   # 奖励函数 ("accuracy" 或 "steps")
)
```

**主要特点：**
- 支持多种优化算法（PPO、GRPO）
- 可配置的奖励函数（基于准确性或步骤数）
- 自我改进机制，从先前尝试中学习
- 模块化设计，易于扩展到其他类型问题

### 2. Spider SQL Agent（SQL 查询代理）

一个复杂的 SQL 查询生成和执行代理，基于 LangChain 和 LangGraph 构建：

```python
# 初始化 SQL 代理
sql_agent = LitSQLAgent(
    db=db,  # SQL 数据库连接
    llm=llm  # 语言模型
)
```

**主要特点：**
- 自动生成 SQL 查询以回答自然语言问题
- 执行生成的查询并验证结果
- 根据反馈重写查询
- 使用 Spider 数据集进行训练和评估

### 3. 实验跟踪集成

集成 Weights & Biases 和 Weave 进行实验跟踪：

```python
import wandb
# 初始化新的运行以跟踪脚本
wandb.init(
    project="example-test",
    name=f"experiment",
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    })
```

## 贡献指南与未来计划

### 贡献流程

我们欢迎社区贡献！要贡献代码，请遵循以下步骤：

1. Fork 仓库
2. 创建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 未来路线图

- 增加更多类型的代理示例（如代码生成、数据分析等）
- 改进现有的奖励函数和优化算法
- 添加更多评估指标和可视化工具
- 扩展对不同 LLM 提供商的支持
- 增强实验跟踪和模型管理功能

## 许可证

该项目基于 MIT 许可证发布 - 查看 [LICENSE](LICENSE) 文件了解详情。