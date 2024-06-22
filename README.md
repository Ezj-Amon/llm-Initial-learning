# llm-Initial-learning

## LLM介绍
LLM 通常指包含**数百亿（或更多）参数的语言模型**，它们在海量的文本数据上进行训练，从而获得对语言深层次的理解。  
### 常见的LLM模型
> 闭源：GPT系列，CLAUDE系列，PaLM/Gemini 系列以及国内的一系列大模型  
> 开源：LLaMA 系列，通义千问，GLM 系列，Baichuan 系列

### RAG
RAG 是一个完整的系统，其工作流程可以简单地分为数据处理、检索、增强和生成四个阶段

### LangChain
**LangChain 框架是一个开源工具，充分利用了大型语言模型的强大能力，以便开发各种下游应用。它的目标是为各种大型语言模型应用提供通用接口，从而简化应用程序的开发流程**。具体来说，LangChain 框架可以实现数据感知和环境互动，也就是说，它能够让语言模型与其他数据来源连接，并且允许语言模型与其所处的环境进行互动。

### LLM项目的一般流程
> 步骤一：项目规划与需求分析  
> 确定项目目标和核心功能；确定技术架构和工具  
> 步骤二：数据准备与向量知识库构建  
> 如：加载本地文档 -> 读取文本 -> 文本分割 -> 文本向量化 -> question 向量化 -> 在文本向量中匹配出与问句向量最相似的 top k 个 -> 匹配出的文本作为上下文和问题一起添加到 Prompt 中 -> 提交给 LLM 生成回答。 
> 步骤三：大模型集成与 API 连接  
> 步骤四：核心功能实现  
> 步骤五：核心功能迭代优化  
> 步骤六：前端与用户交互界面开发  
> 步骤七：部署测试与上线  
> 步骤八：维护与持续改进  



## 使用LLM API开发应用

### 基本概念

#### Prompt（写一个prompt给大模型用于不断优化prompt？）

> 一种输入模板，可以减少模型幻觉，提高模型的输出质量

一个完整的Prompt包括以下几个关键要素：

1. **明确目标**：清晰定义任务，以便模型理解。
2. **具体指导**：给予模型明确的指导和约束。
3. **简洁明了**：使用简练、清晰的语言表达Prompt。
4. **适当引导**：通过示例或问题边界引导模型。
5. **迭代优化**：根据输出结果，持续调整和优化Prompt。

一些有效做法：

- 强调，可以适当的重复命令和操作。
- 给模型一个出路，如果模型可能无法完成,告诉它说“不知道”，别让它乱“联想”。
- 尽量具体，它还是孩子，少留解读空间。

![img](./md_images/README/6ab15bfb37644ee8be73cd000051d4bb.png)

**另**：

> [!NOTE]
>
> 1. **System Prompt** : 该种 Prompt 内容会在整个会话过程中持久地影响模型的回复，且相比于普通 Prompt 具有更高的重要性
> 2. **User Prompt**:  平常的Prompt

```json
{
    "system prompt": "你是一个幽默风趣的个人知识库助手，可以根据给定的知识库内容回答用户的提问，注意，你的回答风格应是幽默风趣的",
    "user prompt": "我今天有什么事务？"
}
```



#### Temperature

> 通过控制 temperature 参数来控制 LLM 生成结果的随机性与创造性。
>
> Temperature 一般取值在 0~1 之间，当取值较低接近 **0** 时，预测的随机性会较低，产生更**保守**、可预测的文本，不太可能生成意想不到或不寻常的词。当取值较高接近 **1** 时，预测的随机性会**较高**，所有词被选择的可能性更大，会产生更有**创意**、多样化的文本，更有可能生成不寻常或意想不到的词。



### API开发

####  智谱API（送的太多了）

```python
from zhipuai import ZhipuAI

client = ZhipuAI(
    api_key=os.environ["ZHIPUAI_API_KEY"]
)

def gen_glm_params(prompt):
    '''
    构造 GLM 模型请求参数 messages

    请求参数：
        prompt: 对应的用户提示词
    '''
    messages = [{"role": "user", "content": prompt}]
    return messages


def get_completion(prompt, model="glm-3-turbo", temperature=0.95):

    messages = gen_glm_params(prompt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"
```