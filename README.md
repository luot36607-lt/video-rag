# 视频精准摘要与智能问答

把一段视频变成**可摘要、可检索、可追问、可跳转**的知识库：自动转写、纠错、递归分块、
按章节生成带时间戳的摘要；提问时返回带时间戳的答案，点一下就能跳到视频里那句话的位置。

解决的痛点很具体：**一场直播或一节课两三个小时，你只想看其中十分钟，但不知道那十分钟在哪。**

---

## 效果

在一段 9 分 35 秒的网课视频上实测（CPU，whisper-small）：

| 环节 | 结果 |
|---|---|
| 转写 | 6 分 51 秒（约 0.7 倍实时），289 段 |
| 纠错 | 词典 86 处 + LLM 修正 55 段 |
| 分块 | 289 句 → 27 段 → 5 章 → 13 块，平均 299 字 / 46 秒 |
| 摘要 | 4 章 / 20 条要点，**20/20 精确定位到原句** |
| 摘要评测 | 覆盖率 3/3 · 可定位率 20/20 · 压缩比 16.4% |
| 检索 | ~500ms（含一次 embedding） |
| 生成 | ~1.6s，约 700 tokens |

问答示例（回答里的 `[1]` 可点击跳转）：

```
问：LangChain 怎么调用通义千问？

答：1. 从 langchain_community 导入通义类 [1][2]
    2. 创建对象，传 model 参数 [1][3]
    3. 调用 invoke 方法 [1]

[1] 01:44 - 02:29   相似度 0.685   ✓ 已引用
[2] 03:12 - 04:02   相似度 0.646   ✓ 已引用
[3] 04:00 - 04:28   相似度 0.635   ✓ 已引用
```

问视频里没有的内容，会直接拒答，且来源卡片不会被标记为已引用。

---

## 架构

```
视频 ──ffmpeg──▶ 音频 ──faster-whisper──▶ 带时间戳的句子
                                              │
                                              ▼
                                    ASR 纠错（词典 + LLM）
                                              │
                                              ▼
                          递归分块：章节 → 段落 → 句子 → 字符
                                              │
                                              ▼
                                  块 = {text, start, end, chapter}
                                              │
                                              ▼
                                     向量索引（bge-m3）
                                              │
                        提问 ──▶ 检索 top-3 ──▶ 生成（只输出引用编号）
                                              │
                                              ▼
                        回答 + [n] ──▶ 程序映射回时间戳 ──▶ 点击跳转
```

---

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env        # 填两个 key
```

把视频放进 `data/`（或设置 `VIDEO_PATH`），然后依次跑四步：

```powershell
# 1. 转写（首次会下载模型，国内走 hf-mirror 镜像）
.venv\Scripts\python.exe video_transcribe.py "data\video.mp4" --id demo

# 2. 纠错（--llm 会用上下文再修一轮同音字）
.venv\Scripts\python.exe asr_fix.py "transcripts\demo.json" --llm

# 3. 递归分块（--semantic 用 embedding 找话题边界）
.venv\Scripts\python.exe chunk_recursive.py "transcripts\demo.fixed.json" --semantic --id demo --title "演示视频"

# 4. 生成分段摘要（--transcript 用来把要点对回精确时间）
.venv\Scripts\python.exe video_summary.py chunks\demo.json --transcript transcripts\demo.fixed.json --md

# 5. 起服务
.venv\Scripts\python.exe -m uvicorn video_api:app --reload --port 8002
```

---

## 多视频

一个视频一个 ID，文件按 ID 存放：

```
chunks/langchain-llm.json      summaries/langchain-llm.json
chunks/ollama-basics.json      summaries/ollama-basics.json
```

再加一个视频，就是把上面四步的 `--id` 换掉再跑一遍：

```powershell
.venv\Scripts\python.exe video_transcribe.py "data\第二个.mp4" --id ollama-basics
.venv\Scripts\python.exe asr_fix.py "transcripts\ollama-basics.json" --llm
.venv\Scripts\python.exe chunk_recursive.py "transcripts\ollama-basics.fixed.json" --semantic --id ollama-basics --title "Ollama 基础"
.venv\Scripts\python.exe video_summary.py chunks\ollama-basics.json --transcript transcripts\ollama-basics.fixed.json
```

**检索是全局的**：提问会搜所有视频，结果里标注来自哪个视频；点来源会自动切换到那个视频并跳到时间点。
页面右上角的下拉框可以手动切换。

打开 http://127.0.0.1:8002/ ，左边播放器，下面提问，点来源跳转。

只想在命令行里问：

```powershell
.venv\Scripts\python.exe video_ask.py chunks\video.fixed.json --q "讲了什么？" --min-score 0.55
```

---

## 四个关键设计

**1. 时间戳不由模型生成**

模型只输出 `[1] [2]` 这样的引用编号，编号到 `start/end` 的映射由程序完成。
所以时间戳永远不会被编造——这是「可解释」能成立的前提。

**2. 递归分块，不在句子中间下刀**

按 章节 → 段落 → 句子 → 字符 逐级拆，块只在段落边界上合并，跨块重叠一句。
转写里没有章节标题，所以「章节」用相邻段落的 embedding 相似度检测——相似度明显下降的地方就是换话题。

**3. 相似度阈值拒答**

实测能回答的问题 top1 相似度在 0.63~0.75，不能回答的在 0.45~0.49。
低于阈值直接拒答，不调用模型：省钱，也不给它硬编的机会。
**这个阈值要在自己的语料上重新标定，不是通用常数。**

**4. 摘要的每条要点都能回到原句**

模型要为每个要点摘录一句原文（`quote`），程序拿这句去转写里精确匹配——
命中就是那句话真正说出来的时刻；匹配不上才退回整块开头，并在前端标 `≈`。
这样「精准摘要」不是形容词：20 条要点，20 条能点回原句。

---

## 摘要评测

| 指标 | 含义 | 结果 |
|---|---|---|
| 覆盖率 | 人工参考要点被覆盖的比例 | 2/3 |
| 可定位率 | 要点能精确回到原句的比例 | 20/20 |
| 压缩比 | 摘要字数 / 转写字数 | 17.1% |

```powershell
.venv\Scripts\python.exe video_eval.py --summary summaries\semantic.json --ref eval\summary_ref.json --transcript transcripts\video.fixed.json
```

参考要点在 `eval/summary_ref.json`，由人写，不由模型代劳。

第一次跑是 **2/3**。没覆盖的那条是「不用 qwen3-max 的原因」，原因不在摘要模型——
转写把两个模型名都识别成了 qwen-max：

> 我们不用 qwen-max …… 因为 qwen-max 是聊天模型
> 然后我们用的 qwen-max 是大语言模型

一句话自相矛盾，摘要只能选择不提。加了一条上下文相关的纠错规则
（`我们不用qwen-max → 我们不用qwen3-max`）后重跑，**3/3**。

**ASR 的错误会一路传导到摘要质量**，这是为什么术语纠错不是做一次就完事。

---

## 踩过的坑

**1. ASR 的专有名词是成片错的，不是随机的**

同一段视频里，LangChain 被识别成「浪欠 / Lunchin」，Ollama 成了「欧拉马」，
qwen-max 成了「q1-max」，向量成了「象量」。

给 Whisper 加领域提示词（`initial_prompt`）能修好开头，但**越往后越漂**——它按 30 秒窗口解码，
提示词的约束会衰减。所以必须做后处理：词典保底 + LLM 上下文纠错。

**2. 视频里没有「章节标题」**

纯口语转写没有结构。用固定字数硬切不是语义分块，所以改成用相邻段落的相似度找边界。
局限也很明显：如果前四分钟一直在讲同一个主题，它就切不开。

**3. 视频跳转不是设个 currentTime 就完事**

- 服务端必须支持 `Range` 请求（返回 206），否则拖不动进度条
- 元数据没加载完时设 `currentTime` 会被忽略
- 视频元素一旦进入 error 状态（比如服务重启过），再设 `currentTime` 和 `play()` 完全没反应——需要先重设 src 再 `load()`

**4. 版权**

视频、音频、转写文本都不进仓库（`.gitignore` 已覆盖 `transcripts/`、`chunks/`、`*.mp4`）。
换素材时请确认你有使用权。

**5. 引用匹配到了错的那一次**

同一个意思在视频里会出现多次——开头预告一次，后面实操一次。一开始是全局搜索第一个匹配，
结果「访问 Ollama 的方法」被定位到 02:43 的预告，而不是 06:56 的实操。

改成**只在当前章节的时间窗口里找**，才对上。这类错误比“编造时间戳”更隐蔽：
时间戳是真实存在的，只是指向了错的地方。

---

## 目录结构

```
video_transcribe.py   视频 → 音频 → 带时间戳的文本
asr_fix.py            词典 + LLM 纠错
chunk_recursive.py    递归分块（章节/段落/句子）
video_summary.py      分段摘要（要点 + 原文引用 + 精确时间）
video_eval.py         摘要评测（覆盖率 / 可定位率 / 压缩比）
video_ask.py          命令行问答
video_api.py          FastAPI 服务（多视频、Range 视频流）
common.py             embedding / 向量索引 / 模型配置
library.py            多视频索引：扫描 chunks/ 拼成检索集合
web/video.html        播放器 + 问答界面
data/                 放视频（不进仓库）
transcripts/          转写结果（不进仓库）
chunks/               分块结果（不进仓库）
summaries/            摘要结果（不进仓库）
eval/                 人工参考摘要 + 评测结果
```

---

## 已知限制

- **只处理音频**，不看画面。PPT 翻页、板书、代码演示都没利用
- **单视频**：一次只能索引一个视频，没有跨视频检索
- **章节检测依赖相似度**，遇到单主题长段落会切不准
- **CPU 转写慢**：1 小时视频约需 45 分钟，做长视频要上 GPU 或云 ASR
- **无并发控制**：转写是同步的，多人同时用会排队

---

## 下一步

1. **画面理解**：抽关键帧 + OCR，把 PPT 内容也纳入检索
2. **异步任务**：转写放到队列里，前端显示进度
3. **数据库**：现在每个视频一个 JSON，多了以后需要按视频过滤、增量更新、并发写——
   到那时再上 Postgres + pgvector，而不是现在
4. **摘要评测**：现在只验证了「要点能定位到原句」，还没验证「要点是否说全了」——需要人工标注参考摘要，算覆盖率和忠实度
