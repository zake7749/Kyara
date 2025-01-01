# Kyara: Knowledge Yielding Adaptive Retrieval Augmentation for LLM Fine-tuning

[![DOI](https://zenodo.org/badge/844304447.svg)](https://zenodo.org/badge/latestdoi/844304447)

<p align="left">
    🤗 <a href="https://huggingface.co/collections/zake7749/kyara-673e129dbf93bea9ca87abeb">Hugging Face</a>&nbsp ｜ 🚀<a href="https://github.com/zake7749/kyara">Github</a>&nbsp ｜ &nbsp📑 <a href="#">Paper</a>&nbsp ｜ &nbsp📖 <a href="https://github.com/zake7749/kyara/blob/main/document/README_EN.md">English</a>&nbsp ｜ &nbsp📖 <a href="https://github.com/zake7749/kyara">Chinese</a>&nbsp; ｜ &nbsp;💻 <a href="https://www.kaggle.com/code/zake7749/kyara-a-compact-yet-powerful-chinese-llm">Kaggle Notebook</a>
</p>
<div style="text-align: center;">
  <img src="https://i.imgur.com/QiWlcYJ.jpeg" alt="kyara"/>
</div>

Kyara 是一個實驗性專案，旨在透過階段性的知識檢索產生合成資料，以增強語言模型的知識範圍與語言理解能力。目前，Kyara 的重心在於填補中文語料庫，尤其是繁體中文領域的空缺。與現今大量且多樣的英文語料相比，中文語料相對匱乏，這在語言模型的訓練與應用上形同一道難以逾越的高牆，限制了中文語言模型的發展潛力。

為了驗證 Kyara 的有效性，我們對 `Gemma-2-2b-it` 等多個模型進行了全參數微調，產生了首版的 Kyara 模型。初步評估結果可參考 [Benchmark](#benchmark)，Kyara 在多個中英資料集中均優於基底模型，並在繁體中文的評估上取得了顯著的提升。

## Table of Contents

- [Benchmark](#benchmark)
- [Usage](#usage)
- [Method](#method)
- [Feature](#feature)
- [Limitation](#limitation)

## Benchmark

### General Benchmark

| Metric                   | Kyara-2b-it    | Gemma-2-2b-it |  Kyara-9b-it   |  Gemma-2-9b-it     |
|--------------------------|----------|-------------|-------------|-------------|
| **[TMMLUPlus](https://huggingface.co/datasets/ikala/tmmluplus)**            | **41.98** | 36.73    | **59.87** | 54.77 |
| &emsp;- STEM               | **43.73**   | 37.84      | **66.98**   | 58.12      |
| &emsp;- Humanities         | **38.72**   | 33.40      | **54.42**   | 48.71      |
| &emsp;- Other              | **40.61**   | 36.00      | **55.26**   | 51.43      |
| &emsp;- Social-Science     | **44.88**   | 39.69      | **62.81**   | 60.84      |
| **[MMLU-Redux](https://github.com/yuchenlin/ZeroEval)**    | **55.44**| 51.94       | 72.57| **72.82**    |
| **[GSM8K](https://github.com/yuchenlin/ZeroEval)**         | **54.21**| 51.63       | **90.60**| 87.41     |
| **[MATH-L5](https://github.com/yuchenlin/ZeroEval)**       | **8.88**| 4.3       | **28.71**|  19.42      |
| **[CRUX](https://github.com/yuchenlin/ZeroEval)**          | **22.75**| 21.5     | **48.62**| 46.00     |
| **[ZebraLogic](https://github.com/yuchenlin/ZeroEval)**    | **5.2**| 4.2  | **16.8** |   12.8    | 
| **Chinese-Reason-Bench** | **4.21** | 3.44        |   **6.13**   |     4.96      |

上述評測皆採 zero-shot 的方式進行評估。TMMLUPlus 分數聚合的策略同官方設計 (macro-average)。

#### [Open-LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

目前 Kyara-2b-it 是 Open-LLM Leaderboard 上綜合分數排名最高的 2B 模型。

<div style="text-align: center">
  <img src="https://i.imgur.com/3NKhAja.png" alt="kyara-2b-it-open-llm-leaderboard">
</div>

### Alignment Benchmark

| Metric                   | Kyara-2b-it    | Gemma-2-2b-it | ChatGPT-3.5-1106 |  Kyara-9b-it   |  Gemma-2-9b-it     |
|--------------------------|----------------|---------------|------------------|----------------|--------------------|
| **[AlpacaEval-LC](https://github.com/tatsu-lab/alpaca_eval)**                   | **35.35** | 32.37       | 19.30    | -        |   -    |
| **[AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)**                      | **43.34** | 32.94       | 9.20     | -        |   -    |
| **[Chatbot-Arena-Hard](https://github.com/lm-sys/arena-hard-auto)**             | **22.60** | 19.4        | 18.87    | **41.1** |  33.6  |
| **[MT-Bench-TW](https://huggingface.co/datasets/MediaTek-Research/TCEval-v2)**  | **7.43**  | 6.35        | 7.10     | **8.36** |  7.80  |
| **[MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench)**                    | 8.28      | 8.17        | 8.32     | **8.81** |  8.53  |

#### [AlignBench](https://github.com/THUDM/AlignBench)

| Fold  | Kyara-2b-it-CHT | Kyara-2b-it-CHS | Gemma-2-2b-it |  ChatGPT-3.5-0613   |
|-------|-----------------|-----------------|---------------| ----       |
| 基本任務   | 6.72     | 6.54 | 6.42 | **6.92** |
| 中文理解	 | 5.78     | 5.24 | 5.03 | **5.91** |
| 綜合問答	 | **8.16** | 7.79 | 7.52 | 6.47     |
| 文本寫作	 | **7.90** | 7.24 | 7.76 | 7.28     |
| 邏輯推理	 | **5.26** | 4.27 | 4.20 | 4.79     |
| 數學計算	 | **5.99** | 5.44 | 5.05 | 5.38     |
| 角色扮演	 | **8.07** | 8.00 | 7.42 | 7.00     |
| 專業能力	 | **6.97** | 6.86 | 5.79 | 6.81     |
| 推理總分   | **5.62** | 4.85 | 4.63 | 5.00     |
| 語言總分   | **7.26** | 6.94 | 6.66 | 6.73     |
| 總分      | **6.44** | 5.90 | 5.64 | 5.91     |

這裡的 CHT 和 CHS 分別代表以繁體中文評估，或是以簡體中文評估。為了能在 AlignBench 上評估繁體中文的表現，我們使用 [OpenCC](https://github.com/BYVoid/OpenCC) 配合 config `s2tw` 進行了簡易的簡繁轉換，將所有問題從簡體中文轉成了繁體中文。

## Usage

Kyara 採用了和 Gemma2 一樣的架構，因此在推理上可以沿用 [Google 的官方教學]([https://huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it#usage))。此外，我們也在 Kaggle 上提供了一個 [Jupyter Notebook](https://www.kaggle.com/code/zake7749/kyara-a-compact-yet-powerful-chinese-llm) ，演示 Kyara 的各項基本功能，如寫作、摘要、開放式問答、數學計算以及 RAG 等各種情境。

### Running with the `pipeline` API

在安裝完 `transformers` 後，可用如下方式和模型互動： 

```
from transformers import pipeline

pipe = pipeline("text-generation", model="zake7749/gemma-2-2b-it-chinese-kyara-dpo",)
messages = [{"role": "user", "content": "你認為科技發展對於未來的工作形態會產生什麼樣的影響？為什麼？"},]
result = pipe(messages, temperature=0.5, max_length=1024)

print(result)

# 科技發展對於未來的工作形態將產生深遠的影響，主要體現在以下幾個方面：\n\n1. **自動化與人工智慧**：\n   - **工作取代與轉型**：許多重複性高、規律性的工作（如製造業、客服等）可能會被自動化技術取代。這將促使企業重新考量工作性質，更多地依賴於需要創造力、情感智能和複雜決策的職位。\n   - **新興職位**：隨著自動化技術的普及，會出現新的職位需求，例如數據分析師、機器學習工程師和AI倫理專家等。\n\n2. **遠程工作與靈活工作模式**：\n   - **地理限制減少**：科技使得遠程工作成為可能，企業可以從全球範圍內招聘人才，這將改變傳統的工作模式，促進多元化和包容性。\n   - **彈性工作時間**：員工將有更大的彈性來安排自己的工作時間，這可能提高工作滿意度和生產力。\n\n3. **協作工具與平台**：\n   - **增強協作**：科技將促進團隊間的協作，使用雲端工具和即時通訊軟體，使跨地域的團隊能夠高效合作。\n   - **虛擬現實與增強現實**：這些技術可能改變會議和培訓的方式，提供更具沉浸感的體驗。\n\n4. **持續學習與技能提升**：\n   - **終身學習的必要性**：隨著技術的快速變化，員工需要不斷更新自己的技能，這將推動在線學習和終身學習的模式。\n   - **個性化學習路徑**：科技將提供更靈活的學習方式，根據個人的需求和興趣量身定制學習內容。\n\n5. **數據驅動的決策**：\n   - **決策過程的變革**：企業將越來越多地依賴數據分析來指導業務決策，這將改變傳統的管理和運營方式。\n\n6. **工作與生活的平衡**：\n   - **重新定義工作與生活**：科技的發展可能會影響人們對工作的看法，使人們更加重視工作與生活的平衡，從而可能減少加班和提高生活質量。\n\n總體而言，科技的發展將促使工作形態變得更加靈活、多樣化和以人為本，但也要求勞動者具備適應新技術和持續學習的能力。企業和個人都需要積極應對這些變化，以抓住新機會並減少潛在的挑戰。
```

## Method

### Dataset Summary

我們一共合成了 3,625,593 則對話，約 4.51B tokens. 以下是根據資料源統計的語言分佈以及資料輪次的概要：

* 資料源語言分佈：

<img src="https://i.imgur.com/fhD5kIy.png" alt="language-distribution" width="500"/>

* 對話輪次分佈：

<img src="https://i.imgur.com/CWQ2shj.png" alt="conv-round-distribution" width="500"/>

### Dataset Construction

Kyara 的資料構建分為中英兩部分，英文的部分我們混用了眾多高品質的英文資料集如 `teknium/OpenHermes-2.5`、`arcee-ai/The-Tome` 並進行語意去重。而中文的部分則是透過以下流程組建：

#### Base Dataset: Knowledge Injection with Retrieval Augmentation

我們基於開放中文知識型語料庫如 [bigscience-data-wikipedia](https://huggingface.co/datasets/bigscience-data/roots_zh-tw_wikipedia) 搭配 [QDrant](https://qdrant.tech/) 構建了一個知識搜尋系統，並透過以下流程構建 SFT Pairs:

1. 迭代知識庫中的文本，針對每篇文本生成使用者可能詢問的知識密集型問題。
2. (可選) 基於 [Evol-Instruct](https://arxiv.org/pdf/2304.12244) 使指令複雜化，複雜化方向如：
    * 追問多個相關問題。
    * 針對原始問題進行逐步分析。
    * 要求特定的結構化輸出，如 JSON / Table。
    * 添加情境假設或背景故事，請模型在該假設下回答問題。
3. 對於生成出的問題進行 Query Expansion，額外召回 Top K 篇相關文件，並個別判斷這些文件是否與原始任務相關：
    * 如果相關，請 LLM 針對問題摘要出重點資訊。
    * 如果無關，則忽略該份文件
4. 請 LLM 閱讀原始文件以及至多 K 份重點資訊，撰寫詳盡且完整的回答。

除此之外，針對高品質文本，我們會直接請 LLM 猜測可以透過哪些 prompt 產出這份文本，並將猜測出的 prompt 配合文本組成 SFT Pair。

##### Chinese Math Dataset

* 資料集：[zake7749/kyara-chinese-math-sft-s0-30K](https://huggingface.co/datasets/zake7749/kyara-chinese-math-sft-s0-30K)

上述策略雖然能廣泛生成知識性文本，但主要屬於資訊尋求（Information Seeking）的任務範疇，不太能有效構建數學與推理相關的語料。為此，我們基於 [PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub) 產生了 5 萬則數學題，並借助 `Gemini-1.5-Flash` 濾除在計算與推理思維上明顯出錯的資料，以此構建出 [zake7749/kyara-chinese-math-sft-s0-30K](https://huggingface.co/datasets/zake7749/kyara-chinese-math-sft-s0-30K)，並將其混入 Base dataset 中進行第一階段的指令微調。

#### High Quality Dataset: Model Refinement 

在使用 Base Dataset 完成指令微調後，我們會再將訓練好的模型微調於一個高品質 Subset 上，主要是為了處理以下三個問題：

1. Base Dataset 中有些回應是由小型 LLM 生成的，有時在指令跟隨上表現不佳。
2. 在實驗初期，我們混用了不同 LLM 以擴充知識多樣性和語言適應性。然而，在後續的評估中有發現不同 LLM 的 response template 以及行文思維有些微妙的差異，導致訓練後的 Chat Model 不太穩定。因此，我們引入了一個高品質的小型資料集，採用同一個 LLM 生成 QA Pairs。
3. Base Dataset 中包含部分給定原文並要求模型猜測 User Queries 所組成的 Q&A Pairs，這些資料點雖知識豐富，但在指令跟隨上相對弱勢。

為兼顧資料多樣性與品質，我們採用了類似 [InsTag](https://arxiv.org/abs/2308.07074) 的策略將資料分門別類，並使用 [ArmoRM](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) 以及一個 LLM Judge 來評估資料品質。最後，只抽取各類別中品質優良的訓練資料，構建出 500K 的 Dataset，並將其用於再次微調 Kyara-SFT Model。

### Preference Learning

除了監督式學習，我們也引入了偏好學習（Preference Learning），讓模型的回答更符合人類偏好，同時加強程式能力與數學推理能力。

Kyara 的偏好學習策略採用了 [Direct Preference Optimization (DPO)](https://arxiv.org/pdf/2305.18290)，我們混用了兩個自製的中文資料集以及以下三個英文資料集：

* [argilla/ultrafeedback-binarized-preferences](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences)
* [xinlai/Math-Step-DPO-10K](https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K)
* [BAAI/Infinity-Preference](https://huggingface.co/datasets/BAAI/Infinity-Preference)

英文資料的設計可參考相關論文，以下簡述中文資料集的構建方式。

#### Chinese DPO

##### [SPIN/SPPO](https://github.com/uclaml/SPIN)

我們仿照 SPIN 的設計，使用 Kyara-SFT 針對 High Quality Dataset 生成了一組對比資料。

##### RLAIF

資料集：[zake7749/kyara-chinese-preference-dpo-s0-30K](https://huggingface.co/datasets/zake7749/kyara-chinese-preference-dpo-s0-30K)

我們自 `Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese`, `hfl/stem_zh_instruction` 與 `FreedomIntelligence/Evol-Instruct-Chinese-GPT4` 中抽取了中文 Prompt，並將同一個 prompt 分發給 4 個不同的 LLM。競爭的 LLM 包括：

* GPT-4o
* GPT-4-0618
* ChatGPT-3.5-0513
* Claude-Sonnet-3.5
* Yi-Large
* Mixtral 8x22B
* Gemini-Flash
* Qwen2-72B-Instruct
* DeepSeek V2

生成完後，我們再透過以下 Prompt 請 LLM 判斷回答的優劣：

```
**[Task]**

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should consider correctness and helpfulness.  
1. First, independently solve the user question step-by-step.  
2. Then, compare both assistants’ answers with your answer. Identify and correct any mistakes.  
3. Do not allow the length of the responses to influence your evaluation.  
4. Be as objective as possible.

After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie or if both A and B are bad.  

If the answers from A and B are very similar in terms of correctness, helpfulness, and relevance, meaning there is no "obvious" winner, judge it as a tie and output [[C]].

**[User Question]**  
{prompt}

---

**[Assistant A’s Answer]**  
{answer}

---

**[Assistant B’s Answer]**  
{prediction}

---
```

## Feature

### Retrieval Augmented Generation (Experimental)

受益於 Kyara 的訓練方式，我們在 SFT 階段加入 RAG 相關語料，並引入了多種任務模板以增強情境適應能力，可參考以下例子構建任務模板:

#### Input

```
# 參考文件
<reference>
<document>
文件ID:id_27025b13
* 文件標題:Flash_memory
* 文件內文:
另一項快閃記憶體的限制是它有抹寫循環的次數限制（大多商業性 SLC 快閃記憶體保證「0」區有十萬次的抹寫能力，但因為製造精度問題其他區塊不保證，有可能還會出現完全無法使用的出廠壞塊）。這個結果部分地被某些韌體或檔案系統為了在相異區塊間分散寫入操作而進行的計算寫入次數與動態重映射所抵銷；這種技巧稱為耗損平衡（wear leveling）。另一種處理方法稱為壞區管理（Bad Block Management, BBM）。這種方法是在寫入時做驗證並進行動態重測，如果有驗證失敗的區塊就加以剔除。對多數行動裝置而言，這些磨損管理技術可以延長其內部快閃記憶體的壽命（甚至超出這些裝置的使用年限）。此外，遺失部分資料在這些裝置上或許是可接受的。至於會進行大量資料讀寫循環的高可靠性資料儲存應用則不建議使用快閃記憶體。不過這種限制不適用於路由器與瘦客戶端（Thin clients）等唯讀式應用，這些裝置往往在使用年限內也只會寫入一次或少數幾次而已。

### 讀取干擾
</document><document>
文件ID:id_858b1787
* 文件標題:Flash_memory
* 文件內文:
* TLC NAND 型快閃記憶體的續航率通常落在 1 千次或更多（Samsung 840）；以多層結構取代微縮及採用 LDPC 校正、都延長了續航率。
* QLC NAND 型快閃記憶體的續航率可以達到 5 百至 1 千次。
* SLC 浮柵 NOR 型快閃記憶體通常有著 10 萬至百萬次的寫入續航率（Numonyx M58BW 100k; Spansion S29CD016J 1,000k）
* MLC 浮柵 NOR 型快閃記憶體通常有著 10 萬的寫入續航率（Numonyx J3 flash）

以上資料只是大概的標稱數值，實際寫入壽命與不同廠商的產品技術及定位有關。使用更細微化的製程，可以提高產品讀寫效能和容量，但同時在寫入壽命方面可能會面臨更大的挑戰。使用如記憶損耗調節及 memory over-provisioning 的特定演算法及設計範例，可以用來調節儲存系統的續航率來符合特定的需求。損耗平衡是快閃記憶體產品使用壽命的必要保證，在 USB 隨身碟和固態硬碟等產品中，均有相關支援。

## 快閃記憶體檔案系統
</document><document>
文件ID:id_df34eb65
* 文件標題:Memory_over-provisioning
* 文件內文:
## SSD 基本操作

由於快閃記憶體操作的性質，資料不能像在硬碟中那樣直接覆寫。當首次向 SSD 寫入資料時，單元都處於已擦除狀態，因而資料可以直接寫入，一次一頁（大小通常為 4 至 8 千位元組（KB））。SSD 中管理快閃記憶體與主控系統介面的 SSD 控制器，使用稱為邏輯區塊位址（LBA）的邏輯到物理對映系統，這是快閃記憶體轉換層（FTL）的一部分。當新的資料要替換已寫入的舊資料時，SSD 控制器將會寫入新的資料至新的位置，並且更新邏輯對映，將其指向新的物理位置。原位置的資料將不再有效。在可以再次寫入之前，它需要先被擦除。

快閃記憶體的編程和擦除次數有限。通常以快閃記憶體在整個壽命中最多可忍受的編程 / 擦除迴圈（P/E 迴圈）次數來表示。單層單元（SLC）快閃記憶體，通常設計為高效能和長壽命，一般能有 50000 到 100000 次迴圈。截至 2011 年 (2011-Missing required parameter 1=_month_!)，設計用於低成本應用的多層單元（MLC）快閃記憶體，迴圈次數就大為減少，一般只有 3000 至 5000 次迴圈。自 2013 年起，已有三層單元（TLC）快閃記憶體，其編程 - 擦除（P/E）迴圈次數又降至 1000。寫入放大越低，則越為理想，因為與之對應的是快閃記憶體中 P/E 迴圈次數減少，所以能延長 SSD 的壽命。
</document>
</reference>

---

# 任務說明

請參考上述 <reference> 內的內容，回答使用者的問題。

回答時注意以下幾點：

1. 適當援引參考文件ID來佐證你的回答論述，引用方式可參考例子：

假設我們有文章1 id: dq945509 與文章2 id: 6a178c5，如果我們想進行引述時規格如下：這是一段事實論述【dq945509】，而這是另一段事實論述【6a178c5】。

2. 如果發現參考文件的所有內容都與無關使用者問題，請輸出：「參考文件缺乏與問題相關資訊」，不要杜撰無關的回答。

# 使用者問題

記憶卡的寫入次數限制會如何影響它的使用壽命？對於長期重複寫入的應用場景，有哪些技術可以延長記憶卡的壽命？
```

#### Output

```
記憶卡的寫入次數限制對其使用壽命有著直接影響。快閃記憶體的寫入次數受到技術和設計的影響，例如：

1. **SLC快閃記憶體**（如Numonyx M58BW和Spansion S29CD016J）通常有10萬至百萬次的寫入續航率，這使得其在高寫入需求的應用中使用壽命可長達數年【id_858b1787】。

2. **TLC和MLC快閃記憶體**則因為使用多層設計，其寫入續航率相對較低，分別約為500到1000次【id_858b1787】。這意味著在高寫入密度的應用中，這些記憶卡的壽命會較快接近其限制，從而需要定期更換。

為了延長記憶卡的壽命，特別是在長期重複寫入的應用場景中，可以採用以下技術：

1. **耗損平衡（Wear Leveling）**：此技術通過分散寫入操作來均勻分配寫入次數，避免某些區域過度使用，從而延長存儲設備的壽命【id_27025b13】。

2. **壞區管理（Bad Block Management, BBM）**：此方法在寫入資料時進行驗證，及時剔除那些寫入失敗的區塊，從而保護存儲的完整性和持續性【id_27025b13】。

3. **記憶損耗調節和記憶過度配置（Memory Over-provisioning）**：這些技術能夠調整存儲系統的容量和壽命，以滿足特定需求【id_df34eb65】。

這些技術能有效地延長快閃記憶卡在高寫入密度的應用中的使用壽命，即使在高性能和高負載的設備中也能有效運行。
```

更多詳細資訊，請參考 [Kyara-RAG](https://github.com/zake7749/kyara-rag)

## Limitation

與多數模型一樣，Kyara 仍存在著幻覺的問題，而且可能是受資料特性的影響，在回答時特別喜歡引經據典，比如請模型解釋一句《活俠傳》中的台詞:

```
該如何理解：「俠者，不受名利所牽，但求俯仰無愧。正邪存乎在我，聲名留問諸君。非我桀傲不馴，乃因唐門薪火，自古如今！」
```

模型的回答會是：

```
這句話出自《神雕俠侶》中的《神雕手記》，是楊過對自己處世態度的一種表達。讓我們逐句拆解理解：

1. **「俠者，不受名利所牽，但求俯仰無愧。」**
   - **俠者**：指俠義之士，講究義氣和正義的英雄。
   - **不受名利所牽**：不受外在名利和利益的影響，保持自我的正直。
   - **但求俯仰無愧**：只求自己行動的坦蕩和內心的清淨，不求外界的認可或評價。

2. **「正邪存乎在我，聲名留問諸君。」**
   - **正邪存乎在我**：善惡的歸屬在於自己，決策和行動的結果在於自己。
   - **聲名留問諸君**：聲名是別人會問津的，但自己對正義的追求和內心的正直，才是最值得自豪和珍視的。

3. **「非我桀傲不馴，乃因唐門薪火，自古如今！」**
   - **非我桀傲不馴**：並不是我桀驁不馴（驕傲自負），而是因唐門的榮耀和傳承，我必須遵循這些責任。
   - **乃因唐門薪火**：是因為唐門（一種武學門派）的薪火相傳，這是一份深重的責任。
   - **自古如今**：自古至今，俠義精神和門派的責任感，一直是人們所崇敬的，這是歷史和傳統的延續。

總結起來，這段話表達了楊過對於自己行為的獨特理解和自豪感。他明白自己的行為和價值觀取決於個人的內心和對正義的追求，而非外界的評價和名利。他也承認了自己的責任，作為唐門下一代，必須繼承和發揚門風，這一點是無可替代的。
```

建議在使用時仍要謹慎判斷文章的正確性，不要全方面相信大型語言模型的輸出。

## Citation

```bibtex
@software{Yang_Kyara_2024,
  author = {Yang, Kai-Chou},
  doi = {10.5281/zenodo.14001626},
  month = oct,
  title = {{Kyara}},
  url = {https://github.com/zake7749/Kyara},
  version = {1.0.0},
  year = {2024}
}
```
