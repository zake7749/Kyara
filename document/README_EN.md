# Kyara: Knowledge Yielding Adaptive Retrieval Augmentation for LLM Fine-tuning

<p align="left">
    🤗 <a href="https://huggingface.co/zake7749/gemma-2-2b-it-chinese-kyara-dpo">Hugging Face</a>&nbsp ｜ 🚀<a href="https://github.com/zake7749/kyara">Github</a>&nbsp ｜ &nbsp📑 <a href="#">Paper</a>&nbsp ｜ &nbsp📖 <a href="#">English</a>&nbsp | &nbsp📖 <a href="https://github.com/zake7749/kyara">Chinese</a>
</p>
<div style="text-align: center;">
  <img src="https://i.imgur.com/QiWlcYJ.jpeg" alt="kyara"/>
</div>

Kyara is an experimental strategy for fine-tuning language models, designed to effectively enhance the model's knowledge adaptation and language understanding capabilities through knowledge retrieval augmentation.

To validate the effectiveness of this method, We conducted full-parameter fine-tuning on  `Gemma-2-2b-it`, resulting in the first version of the Kyara model. Preliminary evaluation results can be seen in the [Benchmark](#benchmark) section.

## Table of Content

- [Benchmark](#benchmark)
   * [**General Benchmark**](#general-benchmark)
   * [**Alignment Benchmark**](#alignment-benchmark)
- [Feature](#feature)
   * [System Prompt](#system-prompt)
      + [Input](#input)
      + [Output](#output)
      + [Input](#input-1)
      + [Output](#output-1)
   * [Retrieval Augmented Generation (Experimental)](#retrieval-augmented-generation-experimental)
      + [Input](#input-2)
      + [Output](#output-2)
- [Method](#method)
   * [Dataset Summary](#dataset-summary)
   * [Dataset Construction](#dataset-construction)
      + [Base Dataset: Knowledge Injection with Retrieval Augmentation](#base-dataset-knowledge-injection-with-retrieval-augmentation)
         - [Chinese Math Dataset](#chinese-math-dataset)
      + [High Quality Dataset: Model Refinement ](#high-quality-dataset-model-refinement)
   * [Preference Learning](#preference-learning)
      + [Chinese DPO](#chinese-dpo)
         - [SPIN/SPPO](#spinsppo)
         - [RLAIF](#rlaif)

## Benchmark

### **General Benchmark**

All evaluations are based-on zero-shot.

| Metric                   | Kyara-2b-it    | Gemma-2-2b-it |
|--------------------------|----------|-------------|
| **[TMMLUPlus](https://huggingface.co/datasets/ikala/tmmluplus)**            | **39.22** | 36.73    |
| &emsp;- STEM               | **40.86**   | 37.84      |
| &emsp;- Humanities         | **36.39**   | 33.40      |
| &emsp;- Other              | **37.97**   | 36.00      |
| &emsp;- Social-Science     | **41.66**   | 39.69      |
| **[MMLU-Redux](https://github.com/yuchenlin/ZeroEval)**           | **53.78**| 51.94       |
| **[GSM8K](https://github.com/yuchenlin/ZeroEval)**                | **54.74**| 51.63       |
| **[Truthful QA](https://github.com/EleutherAI/lm-evaluation-harness)**          | **45.43**| 43.72     |
| **[CRUX](https://github.com/yuchenlin/ZeroEval)**          | **21.25**| 20.75     |
| **Chinese-Reason-Bench** | **3.97** | 3.44        |

 The aggregation method for the groups in TMMLUPlus is macro average, following the practice in the official implementation.

### **Alignment Benchmark**

| Metric                   | Kyara    | Gemma-2-2b-it |
|--------------------------|----------|-------------|
| **[AlpacaEval-LC](https://github.com/tatsu-lab/alpaca_eval)**        | **41.15**| 32.37       |
| **[AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)**           | **43.95**| 32.94       |
| **[MT-Bench-TW](https://huggingface.co/datasets/MediaTek-Research/TCEval-v2)**          | **6.94** | 6.59        |
| **[MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench)**             | 7.92     | **8.17**    |

## Feature

Compared to Gemma-2-2b-it, we have introduced the following additional features:

### System Prompt

Kyara supports System Prompt, as the following example:

#### Input

```
System: 你是一個愛賣弄知識的人，總是喜歡引用典故，把簡單的事情說得非常複雜。
User: 什麼是大型語言模型？
```

#### Output

```
大型語言模型（Large Language Model，簡稱LLM）是一種基於深度學習技術的自然語言處理（NLP）模型。這些模型通過大量文本數據進行訓練，以學習和生成人類語言。具體來說，它們屬於一種被稱為「變換器」（Transformer）的架構。

變換器最初由Vaswani等人在2017年的論文《Attention is All You Need》中提出，這一架構因其在處理序列數據上的卓越性能而迅速風靡。變換器的核心思想是利用「自注意力機制」（Self-Attention Mechanism），這是一種能夠捕捉文本中不同單詞之間長距離依賴關係的方法。

大型語言模型通常具有以下幾個特徵：

1. **訓練數據量巨大**：這些模型通常需要數百GB甚至TB級的訓練數據，以捕捉語言的豐富性和多樣性。

2. **模型規模龐大**：這意味著模型包含大量參數，常見的規模包括幾十億到數萬億個參數。這些參數的數量決定了模型的複雜度和生成文本的質量。

3. **多語言能力**：現代的LLM如GPT-3、BERT等，不僅能夠處理多種語言，還能夠在不同語言之間轉換和理解上下文。

4. **應用廣泛**：這些模型在文本生成、情感分析、問答系統、機器翻譯、摘要生成等多個NLP任務中都有廣泛應用。

例如，OpenAI的GPT-3包含1750億個參數，這使其成為一個極其強大的語言生成模型。BERT則以其雙向編碼器和掩碼語言建模技術，在多種下游任務中取得了卓越的成績。

總結來說，大型語言模型通過大量數據訓練，利用複雜的深度學習架構，能夠生成高質量的自然語言文本，從而在各種NLP應用中發揮重要作用。
```

#### Input

```
System: 你是一名小學老師，會用五歲小孩也能懂的口吻回答使用者的問題
User: 什麼是大型語言模型？
```

#### Output

```
好的！我來用簡單的方式解釋一下「大型語言模型」這個概念。

想像一下，你有一本非常大且詳盡的書，裡面寫了很多很多的故事、知識和資訊。這些故事和知識就像是巨大的語言倉庫，裡面有各種各樣的詞語和句型。

大型語言模型（Large Language Model），簡單來說，就是一種非常聰明的機器人，它學習了這本書裡的大量文字，然後學會了如何用這些文字來回答問題、寫故事、解釋道理等等。

這個機器人的特別之處在於它「看過」了很多很多東西，所以它可以理解很多事情，並用很流暢的語言給你解釋。但要記住，它還是機器，不是真的人，所以有時它的回答可能還不夠完美。

舉個例子，如果我們問它「什麼是太陽？」，它會從它學到的知識中找答案，比如說：「太陽是我們太陽系中心的一顆黃矮星，它給地球提供光和熱，讓植物能夠做光合作用。」

希望這樣的解釋能讓你明白！有其他問題也可以隨時問我。
```

### Retrieval Augmented Generation (Experimental)

Benefiting from Kyara's training method, we incorporated RAG-related content during the SFT phase. You can refer to the following examples to construct task templates:

#### Input

```
# 參考文件
<reference>
<document>
文件ID:id_816fcfd8
文章標題:口呼吸
文章內文:
**口呼吸**指的是用嘴呼吸的行為，這通常是由於鼻子呼吸受阻而引起，鼻子是人體的先天呼吸器官。慢性口呼吸可能與某些疾病有關。
## 參考文獻
</document><document>
文件ID:id_6c0f7501
文章標題:口角炎
文章內文:
**口角炎**（英語：**Angular cheilitis or Angular Stomatitis， perlèche**），或稱爛嘴角，為發生在嘴唇一側或兩側角落部位的炎症，通常為兩側同時發炎。此症是唇炎（cheilitis）的一種形式，發炎部位皮膚通常會紅腫、脫皮及結痂，也可能會造成發癢或疼痛，症狀可持續數天甚至可達數年之久。
口角炎可能因感染、剌激或是過敏而引起。感染源包含如白色念珠菌等真菌以及如金黃色葡萄球菌等細菌。剌激源包含配帶不適當的假牙、舔嘴唇、用嘴巴呼吸導致的嘴部乾燥、日曬、嘴部過度閉合、抽菸，以及輕微創傷。過敏源則包含牙膏、化妝品及食品等物質。其他病因可能包含營養不良或免疫功能不良 。會發生此症通常是多重因素作用的結果，對患者進行感染及皮膚過敏源測試將有助於診斷肇因。
口角炎的治療一般而言是在找出肇因後使用適當的防護霜（護膚膏），也常嘗試使用抗黴菌及抗細菌軟膏加以治療。此症可說是相當常見的疾病，據估計在美國約有 0.7% 的人受到此症影響 。口角炎好發於 30 至 60 歲間的人，但在孩童身上也相對常見。在開發中國家，缺乏鐵質及維他命是此症常見的肇因。長期處於潮濕反而會使口角炎更嚴重，嘴角更乾燥，且脫皮愈嚴重，適當保持乾燥才能讓細菌壞死。
## 病因
因病因不同而分為營養不良性口角炎、球菌性口角炎、真菌性口角炎。
營養不良性口角炎多為缺乏維生素 B 族，多為 B2 核黃素或 B12 鈷胺素，造成的嘴角貧血。以及缺鐵、缺鋅。
球菌性口角炎和真菌性口角炎為細菌或真菌感染，細菌或真菌被帶到嘴角後在濕潤的環境下容易形成炎症，因此應使嘴角儘量乾燥，使細菌不易存活。原因可能為過曬或過干（舔嘴角），機械原因如閉合不當，假牙不適或老年掉牙過度閉合，及嘴角流涎造成的。
## 症狀
單側或兩側口角濕白色，紅腫，潰爛，結痂，有燒灼感。口角發緊，運動開裂。
## 診斷
營養不良性口角炎可能會有舌，口腔，陰部黏膜等全身性的相應症狀。或伴有膿疱，多與化膿球菌感染有關。
## 治療
營養不良性口角炎直接施與 B 族維生素即可，球菌性口角炎藥物一般處方抗生素，而真菌性口角炎藥物則處方抗真菌藥物。
## 參考資料
</document><document>
文件ID:id_a214252f
文章標題:打呼
文章內文:
**鼻鼾**是呼吸系統的結構震動而產生的聲音，原因是睡覺時呼吸被阻擋。在一些情況下聲音較輕，但一般情況下都是嘈吵及煩人的。鼻鼾同時可能是睡眠窒息症的第一個警號。研究指出鼻鼾是睡眠不足的一項因素。
## 名稱
表示發出鼾聲的意思，可用「打鼾」，或俗「打呼」「打呼嚕」。「鼾」其字其音，至少在東漢以前就訓爲鼾聲的意思。
## 成因
鼻鼾通常由於懸雍垂和軟顎鬆弛而引起，鬆弛的軟組織會令到氣管阻塞或不暢通，導致不規則的氣流和振動。以下的情況都可能是鼻鼾的成因：
* 生活習慣，如吸菸、酗酒或濫藥
* 嚥喉無力，導致睡眠期間嚥喉關閉
* 顎位不對齊，通常是由於肌肉緊張所致
* 肥胖症，過多脂肪積聚於嚥喉附近
* 鼻腔阻塞
* 阻塞性睡眠呼吸暫停
* 睡眠不足
* 仰睡
* 以口呼吸
## 參考資料
</document><document>
文件ID:id_964dde34
文章標題:人工呼吸
文章內文:
**口對口人工呼吸**是人工呼吸的一種形式，是人們協助或刺激他人呼吸的行為，人們在人工呼吸時要用嘴抵住被救援者的嘴，然後將空氣吹入被救援者的肺部。通常無法自主呼吸的人需要人工呼吸。18 世紀末，英國的醫生開始積極普及人工呼吸。
## 參考文獻
</document>
</reference>

---

# 任務說明

你是一名研究員，請詳細閱讀上述參考文件，以通俗易懂的繁體中文回答使用者的問題。
在回答問題時，請注意以下幾點：
1. 請適當的援引參考文件的資訊，來佐證你的回答論述，援引方式如撰寫論文時在引用參考資料一般，可參考例子：

假設我們有文章1 id: dq945509 與文章2 id: 6a178c5，如果我們想進行引述時規格如下：這是一段事實論述【dq945509】，而這是另一段事實論述【6a178c5】。

2. **請注意，如果發現參考文件與使用者的問題無關，或者無助於回答使用者的問題，請直接回答：「參考文件缺乏與問題相關資訊」，不要杜撰無關的回答。**

# 使用者問題
為什麼說用嘴呼吸不健康？
```

#### Output

```
用嘴呼吸不健康的原因主要有以下幾點：

1. **乾燥和刺激**：用嘴呼吸會使口腔內部和舌頭變得非常乾燥，容易引起口腔黏膜的刺激和炎症。例如，口角炎就是因為嘴部乾燥、不適當的環境（如用嘴巴呼吸）導致的炎症【id_6c0f7501】。

2. **感染風險增加**：用嘴呼吸會使口腔內的細菌和病毒更容易進入呼吸道，增加感染的風險。例如，口角炎的病因之一就是因為口腔內的細菌或真菌感染【id_6c0f7501】。

3. **睡眠質量下降**：用嘴呼吸會導致打呼，這是因為懸雍垂和軟顎鬆弛，導致氣管阻塞或不暢通，從而引起打鼾，進而影響睡眠質量【id_a214252f】。

4. **慢性問題**：長期用嘴呼吸可能引發慢性口呼吸，這會對健康產生多方面的影響【id_816fcfd8】。

因此，為了維護呼吸道健康，應儘量通過鼻子進行呼吸，以減少上述問題的發生。
```

Please refer to [Kyara-RAG](https://github.com/zake7749/kyara-rag) for more details.

## Method

The following sections provide a brief summary of Kyara's implementation strategy.

### Dataset Summary

We have collected a total of 2.6M conversations, approximately 3.54 billion tokens. The following provides an overview of the language distribution and conversation rounds.

* Language：

<img src="https://i.imgur.com/KvVjti4.png" alt="language-distribution" width="500"/>

* Conversation Rounds：

<img src="https://i.imgur.com/dekAnU0.png" alt="conv-round-distribution" width="500"/>

### Dataset Construction

#### Base Dataset: Knowledge Injection with Retrieval Augmentation

We developed a knowledge search system using open Chinese knowledge corpora, integrated with [QDrant](https://qdrant.tech/). To construct Supervised Fine-Tuning(SFT) pairs, we followed this process:

1. Sample documents from the knowledge base and generate knowledge-intensive questions that users might ask based on these texts.
2. (Optional) Increase instruction complexity using [Evol-Instruct](https://arxiv.org/pdf/2304.12244).
3. Apply query expansion on the generated instructions to retrieve additional Top K documents and individually assess their relevance:
   * For relevant documents, use an LLM to summarize key information related to the question.
   * For irrelevant documents, ignore them.
4. Let the LLM generate a detailed and comprehensive response according to the original document and K supplementary references.

Besides, we would also aks LLM to generate an user prompt for high quality documents, and pair the (generated prompt, original document) as a SFT example.

##### Chinese Math Dataset

* Dataset: [zake7749/kyara-chinese-math-sft-s0-30K](https://huggingface.co/datasets/zake7749/kyara-chinese-math-sft-s0-30K)

While the aforementioned strategy can generate a wide range of knowledge-based texts, it primarily falls within the scope of information-seeking tasks and is not very effective in constructing mathematical and reasoning-related content. To address this, we generated 50,000 math problems based on [PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub). We then used `Gemini-1.5-Flash` to filter out data with obvious errors in calculation and reasoning, thereby creating [kyara-chinese-math-sft-s0-30K](https://huggingface.co/datasets/zake7749/kyara-chinese-math-sft-s0-30K).

#### High Quality Dataset: Model Refinement 

After completing supervised learning using the base dataset, we will fine-tune the LLM again on a high-quality subset, primarily to address the following three issues:

1. Some responses in the Base Dataset were generated from small model, which sometimes performed poorly in following instructions.
2. We used various LLMs in the previous step to introduce knowledge diversity and language adaptability. However, we discovered subtle differences in response templates and reasoning approaches between different LLMs, leading to occasional instability in the trained Chat Model. Therefore, we would like to introduced a high-quality small dataset, using a single strong LLM to generate QA Pairs.
3. The Base Dataset includes some Q&A Pairs composed of generated queries and original document. While these data are rich in knowledge, they are relatively weak in terms of instruction following.

To balance data diversity and quality, we adopted a strategy similar to [InsTag](https://arxiv.org/abs/2308.07074) to classify the data. We then used [ArmoRM](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) and an LLM Judge to evaluate data quality, finally extracting the best training data from each category to create the Stage 1 Dataset of about 200K, which was used to fine-tune the Kyara-SFT Model again.

### Preference Learning

We introduced Preference Learning in Kyara, which allows the model's responses to better align with human preferences while enhancing programming skills and mathematical reasoning abilities.

Kyara’s preference learning strategy utilizes Direct Preference Optimization (DPO), integrating two custom-built Chinese datasets alongside two English datasets.

* [argilla/ultrafeedback-binarized-preferences](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences)
* [xinlai/Math-Step-DPO-10K](https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K)

Here, we summarize the construction strategy of the Chinese datasets.

#### Chinese DPO

##### [SPIN/SPPO](https://github.com/uclaml/SPIN)

We followed the original design, using Kyara-SFT to generate a set of contrastive data for the High Quality Dataset.

##### RLAIF

資料集：[zake7749/kyara-chinese-preference-dpo-s0-30K](https://huggingface.co/datasets/zake7749/kyara-chinese-preference-dpo-s0-30K)

We extracted Chinese Prompts from `Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese`, `hfl/stem_zh_instruction`, and `FreedomIntelligence/Evol-Instruct-Chinese-GPT4`, and distributed the same prompt to four different LLMs. The competitors include:

* GPT-4o
* GPT-4-0618
* ChatGPT-3.5-0513
* Claude-Sonnet-3.5
* Yi-Large
* Mixtral 8x22B
* Gemini-Flash
* Qwen2-72B-Instruct
* DeepSeek V2

After response generation, we ask the LLMs to judge which one is better, using the following prompt:

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

Finally, all four datasets were combined for DPO training.

## Limitation

Like most models, Kyara still suffers from hallucinations, and possibly due to the nature of the data, it particularly enjoys quoting references when answering questions. For example, if you ask the model to explain a line from "Legend of Mortal":

```
如何解釋「俠者，不受名利所牽，但求俯仰無愧。正邪存乎在我，聲名留問諸君。非我桀傲不馴，乃因唐門薪火，自古如今！」
```

The model would respond that this quote is from The "Legend of the Condor Heroes."

```
這段話出自《射鵰英雄傳》中的唐門內部經典，由郭靖和黃蓉在唐門大殿上宣讀。這段話主要表達了唐門門人對於俠義精神和內在修為的追求，以及他們對外界的名利和聲名的不屑。以下是詳細解釋：

1. **「俠者，不受名利所牽」**：
   - 俠者（俠士）是指那些不為名利所動搖、不追求世俗榮華富貴的人。這句話強調了他們追求的是內在的修為和道德，而非外在的名聲和利益。

2. **「但求俯仰無愧」**：
   - 俯仰無愧意指在行動和言行上無愧於心，沒有任何自私的動機。這意味著他們只追求內心的光明和正義，不為外界的評價所動搖。

3. **「正邪存乎在我，聲名留問諸君」**：
   - 正邪存乎在我，意指他們內心有正邪之分，但他們會在行動中保持正義，不讓邪惡侵染。聲名留問諸君，意味著他們的行為和品行會讓世人議論和稱道，但他們並不在意外界的評價。

4. **「非我桀傲不馴，乃因唐門薪火，自古如今！」**：
   - 桀傲不馴（桀傲）指的是桀驁不馴、桀驁不羈，形容那些桀驁不馴、不聽從權威的人。這句話解釋了唐門門人之所以桀驁不馴，是因為他們繼承了唐門薪火，這門派自古以來就以高尚的道德和強大的武功聞名，門人自然要保持這種風範。

總結來說，這段話強調了唐門門人對名利和聲名的不屑，他們追求的是內在的修為和道德，不為外界的評價所動搖。這也體現了他們俠義精神和對正義的堅持。
```

It is recommended to exercise caution when using language models.
