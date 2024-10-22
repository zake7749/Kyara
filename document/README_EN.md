# Kyara: Knowledge Yielding Adaptive Retrieval Augmentation for LLM Fine-tuning

<p align="left">
    🤗 <a href="https://huggingface.co/zake7749/gemma-2-2b-it-chinese-kyara-dpo">Hugging Face</a>&nbsp; ｜ 🚀<a href="https://github.com/zake7749/kyara">Github</a>&nbsp; ｜ &nbsp;📑 <a href="#">Paper</a>&nbsp; ｜ &nbsp;📖 <a href="https://github.com/zake7749/kyara/blob/main/document/README_EN.md">English</a>&nbsp; ｜ &nbsp;📖 <a href="https://github.com/zake7749/kyara">Chinese</a>&nbsp; ｜ &nbsp;💻 <a href="https://www.kaggle.com/code/zake7749/kyara-a-compact-yet-powerful-chinese-llm">Kaggle Notebook</a>
</p>

<div style="text-align: center;">
  <img src="https://i.imgur.com/QiWlcYJ.jpeg" alt="kyara"/>
</div>

Kyara (Knowledge Yielding Adaptive Retrieval Augmentation) is an experimental project aimed at improving language models through knowledge retrieval processes. The project seeks to enhance the model’s ability to adapt knowledge and improve language comprehension, particularly in underrepresented languages like Traditional Chinese. Given the relatively scarce availability of Traditional Chinese data compared to the vast corpus of English data used for model training, Kyara addresses this gap by expanding the limited corpus for this language.

To validate Kyara's effectiveness, we conducted full-parameter fine-tuning on `Gemma-2-2b-it`, resulting in the first iteration of the Kyara model. Initial evaluation results, as detailed in the [Benchmark](#benchmark) section, demonstrate that Kyara outperforms the original `Gemma-2-2b-it` across various benchmarks, with notable improvements in Chinese language evaluations.

## Table of Contents

- [Benchmark](#benchmark)
- [Usage](#usage)
- [Method](#method)
- [Feature](#feature)
- [Limitation](#limitation)

## Benchmark

### General Benchmark

The following evaluations are based-on zero-shot.

| Metric                   | Kyara-2b-it    | Gemma-2-2b-it |
|--------------------------|----------|-------------|
| **[TMMLUPlus](https://huggingface.co/datasets/ikala/tmmluplus)**            | **41.98** | 36.73    |
| &emsp;- STEM               | **43.73**   | 37.84      |
| &emsp;- Humanities         | **38.72**   | 33.40      |
| &emsp;- Other              | **40.61**   | 36.00      |
| &emsp;- Social-Science     | **44.88**   | 39.69      |
| **[MMLU-Redux](https://github.com/yuchenlin/ZeroEval)**    | **55.44**| 51.94       |
| **[GSM8K](https://github.com/yuchenlin/ZeroEval)**         | **54.21**| 51.63       |
| **[MATH-L5](https://github.com/yuchenlin/ZeroEval)**       | **8.88**| 4.3       |
| **[CRUX](https://github.com/yuchenlin/ZeroEval)**          | **22.75**| 21.5     |
| **[ZebraLogic](https://github.com/yuchenlin/ZeroEval)**    | **5.2**| 4.2  |
| **Chinese-Reason-Bench** | **4.21** | 3.44        |

 The aggregation method for the groups in TMMLUPlus is macro average, following the practice in the official implementation.

#### [Open-LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

As of now, Kyara-2b-it is the leading competitor among all 2b-scale models on the OpenLLM Leaderboard.

<div style="text-align: center">
  <img src="https://imgur.com/3NKhAja" alt="kyara-2b-it-open-llm-leaderboard">
</div>

### Alignment Benchmark

| Metric                   | Kyara    | Gemma-2-2b-it | ChatGPT-3.5-1106 |
|--------------------------|----------|---------------|------------------|
| **[AlpacaEval-LC](https://github.com/tatsu-lab/alpaca_eval)**                   | **35.35** | 32.37       | 19.30    |
| **[AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)**                      | **43.34** | 32.94       | 9.20     | 
| **[MT-Bench-TW](https://huggingface.co/datasets/MediaTek-Research/TCEval-v2)**  | **7.43**  | 6.35        | 7.10     |
| **[MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench)**                    | 8.28      | 8.17        | **8.32** |
| **[Chatbot-Arena-Hard](https://github.com/lm-sys/arena-hard-auto)**             | **22.60** | 19.4        | 18.87    |

#### [AlignBench](https://github.com/THUDM/AlignBench)

| Fold  | Kyara-2b-it-CHT | Kyara-2b-it-CHS | Gemma-2-2b-it |  ChatGPT-3.5-0613   |
|-------|-----------------|-----------------|---------------| ----       |
| Fundamental Language Ability   | 6.72     | 6.54 | 6.42 | **6.92** |
| Advanced Chinese Understanding	 | 5.78     | 5.24 | 5.03 | **5.91** |
| Open-ended Questions	 | **8.16** | 7.79 | 7.52 | 6.47     |
| Writing Ability		 | **7.90** | 7.24 | 7.76 | 7.28     |
| Logical Reasoning	 | **5.26** | 4.27 | 4.20 | 4.79     |
| Mathematics	 | **5.99** | 5.44 | 5.05 | 5.38     |
| Task-oriented Role Play	 | **8.07** | 8.00 | 7.42 | 7.00     |
| Professional Knowledge	 | **6.97** | 6.86 | 5.79 | 6.81     |
| Reasoning AVG.   | **5.62** | 4.85 | 4.63 | 5.00     |
| Chinage Language AVG.   | **7.26** | 6.94 | 6.66 | 6.73     |
| Overall      | **6.44** | 5.90 | 5.64 | 5.91     |

where the postfixes CHT and CHS represent Traditional Chinese and Simplified Chinese, respectively. To evaluate the performance on Traditional Chinese in AlignBench, we used [OpenCC](https://github.com/BYVoid/OpenCC) with the `s2tw` configuration to convert all questions from Simplified Chinese to Traditional Chinese.

## Usage

Kyara adopts the same architecture as Gemma2, utilizing identical inference and training methods. We have created a [Jupyter Notebook](https://www.kaggle.com/code/zake7749/kyara-a-compact-yet-powerful-chinese-llm) on Kaggle to demonstrate Kyara’s basic functionality. For service-level deployment, we recommend using Sglang or vllm to achieve greater throughput and robustness.

## Method

The following sections provide a brief summary of Kyara's implementation strategy.

### Dataset Summary

We have collected a total of 3.6M conversations, approximately 4.51 billion tokens. The following provides an overview of the language distribution and conversation rounds.

* Language：

<img src="https://i.imgur.com/fhD5kIy.png" alt="language-distribution" width="500"/>

* Conversation Rounds：

<img src="https://i.imgur.com/CWQ2shj.png" alt="conv-round-distribution" width="500"/>

### Dataset Construction

The data construction for Kyara is divided into two parts: English and Chinese. For the English part, we have incorporated multiple high-quality open-source datasets, such as [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) and [arcee-ai/The-Tome](https://huggingface.co/datasets/arcee-ai/The-Tome), and performing semantic deduplication to drop out near-similar examples. As for the Chinese part, the construction follows the process outlined below:

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

To balance data diversity and quality, we adopted a strategy similar to [InsTag](https://arxiv.org/abs/2308.07074) to classify the data. We then used [ArmoRM](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) and an LLM Judge to evaluate data quality, finally extracting the best training data from each category to create the Stage 1 Dataset of about 500K, which was used to fine-tune the Kyara-SFT Model again.

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

Dataset: [zake7749/kyara-chinese-preference-dpo-s0-30K](https://huggingface.co/datasets/zake7749/kyara-chinese-preference-dpo-s0-30K)

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

## Feature

### Retrieval Augmented Generation (Experimental)

Benefiting from Kyara's training method, we incorporated RAG-related content during the SFT phase. You can refer to the following examples to construct task templates:

#### Input

```
# Reference Document
<reference>
<document>
Document ID: id_27025b13
* Document Title: Flash_memory
* Document Text:
Another limitation of flash memory is its limited number of erase cycles (most commercial SLC flash memory guarantees around 100,000 erase cycles for the "0" zone, but due to manufacturing precision, other blocks are not guaranteed, and some might even have factory defects and be unusable). This limitation is partly offset by firmware or file systems that calculate write counts and perform dynamic remapping to distribute writes across different blocks; this technique is called wear leveling. Another method is known as Bad Block Management (BBM), where blocks are dynamically tested during write operations, and failed blocks are discarded. For most mobile devices, these wear management techniques can extend the life of internal flash memory (sometimes even beyond the device's lifespan). Moreover, partial data loss in these devices may be acceptable. However, for high-reliability data storage applications that require heavy data write cycles, flash memory is not recommended. But this limitation does not apply to read-only applications, such as routers and thin clients, which often only write once or a few times throughout their lifespan.

### Read Disturbance
</document>
<document>
Document ID: id_858b1787
* Document Title: Flash_memory
* Document Text:
* TLC NAND flash memory typically has an endurance of around 1,000 or more cycles (Samsung 840); using multi-layer structures and adopting LDPC correction have extended the endurance.
* QLC NAND flash memory can have an endurance ranging from 500 to 1,000 cycles.
* SLC floating-gate NOR flash memory typically has a write endurance of 100,000 to 1,000,000 cycles (Numonyx M58BW 100k; Spansion S29CD016J 1,000k).
* MLC floating-gate NOR flash memory usually has a write endurance of 100,000 cycles (Numonyx J3 flash).

These values are approximate and depend on the technology and positioning of different manufacturers' products. Finer process technologies can improve read/write performance and capacity, but they may also pose greater challenges in terms of write endurance. Specific algorithms and design examples, such as wear leveling and memory over-provisioning, can be used to adjust storage system endurance to meet specific needs. Wear leveling is essential for ensuring the lifespan of flash memory products, and it is supported in products like USB flash drives and SSDs.

## Flash Memory File Systems
</document>
<document>
Document ID: id_df34eb65
* Document Title: Memory_over-provisioning
* Document Text:
## Basic SSD Operations

Due to the nature of flash memory operations, data cannot be overwritten directly like in hard drives. When data is first written to an SSD, the cells are in an erased state, so the data can be written directly, one page at a time (usually 4 to 8 KB in size). The SSD controller, which manages the flash memory and interfaces with the main control system, uses a logical-to-physical mapping system called Logical Block Addressing (LBA), part of the flash translation layer (FTL). When new data needs to replace old data, the SSD controller writes the new data to a new location and updates the logical mapping to point to the new physical location. The original data becomes invalid and must be erased before it can be rewritten.

Flash memory has a limited number of program/erase (P/E) cycles. Typically, this is expressed as the maximum number of P/E cycles that flash memory can endure over its lifetime. Single-level cell (SLC) flash memory is generally designed for high performance and long life, typically supporting 50,000 to 100,000 cycles. As of 2011, multi-level cell (MLC) flash memory, designed for low-cost applications, has far fewer cycles, usually only 3,000 to 5,000 cycles. Since 2013, triple-level cell (TLC) flash memory has been introduced, with P/E cycles dropping to around 1,000. The lower the write amplification, the better, as it corresponds to fewer P/E cycles, which extends the lifespan of the SSD.
</document>
</reference>

---

# Task Description

Please refer to the content in the <reference> above and answer the user's question.

When answering, pay attention to the following points:

1. Appropriately cite the reference document ID to support your argument. For example:
   
   If we have Article 1 with ID: dq945509 and Article 2 with ID: 6a178c5, you can cite as follows: "This is a factual statement"【dq945509】 and "This is another factual statement"【6a178c5】.

2. If you find that the reference document lacks relevant information to the user's question, output: "The reference documents lack information relevant to the question." Do not fabricate unrelated answers.

# User's Question

How does the write limit of memory cards affect their lifespan? What techniques can extend the lifespan of memory cards in scenarios with frequent writes?
```

#### Output

```
The write limit of memory cards, particularly flash memory, significantly affects their lifespan due to the limited number of program/erase (P/E) cycles they can endure. For instance, single-level cell (SLC) flash memory, which is designed for high performance and long life, typically supports 50,000 to 100,000 cycles【id_df34eb65】. In contrast, multi-level cell (MLC) flash memory, which is more cost-effective, has fewer cycles, usually around 3,000 to 5,000 cycles【id_df34eb65】. As of 2013, triple-level cell (TLC) flash memory has even lower endurance, with around 1,000 cycles【id_df34eb65】.

To mitigate the impact of these limitations, several techniques can extend the lifespan of memory cards in scenarios with frequent writes:

1. **Wear Leveling**: This technique involves distributing writes across different blocks to prevent any single block from being overworked. Wear leveling is crucial for ensuring the lifespan of flash memory products, and it is supported in products like USB flash drives and SSDs【id_858b1787】.

2. **Bad Block Management (BBM)**: In this method, blocks are dynamically tested during write operations, and failed blocks are discarded. This ensures that only healthy blocks are used for data storage, thereby extending the overall lifespan of the memory card【id_27025b13】.

3. **Memory Over-Provisioning**: This involves allocating extra storage space beyond the actual capacity of the flash memory to account for wear and tear. This can help to maintain performance and extend the lifespan of the memory card by allowing for more writes before reaching the physical limits of the memory cells【id_df34eb65】.

These techniques are particularly important for mobile devices and applications that require frequent data writes, such as routers and thin clients, where the partial data loss in these devices may be acceptable【id_27025b13】.
```

Please refer to [Kyara-RAG](https://github.com/zake7749/kyara-rag) for more details.


## Limitation

Like most models, Kyara still suffers from hallucinations, and possibly due to the nature of the data, it particularly enjoys quoting references when answering questions. For example, if you ask the model to explain a line from "Legend of Mortal":

```
該如何理解：「俠者，不受名利所牽，但求俯仰無愧。正邪存乎在我，聲名留問諸君。非我桀傲不馴，乃因唐門薪火，自古如今！」
```

However, the model would respond that this quote is from The "Legend of the Condor Heroes."

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

It is recommended to exercise caution when using language models.
