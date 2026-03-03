# Tables

## MultiNERD

Original [MultiNERD](https://github.com/Babelscape/multinerd/tree/master?tab=readme-ov-file#data)
dataset

| Dataset | Sentences | Tokens |   PER |   ORG |    LOC |  ANIM |  BIO |  CEL |   DIS |  EVE |  FOOD | INST | MEDIA | MYTH | PLANT |  TIME | VEHI | OTHER |
| :------ | --------: | -----: | ----: | ----: | -----: | ----: | ---: | ---: | ----: | ---: | ----: | ---: | ----: | ---: | ----: | ----: | ---: | ----: |
| EN      |    164.1K |   3.6M | 75.8K | 33.7K |  78.5K | 15.5K | 0.2K | 2.8K | 11.2K | 3.2K | 11.0K | 0.4K |  7.5K | 0.7K |  9.5K |  3.2K | 0.5K |  3.1M |
| ES      |    173.2K |   4.3M | 70.9K | 20.6K |  90.2K | 10.5K | 0.3K | 2.4K |  8.6K | 6.8K |  7.8K | 0.6K |  8.0K | 1.6K |  7.6K | 45.3K | 0.3K |  3.8M |
| NL      |    171.7K |   3.0M | 56.9K | 21.4K |  78.7K | 34.4K | 0.1K | 2.1K |  6.1K | 4.7K |  5.6K | 0.2K |  3.8K | 1.3K |  6.3K | 31.0K | 0.4K |  2.7M |
| DE      |    156.8K |   2.7M | 79.2K | 31.2K |  72.8K | 11.5K | 0.1K | 1.4K |  5.2K | 4.0K |  3.6K | 0.1K |  2.8K | 0.8K |  7.8K |  3.3K | 0.5K |  2.4M |
| RU      |    129.0K |   2.3M | 43.4K | 21.5K |  75.2K |  7.3K | 0.1K | 1.2K |  1.9K | 2.8K |  3.2K | 1.1K | 11.3K | 0.6K |  4.8K | 22.8K | 0.5K |  2.0M |
| IT      |    181.9K |   4.7M | 75.3K | 19.3K |  98.5K |  8.8K | 0.1K | 5.2K |  6.5K | 5.8K |  5.8K | 0.8K |  8.6K | 1.8K |  5.1K | 71.2K | 0.6K |  4.2M |
| FR      |    176.2K |   4.3M | 89.6K | 28.2K |  90.9K | 11.4K | 0.1K | 2.3K |  3.1K | 7.4K |  3.2K | 0.7K |  8.0K | 2.0K |  4.4K | 27.4K | 0.6K |  3.8M |
| PL      |    195.0K |   3.0M | 66.5K | 29.2K | 100.0K | 19.7K | 0.1K | 3.3K |  6.5K | 6.7K |  3.3K | 0.6K |  4.9K | 1.3K |  6.6K | 44.1K | 0.7K |  2.5M |
| PT      |    177.6K |   3.9M | 54.0K | 13.2K | 124.8K | 14.7K | 0.1K | 4.2K |  6.8K | 5.9K |  5.4K | 0.6K |  9.1K | 1.6K |  9.2K | 48.6K | 0.3K |  3.4M |
| ZH      |    195.3K |   5.8M | 68.3K | 20.8K |  49.6K | 26.1K | 0.4K | 0.8K |  0.1K | 5.1K |  1.9K | 1.1K | 55.9K | 1.8K |  6.1K |  0.4K | 0.3K |  3.4M |

Modified english only subset of MultiNERD, multi span entities with _B-[TAG]_
and _I-[TAG]_ are joined into one _[TAG]_.

| Split | Samples |
| :---- | ------: |
| Train |    2.6M |
| Eval  |    358K |
| Test  |    327K |

Prompt template:

> Task: Named Entity Recognition<br>
> <br>
> Classify the NER tag of the target entity in the sentence:<br>
> <br>
> Sentence: Paris is the capital of France.<br>
> Target: Paris<br>
> Answer: LOC<br>
> <br>
> Sentence: [SENTENCE]<br>
> Target: [ENTITY]<br>
> Answer:

**Parameters**

| Model           | Total | Head | Prefix | Tokens |
| :-------------- | ----: | ---: | -----: | -----: |
| DistilBERT-Base |   67M | 602K |    26K |     35 |
| mmBERT-Small    |  140M |   6K |    14K |     39 |
| mmBERT-Base     |  307M |  12K |    29K |     39 |
| GPT2            |  124M |      |        |        |
| GPT2-Medium     |  355M |      |        |        |
| GPT2-Large      |  774M |      |        |        |
| GPT2-XL         |  1.5B |      |        |        |
| T5-Small        |   60M |      |        |        |
| T5-Base         |  220M |      |        |        |
