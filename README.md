# Multilingual-text-emotion-analysis-based-on-BERT

## Introduction
Social media are widely used in modern people’s communications, Twitter is widely used in English speaking countries to express their emotions. Weibo is widely used in China as the same tool. Both of them express the emotions through couple of texts. It is necessary to design a system which can do emotion classification for multiple languages.

The difficulties to do emotion classification for texts are as follows. First, in the irony issues, such as a traffic cop gets his license suspended because of unpaid parking tickets. Second, in the domain-related  issues,  such  as,  it  is  negative to  say  that my computer’s cooling system sound very loud.  It is positive to say my house’s audio is very loud. Third, internet buzzwords will also affect emotion analysis, the meaning will totally been changed after tokenization.  In  order  to  avoid  the  side effect, human intervention has  to  been  added.  Fourth,  the  text  is  relatively  short and there is some omission sometimes, all of them lead to ambiguity or reference errors. The traditional  methods  which  combine  statistics  and  rules  cannot  solve these difficulties well.  The powerful feature extraction abilities of deep learning can work well to solve these issues.

In October 2018,  Google proposed the Bert model [1].  This model not only integrate the bidirectional encoding mechanism of LSTM but also use Transformer in GPT to do feature extraction, which has a very powerful feature extraction abilities and can learns the potential syntactic and semantic meanings in the sentence. Besides, Bert implement the embedding in the character level, which will avoid the disadvantages of tokens which have not appear in the training set. These advantages make Bert can better do  emotion  classification  tasks.  In  this  paper,  the  experiment  is  based on Google’s open source Bert pre-trained Chinese model for fine-tuning. We will compare the performance with the traditional machine learning algorithms.

## Related Work
The related works on emotion classification tasks have produced a lot of techniques that include supervised and unsupervised methods. In supervised methods, early papers used all supervised machine learning algorithms like SVM, maximum en- tropy, Native Bayes, etc. Unsupervised methods included using emotion dictionar- ies, grammatical analysis, syntactic patterns etc. 

About a decade ago, deep learning became a powerful machine learning technology, it achieved the best performance in many areas, including computer vision, speech recognition,NLP. There also has trend to use deep learning in emotion analysis.

Emotion analysis can be divided into three granularities: document granularity, sentence granularity and phrase granularity. In this paper, I decided to do the emotion classification mainly based on sentence granularity.

Kim [2] et al. proposed CNN text classification in 2013, which became one of the most import baseline for sentence level emotion classification.
The basic LSTM model added the pooling strategy also construct the classification model, which is usually used as method for emotion classification in sentence level. Tang et al. [8] used two different RNN network which combined the text and themes for emotion classification in 2015.

The breakthroughs in emotion classification are mainly in the deep learning ar- eas recently. Deep learning extracts the deep level text features by learning text encoding representations, which solve the poor ability of learning text features of traditional machine learning algorithms. Literatures [4] [5] [9] [6] [7] are several major achievement in text feature extraction using deep learning from 2013 to 2018, which includes Word2Vec, GLoVe, Transformer,ELMo and GPT. In this paper, I will use bert model [1] which is a combination of the former models. We can see their relationship in figure 1.

Bert combines the advantages of ELMo and Transformer. Bert solve the long dis- tance dependency problem compared with LSTM. Bert learn the syntactic features and deep semantic features of sentences. Besides, Bert has a stronger feature ex- traction ability.
