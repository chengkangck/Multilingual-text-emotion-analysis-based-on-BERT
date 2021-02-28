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
![image](https://github.com/chengkangck/Multilingual-text-emotion-analysis-based-on-BERT/blob/main/images/The%20relationship%20between%20BERT%2C%20Word2VEC%2C%20GPT%2C%20and%20ELMO.PNG)

## Goals
The difficulty to do emotion classification in the social media text is to extract the features which are closely related to the emotion expressions. Using features which are annotated by humans and from singles word will ignore the contextual semantic information of the word. As a result, these features will lead to the performance of classification is not satisfactory.

In order to fix the feature extraction difficulty, in this paper, I decide to use the Bert (bidirectional encoder representations from transformer) which was proposed in 2018 by Google as a text pre-training model. In our experiment, I will do emotion classification on both English and Chinese corpus. Bert model learns the bidirec- tional coding of words through the super feature extraction ability of transformer. Word coding which consider the contextual information can better make emotion classification.

## Methods

### 1.	Data preprocessing
The chinese dataset are from paper [3]. This dataset in data directory is emotion analysis corpus, with each sample annotated with one emotion label. The label set is like, happiness, sadness, anger, disgust, fear and surprise. The English dataset will use the tweet dataset from my previous teamlab project.

Bert has a very important super parameter: the length of input sequence. It is important to determinate the maximum length in training set and test set. The parameter should be reasonable. There is no running error even if the parameter is too small, but the side effect is that the sequence will be interrupted anomaly, as a result, the predication is not accurate because of missing information.

When the training data is not evenly distributed for each emotion. In order to let Bert model to learn the features on the dataset and reduce the effect of uneven dataset samples, it is necessary to adjust the loss weight for different emotions. Emotions with more samples have smaller weight. Emotions with few samples have greater weight.
Validation set is necessary in experiment. If there is no validation set, it is feasible to divide a validation set from training set. Validation set is used to evaluate the F1 score during the model training process, the models with the highest F1 score models are saved to the local. During the predication period, I can load the checkpoint with good F1 score.

### 2.	Baseline model

The baseline model uses the traditional machine learning algorithms. First use TF-IDF to extract 2000 feature words, and then each text is expressed as a frequent vector by these 2000 words. The whole training set and test set are converted into frequent matrix. The dimension is number of samples * number of feature words. Then use Native Bayes classifier to train and predict on the frequent matrix.

### 3.	Bert model

First, the official bert code should be cloned, the address is Bert source code. Then I need mainly focus on run classifier.py. This python file is the interface for Bert classification. The python file should be adjusted to my demand.

The Chinese pre-training model use the Chinese L-12 H-768 A-12 which is a very large scale Chinese corpus by Google search. The English pre-training model use the BERT-base, uncased dataset.

The set of learning rate is critical, if the learning rate is too large, it is not easy to converge to the opinion value. If the learning rate is too small, the convergence is too slow and the efficiency is too poor. It is wise to gradually increase the learning rate to speed up convergence at the beginning of training. When it comes to a threshold, the learning rate should be reduced gradually.
 
The gradient changes slow down and the model will come to a local optimal solution.

## Reference

[1]	Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Asso- ciation for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.

[2]	Yoon Kim. Convolutional neural networks for sentence classification. In Pro- ceedings of the 2014 Conference on Empirical Methods in Natural Language Pro- cessing (EMNLP), pages 1746–1751, Doha, Qatar, October 2014. Association for Computational Linguistics.

[3]	Minglei Li,   Yunfei Long,   Qin Lu,   and Wenjie Li.   Emotion corpus construc- tion based on selection from hashtags. In Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Sara Goggi, Marko Grobelnik, Bente Maegaard, Joseph Mar- iani, H´el`ene Mazo, Asuncio´n Moreno, Jan Odijk, and Stelios Piperidis, editors, Proceedings of the Tenth International Conference on Language Resources and
Evaluation LREC 2016, Portoroˇz, Slovenia, May 23-28, 2016.  European  Lan- guage Resources Association (ELRA), 2016.

[4]	Tomas Mikolov, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781, 2013.

[5]	Jeffrey Pennington, Richard Socher, and Christopher Manning. Glove:  Global vectors for word representation. In Proceedings of the 20
14 Conference on Em- pirical Methods in Natural Language Processing (EMNLP), pages 1532–1543, Doha, Qatar, October 2014. Association for Computational Linguistics.

[6]	Matthew Peters,   Mark   Neumann,   Mohit   Iyyer,   Matt   Gardner,   Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word represen- tations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 2227–2237, New Orleans, Louisiana, June 2018. Association for Computational Linguistics.

[7]	Alec Radford. Improving language understanding by generative pre-training. 2018.

[8]	Duyu Tang, Bing Qin, Xiaocheng Feng, and Ting Liu. Target-dependent senti- ment classification with long short term memory. 12 2015.

[9]	Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, L-  ukasz Kaiser, and Illia Polosukhin. Attention is all you need.
In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach,  R. Fergus,  S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 5998–6008. Curran Associates, Inc., 2017.





