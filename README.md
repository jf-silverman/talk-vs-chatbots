Project 3: Web APIs & NLP
Joel Silverman
DSI 1129
## Executive Summary ##

**Problem Statement**

Online discussion forums provide important public spaces where the meaning and direction of society are developed and participants from around the world can exchange ideas.  These forums however are not a panacea for social dialogue.  One problem in digital text forums is distinguishing genuine human participants from auto-generated or auto-dispersed dialogue surreptitiously inserted for various malicious intents.  With the increasing rise of artificial intelligence (AI) and it's widely adopted use in natural language processing (think Alexa, Siri, etc.), there is a risk that AI will be used to pollute digital text communications with dialogue that is very difficult to distinguish from real dialogue.  While no AI has yet passed the Turing Test (see: https://en.wikipedia.org/wiki/Turing_test), a simpler but still important test is this:  how well can we discern human dialogue from AI chatbots in a text discussion forum?  This question has wide implications for public trust in institutions, politics, and text-based digital communications.

In this project, natural language processing (NLP) and supervised classification models are utilized to differentiate between human dialogue found on one general discussion subreddit and AI-generated posts found on a special subreddit dedicated to benign AI chatbots.  

**Background**

The idea behind the project was to explore natural language processing (NLP) and supervised classifiers, utilizing text obtained from two separate Subreddits.  The first subreddit, SubSimulatorGPT2 (hereafter, SubSim) was selected to explore the use of AI in NLPs.  SubSim is a collection of chatbots, each of which was trained on a different subreddit (or mix of several) and produces posts in the linguistic style and topic of its parent subreddit(s).  SubSim was developed by Reddit author Disumbrationist using the Generative Pre-trained Transformer 2 (GPT2), an AI which can generate text (among other functions) (see https://www.reddit.com/r/SubSimulatorGPT2/comments/btfhks/what_is_rsubsimulatorgpt2/).  GPT2 is a general-purpose learner who implements a transformer model deep neural network.  It was trained using text gathered from 8 million websites that were each shared on Reddit and received at least 2 upvotes.  Released in 2019, GPT2 is a project of OpenAI, a research laboratory whose stated goal is promoting AI in a way that benefits humanity (see: https://en.wikipedia.org/wiki/GPT-2, https://en.wikipedia.org/wiki/OpenAI).  Casual Conversation (CasCon) was chosen as a comparator because it is broadly open across topics, draws from 1.8 million users and the forum generally only allows text-based posts (as opposed to photos, links, etc.).   

**Data Collection**

Data for the project was collected from 2 Reddit discussion forums (known as subreddits).  For convenience the Pushshift API (developed for pulling Reddit posts was utilized to download the data.  Since data pulls are limited to 100 records per pull to prevent server request overload, a while loop deployed in Python to pull, aggregate, and transfer JSON data into a Pandas dataframe that could be further explored in Python.  Each consecutive pull in the loop used a "date created" field to determine the starting post for the next pull.  A total of 12,000 posts were pulled from SubSim and 5,000 posts were pulled from CasCon.  Ultimately a relatively even ratio of each subreddit was desired and some initial exploration of the content hinted this 12/5 ratio would lead to 50/50 split in clean data.

**Data Cleaning & EDA**

A simple pursual of SubSim posts is fascinating in that many posts, especially from bots trained on a single subreddit appear real, as demonstrated by the dream example in the presentation (the second one is from SubSim).  CasCon covers many subjects, and the dialogue is very coherent, and well, casual.  

Since SubSim is completely auto-generated, it did not require much cleaning besides for removal of posts with no text body.  Of the 12,000 posts, 7,000 only had images or a hyperlink as the body.  These were removed, leaving just under 5000 posts.  CasCon required removal of approximately 1800 posts, ~90% of which had been removed, and the remainder had no text body. Data was checked for nulls and duplicates, and initially an error was found with the data pull and corrected.  

Once cleaned, 4921 SubSim posts and 3232 CasCon posts remained, making a ratio of 60% for the baseline accuracy of the majority class.  This was determined to be close enough to even between the two classes for modeling.  Additional EDA is described in the next section since preprocessing was performed prior to it.  

**Preprocessing & Modeling**

The title and body of each post were combined into a new column called 'doc' (short for document) to be used in the model.  The rational was that both parts probably contained information for classification.  A separate column was also made to find the character length of each document.  From this, the average number of characters per document in each subreddit was calculated (SubSim = 849, CasCon = 689). Next, the Count Vectorizer (CVEC) NLP transformer was used on the CasCon records to assess top 10 words, after the default 'English' stopwords list of common words was removed.  A separate CVEC was run on SubSim to find the top 10 words (after stopword removal).  These lists can be seen in the slides presentation.  About half the words are the same and half are different, between the two.  Neither contained any unexpected words. When stopwords were removed from each, they each contained roughly 8000 tokens.  Stemming and lemmatizing were not used, except a small test of manual stemming by adding "ve" to the stopword list.  "Ve" was initially a top 10 word, presumably from conjunctions such as "I've", which would then be shortened to "I".

After EDA, the first full model included testing both CVEC and the Term Frequency-Inverse Document Frequency (TF-IDF) transformers with a Multinomial Naive Bayes (MNB)classifier.  A pipeline was fit and used in conjunction with a grid search function to test and score several hyperparameters in the transformers.  Due to less experience using a MNB classifier, only the defaults were utilized.  Best models were similar with test data gaining 88% to 89% accuracy. Since CVEC and TF-IDF appeared to perform similarly, only CVEC was used in subsequent models.  Similarly, stopwords were tested above then dropped because they did not perform as well as other similar parameters such as df_max (fraction of common tokens to drop) or df_features (total number of tokens to keep). 

The next full model included the CVEC transformer and Logistic Regression as the classification estimator.  Several hyperparameters in both the transformer and estimator were attempted, but none improved the train or test scores above the default values, which yielded 99% and 90% respectively. Luckily, we were primarily concerned with prediction, not inference here. 

Lastly, a full model was run with the CVEC transformer and Random Forest Classifier as the estimator.  A pipeline was fit and a grid search of hyperparameters of both model components was made, where the transformer hyperparameters searched were like the prior models.  Final model accuracy in train and test were 100% (indicating very overfit) and 86% respectively.  Perhaps with more time, the grid space could yield a better test accuracy.  Also interestingly, this model gave the best specificity (96%) for SubSim, meaning that this might be the best model if the primary goal is to not miss any chat bots in a predictive net.  

**Evaluation**

Overall, the models all performed much better than the baseline accuracy score of 60% found in the proportion of SubSim to CasCon data. Model test accuracy scores ranged from 86% to 90%.  Sensitivity scores reached as high as 96% for SubSim class prediction. If more time was available, a review of logistic regression coefficients might be made to distinguish some important words.  Likewise, review of misclassification could yield more insights.      

**Conclusion and Recommendations**

Even though it can be difficult for humans to discern some chatbot posts in SubSim from human text dialogue in CasCon, it is not difficult for computer models to tell the two subreddits apart.  And in truth, many posts can readily be distinguished by humans between the two, primarily due to grammatical and factual "tell" signs.  Still, the dialogue produced by AI is uncannily like humans.  The newer GP3 and other AI-based chatbots are undoubtedly making progress on mimicking human text dialogue.  However, with the help of classification models, NLPs, (and AI?), data scientists will probably be able to "weed out" malicious AI chatbots found in forums for the time being.

In this subreddit comparison, SubSim is an aggregation of all the linguistic styles and topics across Reddit and as such is much more variable than CasCon.  If time allows for follow-up, it would be quite interesting to compare individual GPT bots with their parent subreddit. This might be a far tougher classification problem in that the bot will use the linguistic signs of only that subreddit.  

Another good source used in developing this project:  https://www.theverge.com/2019/6/6/18655212/reddit-ai-bots-gpt2-openai-text-artificial-intelligence-subreddit
