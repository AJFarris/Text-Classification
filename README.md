
Disclaimer:  I'm not an ml engineer or a professional software engineer...yet ;). This is my first real dive into how machine learning works and my first attempt at making something fairly complex in Go. You will probably find some non-idiomatic, unoptimized code or some miss representation of the math in this document. If you do, I'd love to hear about it so I can update and refine both my understanding and this document!
<H1> Multiclass Text Classification</H1>
<h2>The problem; and a possible solution</h2>
Lets imagine we're archivists who have been given a digital collection with 100000+ articles from early 1800s news papers that we need to sort by topic. If we took the time to read each document before classifying them, our obituaries would be found in the archive before we've labeled half of the articles.

Fortunately for us, Classifying text is literally a subfield of **Natural Language Processing (NLP)**. Instead of reading every document, we can label a few examples of that fit the categories we want. We can then use those as examples to train a statistics model that predicts the subject of a document. 

<H2>To start, we need a (Weighted) Bag o' Words👜</H2>
Last time I checked, computers still don't speak English natively. So we need a computer-friendly way to represent text. For that, we can use something called a **Bag-of-Words** approach.  It starts by making a list of the unique words in the dataset set. Kind of like a **vocabulary** that holds all the words the model knows without any context. Since computers don't understand grammar or sentence structure, we can treat each document like a big bag of unordered words. We can then represent each document by the words in the models vocabulary and the number of times they appear in the given document/bag. Say we have the following vocabulary:


	"where","the","prime","minister","makes","bacon","on","tuesday"

and we get the document: 

	"the prime minister makes bacon"

then our bag-of-words for the document would be:

	{where: 0, the: 1, prime: 1, minister: 1, makes: 1, bacon: 1, on: 0, tuesday: 0}


this is a strong start, but it we have few subtle issues. 

First, words like "and", "was", "is" and so on, will always seem more important to the since they appear more frequently. Secondly, this method doesn't account for a documents length. If two documents contain the word "bacon" 5 times, our model will thinks that "bacon" is equally important in both. But our first document may be a 10 word text message about breakfast, and the second could be a 200 word article about Kevin Bacon's role in the 2023 movie "Toxic Avenger"... "bacon" is not equally as important between these two.  To solve this, we can turn to something called **TF-IDF**. 

**TF-IDF**, or it's less fun government name "**Term Frequency-Inverse Document Frequency**", still uses the number of times a word appears in a document (**TF**), but accounts for how long the document by ratioing it against the length of the document([[Formulas#^13963f]]).  

Then comes the **IDF**. If your a nerd like me, you can see the formula here => ([[Formulas#^0c25a2]]). Suffice to say,  **IDF** is a rarity score. Words that appear less frequently in the whole dataset, get a higher score then words that appear more often. If we multiply a words **TF** and it's rarity score we get a value that represents how much a word "impacts" the overall meaning of a document that we can use in our **Bag-of-Words**.

<h2> Always tell me the odds! 🔢</h2>
"But context is a huge part of knowing a documents subject. How do you classify a document with out the context of word order?", I hear you shouting in my imagination. Enter, **Class Weights**. These are numeric values for words in the vocabulary for every class. If your model has 4 classes, you have 4 sets of of the weights for each word. These weights represents how much the model "thinks" a word pushes a word towards or away from each subject. The model then uses those weights, the document's weights, and a **Class Bias** to score documents in every class. It then converts those scores to probabilities and picks the class that it finds to be the most probable.

While there is a bit more to it, this is the gist of what the model doing when it attempts to classify a document. To get deeper into it, we would need to dig into the math. But just look at this formula for how the model scores documents:
$$
\text{logit}_k = (\sum_{i=1}^{d} W_{k,i} \, x_i) + b_k
$$
...its gross.
<h2> Smarter Every Document 📃</h2>
If we were to set static values for these weights it take *forever*. Also, the model wouldn't be able to learn new patterns or understand how words that might appear in two subjects can be distinguished. To fix that, we need a way for the model to update the class weights based on how far off it's predictions were. Thankfully, this process is pretty simple:

1) The model reads a labeled document, calculates its scores, and makes a prediction
2) For each class score, it calculates how far each score is from what we expected (1 for the correct class, 0 for every other class). This is called the <b>Loss Gradient</b>.
3) That gradient is used in along side a **learning rate** to updates the weights in every class for every word in the document.

This **learning rate** is like a volume nob. If the model is over correcting when it makes the wrong prediction, we can decrease the learning rate and lower the amount the model adjusts it's weights. Conversely, if the model is learning to slowly, we can increase the learning rate and make updates to the weights drastic. That gives us the tools needed to train the model. We simply


<h2>Limitations</h2>
As you might expect, this isn't perfect. For starters, the model can only reason about words that exists in it's vocabulary. While that works for whatever dataset it is trained on. If you were to feed it a document that has words not found in the dataset, the model is likely to get its classification wrong because it dosen't know how to value those words. There are some ways around this, namely something like a contious learning pipeline that will update the vocabulary as the model sees new words. But that is outside the scope of what we are trying to accomplish here. Hopefully, though, you got 