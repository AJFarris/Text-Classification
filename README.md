
Disclaimer:  I'm not an ML engineer or a professional software engineer...yet ;). This is my first dive into how machine learning works and my first attempt at making something fairly complex in Go. You will probably find some non-idiomatic, unoptimized code or some misrepresentation of the math in this document. If you do, I'd love to hear about it so I can update and refine both my understanding and this document!
<H1> Multiclass Text Classification</H1>
<h2>The problem; and a possible solution</h2>
Let's imagine we're archivists who have been given a digital collection with 100000+ articles from early 1800s newspapers that we need to sort by topic. If we took the time to read each document before classifying them, our obituaries would be  in the archive before we've labeled half of the articles.

Fortunately for us, classifying text is literally a subfield of **Natural Language Processing (NLP)**. Instead of reading every document, we can label a few examples that fit the categories we want. We can then use those as examples to train a statistics model to predict the subject of a document. 

<H2>To start, we need a (Weighted) Bag o' Words👜</H2>
Last time I checked, computers still don't speak English natively. So we need a computer-friendly way to represent text. For that, we can use a **Bag-of-Words** approach.  It starts by making a list of the unique words in the dataset. Consider it a **vocabulary** that holds all the words the model knows without any context. Since computers don't understand grammar or sentence structure, we can treat each document like a big bag of unordered words. We can then represent each document by the words in the model's vocabulary and the number of times they appear in the given document/bag. Say we have the following vocabulary:
z

	"where","the","prime","minister","makes","bacon","on","tuesday"

And we get the document: 

	"the prime minister makes bacon"

Then our bag-of-words for the document would be:

	{where: 0, the: 1, prime: 1, minister: 1, makes: 1, bacon: 1, on: 0, tuesday: 0}


This is a strong start, but there are a few subtle issues. 

First, words like "and", "was", "is", and so on, will always seem more important to the model since they appear more frequently in documents of every category. Secondly, this method doesn't account for a document's length. If two documents use the word "bacon" 5 times, our model will see "bacon" as equally important to both. But our first document may be a 10-word text message about breakfast, and the second could be a 200-word article about Kevin Bacon's role in the 2023 movie "Toxic Avenger"... "bacon" is not equally as important in both cases.  

To address this, we can use **TF-IDF**, or its less fun government name "**Term Frequency-Inverse Document Frequency**". This value still uses the number of times a word appears in a document (**TF**). However, it accounts for the document's length by dividing it by the document's length.  

Then comes the **IDF**. If you're a nerd like me, you can see the formulas page for a full breakdown of the equation. Suffice to say,  **IDF** is a rarity score. Words that appear less frequently in the dataset get a higher score than words that appear more often. If we multiply a word's **TF** and its rarity score, we get a value that represents how much a word "impacts" the overall meaning of a document that we can use in our **Bag-of-Words**.

<h2> Always tell me the odds! 🔢</h2>
"But order is a huge part of the context that defines a document's topic. How do you classify a document without the context of word order?", I hear you shouting in my imagination. Enter, **Class Weights**. These are numeric values for words in the vocabulary for every class. If your model has 4 classes, you have 4 sets of weights for each word. These weights represent how much the model "thinks" a word pushes a document towards or away from each subject. The model then uses those weights, the document's weights, and a **Class Bias** to score documents in every class. It then converts those scores into probabilities and selects the class with the highest probability.

<h2> Smarter Every Document 📃</h2>
If we were to set static values for these weights, it would take *forever*. Also, the model wouldn't learn new patterns that indicate the overall subject. Instead, we need a way for the model to update the class weights for each word after it has predicted a class. Thankfully, this process is pretty simple:

1) The model reads a labeled document, calculates its scores, and makes a prediction
2) For each class score, it calculates how far each score is from what we expected (1 for the correct class, 0 for every other class). This is called the <b>Loss Gradient</b>.
3) That gradient is used alongside a **learning rate** to update the weights in every class for every word in the document.


The **learning rate** is like a volume nob. If the model is over-correcting for inaccurate prediction, we can decrease the learning rate  to reduce the amount the weights change each time the model updates. Conversely, if the model is learning to slowly, we can increase the learning rate and make updates to the weights drastic. That gives us the tools needed to train the model.

From here, training the model is a breeze. We divide our labeled examples into a test set and a training set. Then we train the model by having it predict values for the documents and adjust its weights accordingly. We confirm that the model is accurate by giving it the test set and analyzing how accurate its predictions are. Once the model reaches the desired accuracy, it can then be used to classify the rest of the documents in the entire dataset.

