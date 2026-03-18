Formulars (and a quick summary for each)
1) Term Frequency (TF): ^13963f

$$
TF(w,d) = \dfrac{f(w,d)}{\sum_{w_\in d}(w, d)}
$$
Summary:
	Dividing the number of times # of times a word appears in a document (${f(w,d)}$) by the total number of words in the document ($({\sum_{w_\in d}(w, d)}$). This gives a normalized measure of how often a word appears per document relative to the length of the document


2) Document Frequency (Code Name: BuildDF):
$$
 \text{df}(w, D) = \lvert \{\, d \in D \mid w \in d \,\} \rvert
 $$
Summary:
	 The number of documents ($d \in D$) that contain a given word 
	 ($w \in d$)^d99635
 
3) Inverse Document Frequency(Code Name: BuildIDF) :
	$$
IDF(w) = \log({\frac{N+1}{DF(w)+1})+1}
$$
Summary:
	Take the total number of documents ($N$), divide by the **Document Frequency** of the word($DF(w)$), apply a smoothing ($+1$), and take the logarithm. This give a numeric value to how informative a given word is in the dataset^982c32
 ^0c25a2
4) Logistic Regression (Code Name: CalcLogits ):
   $$
	\text{logit}_k = (\sum_{i=1}^{d} W_{k,i} \, x_i) + b_k
	$$
Summary:
	 For each class compute a logit( or "score") by summing the product of each TF-IDF value (word) with its corresponding class weight. Then add the class bias($b_k) . This score is the models raw score for the given class 

5) Stabilized SoftMax (Code Name: SoftMax):
	$$
	\text{Let } z_k \text{ be the scores and computer the maximum}: (m = \max_k z_k)
	$$
$$
\text{Compute stabilized logits: }
\tilde{z}_k = z_k - m
$$
$$
\text{Compute denominator: }
D = \sum_{j} e^{\tilde{z}_j}
$$
$$
\text{Softmax probabilities: }
p_k = \frac{e^{\tilde{z}_k}}{D}
	$$
$$
\text{Compute denominator: }
D = \sum_{j} e^{\tilde{z}_j}

\text{Softmax probabilities: }
p_k = \frac{e^{\tilde{z}_k}}{D}
$$
$$
\text{Softmax Probabilites: } p_k = \frac{e^{\tilde{z}_k}}{D}
$$
Summary:
	Subtract the maximum score ($m$) from all logits to stabilized the exponentials. Compute the denominator ($D$) by summing the exponentials of stabilized logits. The probability for class $k$ is the exponential of its stabilized logit divided by the total.  (*Note:* subtracting the maximum logit prevents overflow in the exponential function which keeps the probabilities numerically stable)

6) Loss Gradient (Code Name: LossGradients):

$$
\text{For each class } k:

y_k =
\begin{cases}
1 & \text{if } k = \text{TrueClass} \\
0 & \text{otherwise}
\end{cases}
$$
$$
\text{Weight gradient for feature } i:
\frac{\partial \mathcal{L}}{\partial W_{k,i}}
= (\hat{y}_k - y_k) \, x_i
$$
$$
\text{Bias gradient:}
\frac{\partial \mathcal{L}}{\partial b_k}
= \hat{y}_k - y_k
$$
Summary:
	Define $y_k$ as 1 for the true class and 0 for all others. The gradient for each weight is the difference between the predicted probability $\hat{y}_k$ and the true label ($yk$), multiplied by the TF‑IDF value ($x_i$). The bias gradient is the same expression without the TF‑IDF term.